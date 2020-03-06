import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import collections
import math
from io import open
import json

from .data_qa import BertQADataBunch
from .learner_util import Learner

from fastprogress.fastprogress import master_bar, progress_bar
from tensorboardX import SummaryWriter

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

from transformers import (BertConfig, BertForQuestionAnswering,
                          XLNetConfig, XLNetForQuestionAnswering, 
                          XLMConfig, XLMForQuestionAnswering,
                          DistilBertConfig, DistilBertForQuestionAnswering)

from .utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering),
    'xlm': (XLMConfig, XLMForQuestionAnswering),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering)
}

class BertQALearner(Learner):
    
    @staticmethod
    def from_pretrained_model(dataBunch, pretrained_path, output_dir, device, logger, 
                              multi_gpu=True, is_fp16=True, warmup_steps=0, fp16_opt_level='O1',
                              grad_accumulation_steps=1, max_grad_norm=1.0, adam_epsilon=1e-8, 
                              logging_steps=100):
        
        model_state_dict = None
        model_type = dataBunch.model_type
        
        config_class, model_class = MODEL_CLASSES[model_type]
        
        config = config_class.from_pretrained(pretrained_path)
        model = model_class.from_pretrained(pretrained_path, config=config)  
            
        model.to(device)
        
        return BertQALearner(dataBunch, model, pretrained_path, output_dir, device, logger,
                           multi_gpu, is_fp16, warmup_steps, fp16_opt_level, grad_accumulation_steps, 
                           max_grad_norm, adam_epsilon, logging_steps)

    
    # Learner initialiser
    def __init__(self, data: BertQADataBunch, model: torch.nn.Module, pretrained_model_path, output_dir, device,logger,
                 multi_gpu=True, is_fp16=True, warmup_steps=0, fp16_opt_level='O1',
                 grad_accumulation_steps=1, max_grad_norm=1.0, adam_epsilon=1e-8, logging_steps=100):
        
        super(BertQALearner, self).__init__(data, model, pretrained_model_path, output_dir, device, logger, 
                                            multi_gpu, is_fp16, warmup_steps, fp16_opt_level, grad_accumulation_steps,
                                            max_grad_norm, adam_epsilon, logging_steps)
        
        self.validation_out = self.output_dir/'validation_out'
        self.validation_out.mkdir(exist_ok=True)

    ### Train the model ###    
    def fit(self, epochs, lr, validate=True, schedule_type="warmup_cosine", optimizer_type='lamb'):
        
        tensorboard_dir = self.output_dir/'tensorboard'
        tensorboard_dir.mkdir(exist_ok=True)
        
        # Train the model
        tb_writer = SummaryWriter(tensorboard_dir)

        train_dataloader = self.data.train_dl
        if self.max_steps > 0:
            t_total = self.max_steps
            self.epochs = self.max_steps // len(train_dataloader) // self.grad_accumulation_steps + 1
        else:
            t_total = len(train_dataloader) // self.grad_accumulation_steps * epochs
        
        # Prepare optimiser
        optimizer = self.get_optimizer(lr, optimizer_type=optimizer_type)
        
        # get the base model if its already wrapped around DataParallel
        if hasattr(self.model, 'module'):
            self.model = self.model.module
        
        if self.is_fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError('Please install apex to use fp16 training')
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.fp16_opt_level)
        
        # Get scheduler
        scheduler = self.get_scheduler(optimizer, t_total=t_total, schedule_type=schedule_type)
        
        # Parallelize the model architecture
        if self.multi_gpu == True:
            self.model = torch.nn.DataParallel(self.model)
            
        # Start Training
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataloader.dataset))
        self.logger.info("  Num Epochs = %d", epochs)
        self.logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       self.data.train_batch_size * self.grad_accumulation_steps)
        self.logger.info("  Gradient Accumulation steps = %d", self.grad_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step =  0
        epoch_step = 0
        tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()
        pbar = master_bar(range(epochs))
        
        for epoch in pbar:
            epoch_step = 0
            epoch_loss = 0.0
            for step, batch in enumerate(progress_bar(train_dataloader, parent=pbar)):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':       batch[0],
                          'attention_mask':  batch[1], 
                          'token_type_ids':  None if self.model_type == 'xlm' else batch[2],  
                          'start_positions': batch[3], 
                          'end_positions':   batch[4]}
                
                if self.model_type in ['xlnet', 'xlm']:
                    inputs.update({'cls_index': batch[5],
                                   'p_mask':    batch[6]})
                
                    
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu parallel training
                if self.grad_accumulation_steps > 1:
                    loss = loss / self.grad_accumulation_steps

                if self.is_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                tr_loss += loss.item()
                epoch_loss += loss.item() 
                if (step + 1) % self.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    
                    self.model.zero_grad()
                    global_step += 1
                    epoch_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        if validate:
                            # evaluate model
                            results = self.validate()
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                                self.logger.info("eval_{} after step {}: {}: ".format(key, global_step, value))
                        
                        # Log metrics
                        self.logger.info("lr after step {}: {}".format(global_step, scheduler.get_lr()[0]))
                        self.logger.info("train_loss after step {}: {}".format(global_step, (tr_loss - logging_loss)/self.logging_steps))
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss)/self.logging_steps, global_step)

                        
                        logging_loss = tr_loss
            
            # Evaluate the model after every epoch
            if validate:
                results = self.validate("epoch_{}".format(epoch+1))
                for key, value in results.items():
                    self.logger.info("eval_{} after epoch {}: {}: ".format(key, (epoch + 1), value))
                
            # Log metrics
            self.logger.info("lr after epoch {}: {}".format((epoch + 1), scheduler.get_lr()[0]))
            self.logger.info("train_loss after epoch {}: {}".format((epoch + 1), epoch_loss/epoch_step))  
            self.logger.info("\n")
            
        tb_writer.close()
        return global_step, tr_loss / global_step   
    
    ### Evaluate the model    
    def validate(self, prefix="SQUAD", n_best_size=20, max_answer_length=30, verbose_logging=False, null_score_diff_threshold=0.0):
        self.logger.info("Running evaluation")
        
        self.logger.info("  Num examples = %d", len(self.data.val_dl.dataset))
        self.logger.info("  Batch size = %d", self.data.val_batch_size)
        
        all_logits = None
        all_labels = None
        
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        
        preds = None
        out_label_ids = None
        all_results = []
        for step, batch in enumerate(progress_bar(self.data.val_dl)):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': None if self.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
                         }
                example_indices = batch[3]
                if self.model_type in ['xlnet', 'xlm']:
                    inputs.update({'cls_index': batch[4],
                                   'p_mask':    batch[5]})

                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            
            for i, example_index in enumerate(example_indices):
                eval_feature = self.data.val_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                if self.model_type in ['xlnet', 'xlm']:
                    # XLNet uses a more complex post-processing procedure
                    result = RawResultExtended(unique_id            = unique_id,
                                               start_top_log_probs  = to_list(outputs[0][i]),
                                               start_top_index      = to_list(outputs[1][i]),
                                               end_top_log_probs    = to_list(outputs[2][i]),
                                               end_top_index        = to_list(outputs[3][i]),
                                               cls_logits           = to_list(outputs[4][i]))
                else:
                    result = RawResult(unique_id    = unique_id,
                                       start_logits = to_list(outputs[0][i]),
                                       end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
            
            nb_eval_steps += 1
            
        # Compute predictions
        output_prediction_file = os.path.join(self.validation_out, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(self.validation_out, "nbest_predictions_{}.json".format(prefix))
        if self.data.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.validation_out, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        if self.model_type in ['xlnet', 'xlm']:
            # XLNet uses a more complex post-processing procedure
            write_predictions_extended(self.data.val_examples, self.data.val_features, all_results, n_best_size,
                            max_answer_length, output_prediction_file,
                            output_nbest_file, output_null_log_odds_file, self.data.val_file,
                            self.model.config.start_n_top, self.model.config.end_n_top,
                            self.data.version_2_with_negative, self.data.tokenizer, verbose_logging, self.logger)
        else:
            write_predictions(self.data.val_examples, self.data.val_features, all_results, n_best_size,
                            max_answer_length, self.data.do_lower_case, output_prediction_file,
                            output_nbest_file, output_null_log_odds_file, verbose_logging,
                            self.data.version_2_with_negative, null_score_diff_threshold, self.logger)

        # Evaluate with the official SQuAD script
        evaluate_options = EVAL_OPTS(data_file=self.data.validation_file,
                                     pred_file=output_prediction_file,
                                     na_prob_file=output_null_log_odds_file)
        results = evaluate_on_squad(evaluate_options)
        return results
    
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

RawResultExtended = collections.namedtuple("RawResultExtended",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold, logger):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, logger, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

def write_predictions_extended(all_examples, all_features, all_results, n_best_size,
                                max_answer_length, output_prediction_file,
                                output_nbest_file,
                                output_null_log_odds_file, orig_data_file,
                                start_n_top, end_n_top, version_2_with_negative,
                                tokenizer, verbose_logging, logger):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.
        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index",
        "start_log_prob", "end_log_prob"])

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

    logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            # 
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer.do_lower_case,
                                        logger, verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_log_prob=-1e6,
                end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    with open(orig_data_file, "r", encoding='utf-8') as reader:
        orig_data = json.load(reader)["data"]

    qid_to_has_ans = make_qid_to_has_ans(orig_data)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(orig_data, all_predictions)
    out_eval = {}

    find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw, scores_diff_json, qid_to_has_ans)

    return out_eval

def get_final_text(pred_text, orig_text, do_lower_case, logger, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def to_list(tensor):
    return tensor.detach().cpu().tolist()