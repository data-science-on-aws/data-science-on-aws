import os
import torch
from pathlib import Path
import numpy as np

from fastprogress.fastprogress import master_bar, progress_bar
from tensorboardX import SummaryWriter

from .learner_util import Learner

from .data_lm import BertLMDataBunch

from transformers import (WEIGHTS_NAME, BertConfig, BertForMaskedLM, RobertaConfig, RobertaForMaskedLM, DistilBertConfig, DistilBertForMaskedLM, CamembertConfig, CamembertForMaskedLM)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForMaskedLM),
    'roberta': (RobertaConfig, RobertaForMaskedLM),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM),
    'camembert-base': (CamembertConfig, CamembertForMaskedLM)
}

class BertLMLearner(Learner):

    @staticmethod
    def from_pretrained_model(dataBunch, pretrained_path, output_dir, metrics, device, logger,
                              multi_gpu=True, is_fp16=True, warmup_steps=0, fp16_opt_level='O1',
                              grad_accumulation_steps=1, max_grad_norm=1.0, adam_epsilon=1e-8,
                              logging_steps=100):

        model_state_dict = None
        model_type = dataBunch.model_type

        config_class, model_class = MODEL_CLASSES[model_type]

        config = config_class.from_pretrained(pretrained_path)
        model = model_class.from_pretrained(pretrained_path, config=config)

        model.to(device)

        return BertLMLearner(dataBunch, model, pretrained_path, output_dir, metrics, device, logger,
                           multi_gpu, is_fp16, warmup_steps, fp16_opt_level, grad_accumulation_steps,
                           max_grad_norm, adam_epsilon, logging_steps)

    # Learner initialiser
    def __init__(self, data: BertLMDataBunch, model: torch.nn.Module, pretrained_model_path, output_dir, metrics, device,logger,
                 multi_gpu=True, is_fp16=True, warmup_steps=0, fp16_opt_level='O1',
                 grad_accumulation_steps=1, max_grad_norm=1.0, adam_epsilon=1e-8, logging_steps=100):

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.data = data
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.metrics = metrics
        self.multi_gpu = multi_gpu
        self.is_fp16 = is_fp16
        self.fp16_opt_level = fp16_opt_level
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.device = device
        self.logger = logger
        self.optimizer = None
        self.n_gpu = 0
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.max_steps = -1
        self.weight_decay = 0.0
        self.model_type = data.model_type

        self.output_dir = output_dir

        if self.multi_gpu:
            self.n_gpu = torch.cuda.device_count()


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

        # Prepare optimiser and schedule
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

                inputs, labels = self.data.mask_tokens(batch)
                cpu_device = torch.device('cpu')

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.model.train()

                outputs = self.model(inputs, masked_lm_labels=labels)
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

                batch.to(cpu_device)
                inputs.to(cpu_device)
                labels.to(cpu_device)
                torch.cuda.empty_cache()

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
                results = self.validate()
                for key, value in results.items():
                    self.logger.info("eval_{} after epoch {}: {}: ".format(key, (epoch + 1), value))

            # Log metrics
            self.logger.info("lr after epoch {}: {}".format((epoch + 1), scheduler.get_lr()[0]))
            self.logger.info("train_loss after epoch {}: {}".format((epoch + 1), epoch_loss/epoch_step))
            self.logger.info("\n")

        tb_writer.close()
        return global_step, tr_loss / global_step

    ### Evaluate the model
    def validate(self):
        self.logger.info("Running evaluation")

        self.logger.info("Num examples = %d", len(self.data.val_dl.dataset))
        self.logger.info("Validation Batch size = %d", self.data.val_batch_size)

        all_logits = None
        all_labels = None


        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps = 0

        preds = None
        out_label_ids = None

        validation_scores = {metric['name']: 0. for metric in self.metrics}

        for step, batch in enumerate(progress_bar(self.data.val_dl)):
            self.model.eval()
            batch = batch.to(self.device)

            with torch.no_grad():
                outputs = self.model(batch, masked_lm_labels=batch)
                tmp_eval_loss = outputs[0]
                eval_loss += tmp_eval_loss.mean().item()

                cpu_device = torch.device('cpu')
                batch.to(cpu_device)
                torch.cuda.empty_cache()

            nb_eval_steps += 1


        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        results = {'loss': eval_loss,
                   'perplexity': float(perplexity)}

        results.update(validation_scores)

        return results

    def save_model(self):

        path = self.output_dir/'model_out'
        path.mkdir(exist_ok=True)

        torch.cuda.empty_cache()
        # Save a trained model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        model_to_save.save_pretrained(path)

        # save the tokenizer
        self.data.tokenizer.save_pretrained(path)
