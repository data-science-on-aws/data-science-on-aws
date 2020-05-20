import json
import os
import torch
from pathlib import Path
import pickle
import logging
import collections
from io import open
import shutil

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

from transformers import (WEIGHTS_NAME, 
                          BertTokenizer, 
                          XLNetTokenizer, 
                          DistilBertTokenizer)
                          

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'xlnet': XLNetTokenizer,
    'distilbert': DistilBertTokenizer
}

class QAExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s
    
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def read_examples(input_file, is_training, version_2_with_negative, logger):
    """Read a SQuAD json file into a list of QAExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = QAExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training, logger,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, ):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    features = []
    for (example_index, example) in enumerate(examples):
        
        if example_index % 10000 == 0:
            if logger:
                logger.info("Writing example {} of {}".format(example_index, len(examples)))

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index


            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible))
            unique_id += 1

    return features

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

class BertQADataBunch(object):

    def __init__(self, data_dir, tokenizer, train_file='train-v2.0.json', val_file='dev-v2.0.json', test_data=None,
                 batch_size_per_gpu=16, 
                 max_seq_length=512,
                 doc_stride=128,
                 max_query_length=64,
                 version_2_with_negative=True,
                 multi_gpu=True, 
                 model_type='bert', 
                 logger=None, clear_cache=False, no_cache=False):

        # just in case someone passes string instead of Path
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if isinstance(tokenizer, str):
            tokenizer_class = TOKENIZER_CLASSES[model_type]
            # instantiate the new tokeniser object using the tokeniser name
            self.do_lower_case = ('uncased' in tokenizer)
            tokenizer = tokenizer_class.from_pretrained(
                tokenizer, do_lower_case=self.do_lower_case)

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.cache_dir = data_dir/'cache'
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.batch_size_per_gpu = batch_size_per_gpu
        self.version_2_with_negative = version_2_with_negative
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.n_gpu = 1
        self.no_cache = no_cache
        self.model_type = model_type
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        if multi_gpu:
            self.n_gpu = torch.cuda.device_count()

        if clear_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            
        # Create folder if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        if train_file:
            # Train DataLoader
            train_examples = None
            cached_features_file = os.path.join(self.cache_dir, 'cached_{}_{}_{}_{}_{}'.format(
                self.model_type,
                'train',
                str(self.max_seq_length),
                str(self.doc_stride),
                str(self.max_query_length)))
            
            if os.path.exists(cached_features_file) and no_cache == False:
                self.logger.info(
                    "Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
            else:
                training_file = str(self.data_dir/train_file)
                self.logger.info("Creating features from dataset file at %s", training_file)
                examples = read_examples(input_file=training_file,
                                         is_training=True,
                                         version_2_with_negative=self.version_2_with_negative, 
                                         logger=self.logger)
                
                features = convert_examples_to_features(examples=examples,
                                                tokenizer=self.tokenizer,
                                                logger=self.logger,
                                                max_seq_length=self.max_seq_length,
                                                doc_stride=self.doc_stride,
                                                max_query_length=self.max_query_length,
                                                is_training=True)
                
                if self.no_cache == False:
                    self.logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features, cached_features_file)
                
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)
            
            self.train_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)
            train_sampler = RandomSampler(dataset)
            self.train_dl = DataLoader(dataset, sampler=train_sampler, batch_size=self.train_batch_size)
        
        if val_file:
            # Val DataLoader
            train_examples = None
            cached_features_file = os.path.join(self.cache_dir, 'cached_{}_{}_{}_{}_{}'.format(
                self.model_type,
                'dev',
                str(self.max_seq_length),
                str(self.doc_stride),
                str(self.max_query_length)))
            
            if os.path.exists(cached_features_file) and no_cache == False:
                self.validation_file = str(self.data_dir/val_file)
                examples = read_examples(input_file=self.validation_file,
                                         is_training=False,
                                         version_2_with_negative=self.version_2_with_negative, 
                                         logger=self.logger)
                self.logger.info(
                    "Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
            else:
                self.validation_file = str(self.data_dir/val_file)
                self.logger.info("Creating features from dataset file at %s", self.validation_file)
                examples = read_examples(input_file=self.validation_file,
                                         is_training=False,
                                         version_2_with_negative=self.version_2_with_negative, 
                                         logger=self.logger)
                
                features = convert_examples_to_features(examples=examples,
                                                tokenizer=self.tokenizer,
                                                logger=self.logger,
                                                max_seq_length=self.max_seq_length,
                                                doc_stride=self.doc_stride,
                                                max_query_length=self.max_query_length,
                                                is_training=False)
                
                if self.no_cache == False:
                    self.logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features, cached_features_file)
            
            # Save reference to validation features on the databunch object
            self.val_features = features   
            self.val_examples = examples
            
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
            
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_example_index, all_cls_index, all_p_mask)
            
            self.val_batch_size = self.batch_size_per_gpu * 2 * max(1, self.n_gpu)
            val_sampler = SequentialSampler(dataset)
            self.val_dl = DataLoader(dataset, sampler=val_sampler, batch_size=self.train_batch_size)

    def save(self, filename="databunch.pkl"):
        tmp_path = self.data_dir/'tmp'
        tmp_path.mkdir(exist_ok=True)
        with open(str(tmp_path/filename), "wb") as f:
            pickle.dump(self, f)
                