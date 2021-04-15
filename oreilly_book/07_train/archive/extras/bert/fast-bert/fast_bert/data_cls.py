import pandas as pd
import os
import torch
from pathlib import Path
import pickle
import logging

import shutil

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    CamembertConfig,
    CamembertForSequenceClassification,
    CamembertTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "camembert-base": (CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer)
}


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        if isinstance(label, list):
            self.label = label
        elif label:
            self.label = str(label)
        else:
            self.label = None


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode="classification",
    cls_token_at_end=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    logger=None,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            if logger:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if isinstance(example.label, list):
            label_id = []
            for label in example.label:
                label_id.append(float(label))
        else:
            if example.label is not None:
                label_id = label_map[example.label]
            else:
                label_id = ""

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, filename, size=-1):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, filename, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, filename, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class TextProcessor(DataProcessor):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.labels = None

    def get_train_examples(
        self, filename="train.csv", text_col="text", label_col="label", size=-1
    ):

        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))

            return self._create_examples(
                data_df, "train", text_col=text_col, label_col=label_col
            )
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(
                data_df.sample(size), "train", text_col=text_col, label_col=label_col
            )

    def get_dev_examples(
        self, filename="val.csv", text_col="text", label_col="label", size=-1
    ):

        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(
                data_df, "dev", text_col=text_col, label_col=label_col
            )
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(
                data_df.sample(size), "dev", text_col=text_col, label_col=label_col
            )

    def get_test_examples(
        self, filename="val.csv", text_col="text", label_col="label", size=-1
    ):
        data_df = pd.read_csv(os.path.join(self.data_dir, filename))
        #         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
        if size == -1:
            return self._create_examples(
                data_df, "test", text_col=text_col, label_col=None
            )
        else:
            return self._create_examples(
                data_df.sample(size), "test", text_col=text_col, label_col=None
            )

    def get_labels(self, filename="labels.csv"):
        """See base class."""
        if self.labels is None:
            self.labels = list(
                pd.read_csv(os.path.join(self.label_dir, filename), header=None)[0]
                .astype("str")
                .values
            )
        return self.labels

    def _create_examples(self, df, set_type, text_col, label_col):
        """Creates examples for the training and dev sets."""
        if label_col is None:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index, text_a=str(row[text_col]), label=None
                    ),
                    axis=1,
                )
            )
        else:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index, text_a=str(row[text_col]), label=str(row[label_col])
                    ),
                    axis=1,
                )
            )


class MultiLabelTextProcessor(TextProcessor):
    def _create_examples(self, df, set_type, text_col, label_col):
        def _get_labels(row, label_col):
            if isinstance(label_col, list):
                return list(row[label_col])
            else:
                # create one hot vector of labels
                label_list = self.get_labels()
                labels = [0] * len(label_list)
                labels[label_list.index(row[label_col])] = 1
                return labels

        """Creates examples for the training and dev sets."""
        if label_col is None:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index, text_a=row[text_col], label=[]
                    ),
                    axis=1,
                )
            )
        else:
            return list(
                df.apply(
                    lambda row: InputExample(
                        guid=row.index,
                        text_a=row[text_col],
                        label=_get_labels(row, label_col),
                    ),
                    axis=1,
                )
            )


class BertDataBunch(object):
    def __init__(
        self,
        data_dir,
        label_dir,
        tokenizer,
        train_file="train.csv",
        val_file="val.csv",
        test_data=None,
        label_file="labels.csv",
        text_col="text",
        label_col="label",
        batch_size_per_gpu=16,
        max_seq_length=512,
        multi_gpu=True,
        multi_label=False,
        backend="nccl",
        model_type="bert",
        logger=None,
        clear_cache=False,
        no_cache=False,
    ):

        # just in case someone passes string instead of Path
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if isinstance(label_dir, str):
            label_dir = Path(label_dir)

        if isinstance(tokenizer, str):
            _, _, tokenizer_class = MODEL_CLASSES[model_type]
            # instantiate the new tokeniser object using the tokeniser name
            tokenizer = tokenizer_class.from_pretrained(
                tokenizer, do_lower_case=("uncased" in tokenizer)
            )

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_data = test_data
        self.cache_dir = data_dir / "cache"
        self.max_seq_length = max_seq_length
        self.batch_size_per_gpu = batch_size_per_gpu
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.multi_label = multi_label
        self.n_gpu = 1
        self.no_cache = no_cache
        self.model_type = model_type
        self.output_mode = "classification"
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        if multi_gpu:
            self.n_gpu = torch.cuda.device_count()

        if clear_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

        if multi_label:
            processor = MultiLabelTextProcessor(data_dir, label_dir)
        else:
            processor = TextProcessor(data_dir, label_dir)

        self.labels = processor.get_labels(label_file)

        if train_file:
            # Train DataLoader
            train_examples = None
            cached_features_file = os.path.join(
                self.cache_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    self.model_type,
                    "train",
                    "multi_label" if self.multi_label else "multi_class",
                    str(self.max_seq_length),
                    os.path.basename(train_file),
                ),
            )

            if os.path.exists(cached_features_file) is False or self.no_cache is True:
                train_examples = processor.get_train_examples(
                    train_file, text_col=text_col, label_col=label_col
                )

            train_dataset = self.get_dataset_from_examples(
                train_examples, "train", no_cache=self.no_cache
            )

            self.train_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)
            train_sampler = RandomSampler(train_dataset)
            self.train_dl = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=self.train_batch_size
            )

        if val_file:
            # Validation DataLoader
            val_examples = None
            cached_features_file = os.path.join(
                self.cache_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    self.model_type,
                    "dev",
                    "multi_label" if self.multi_label else "multi_class",
                    str(self.max_seq_length),
                    os.path.basename(val_file),
                ),
            )

            if os.path.exists(cached_features_file) is False:
                val_examples = processor.get_dev_examples(
                    val_file, text_col=text_col, label_col=label_col
                )

            val_dataset = self.get_dataset_from_examples(
                val_examples, "dev", no_cache=self.no_cache
            )

            # no grads necessary, hence double val batch size
            self.val_batch_size = self.batch_size_per_gpu * 2 * max(1, self.n_gpu)
            val_sampler = SequentialSampler(val_dataset)
            self.val_dl = DataLoader(
                val_dataset, sampler=val_sampler, batch_size=self.val_batch_size
            )

        if test_data:
            # Test set loader for predictions
            test_examples = []
            input_data = []

            for index, text in enumerate(test_data):
                test_examples.append(InputExample(index, text))
                input_data.append({"id": index, "text": text})

            test_dataset = self.get_dataset_from_examples(
                test_examples, "test", is_test=True, no_cache=self.no_cache
            )

            self.test_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)
            test_sampler = SequentialSampler(test_dataset)
            self.test_dl = DataLoader(
                test_dataset, sampler=test_sampler, batch_size=self.test_batch_size
            )

    def get_dl_from_texts(self, texts):

        test_examples = []
        input_data = []

        for index, text in enumerate(texts):
            test_examples.append(InputExample(index, text, label=None))
            input_data.append({"id": index, "text": text})

        test_dataset = self.get_dataset_from_examples(
            test_examples, "test", is_test=True, no_cache=True
        )

        test_sampler = SequentialSampler(test_dataset)
        return DataLoader(
            test_dataset, sampler=test_sampler, batch_size=self.batch_size_per_gpu
        )

    def save(self, filename="databunch.pkl"):
        tmp_path = self.data_dir / "tmp"
        tmp_path.mkdir(exist_ok=True)
        with open(str(tmp_path / filename), "wb") as f:
            pickle.dump(self, f)

    def get_dataset_from_examples(
        self, examples, set_type="train", is_test=False, no_cache=False
    ):

        if set_type == "train":
            file_name = self.train_file
        elif set_type == "dev":
            file_name = self.val_file
        elif set_type == "test":
            file_name = (
                "test"
            )  # test is not supposed to be a file - just a list of texts

        cached_features_file = os.path.join(
            self.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                self.model_type,
                set_type,
                "multi_label" if self.multi_label else "multi_class",
                str(self.max_seq_length),
                os.path.basename(file_name),
            ),
        )

        if os.path.exists(cached_features_file) and no_cache is False:
            self.logger.info(
                "Loading features from cached file %s", cached_features_file
            )
            features = torch.load(cached_features_file)
        else:
            # Create tokenized and numericalized features
            features = convert_examples_to_features(
                examples,
                label_list=self.labels,
                max_seq_length=self.max_seq_length,
                tokenizer=self.tokenizer,
                output_mode=self.output_mode,
                # xlnet has a cls token at the end
                cls_token_at_end=bool(self.model_type in ["xlnet"]),
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                # pad on the left for xlnet
                pad_on_left=bool(self.model_type in ["xlnet"]),
                pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
                logger=self.logger,
            )

            # Create folder if it doesn't exist
            if no_cache is False:
                self.cache_dir.mkdir(exist_ok=True)
                self.logger.info(
                    "Saving features into cached file %s", cached_features_file
                )
                torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )

        if is_test is False:  # labels not available for test set
            if self.multi_label:
                all_label_ids = torch.tensor(
                    [f.label_id for f in features], dtype=torch.float
                )
            else:
                all_label_ids = torch.tensor(
                    [f.label_id for f in features], dtype=torch.long
                )

            dataset = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids
            )
        else:
            all_label_ids = []
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        return dataset
