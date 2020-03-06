import pandas as pd
import os
import torch
from pathlib import Path
import pickle

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer)
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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        try:
            tokens_a = tokenizer.tokenize(example.text_a)
        except:
            print("Cannot tokenise item {}, Text:{}".format(
                ex_index, example.text_a))

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
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if isinstance(example.label, list):
            label_id = []
            for label in example.label:
                label_id.append(float(label))
        else:
            if example.label != None:
                label_id = label_map[example.label]
            else:
                label_id = ''

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
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


class NERTextProcessor(DataProcessor):

    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.labels = None

    def get_train_examples(self, filename='train.txt'):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(self.read_col_file(os.path.join(self.data_dir, filename)), "train")

    def get_dev_examples(self, filename='val.txt', size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(self.read_col_file(os.path.join(self.data_dir, filename)), "val")

    def get_test_examples(self, filename='test.txt', size=-1):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(self.read_col_file(os.path.join(self.data_dir, filename)), "test")

    def get_labels(self, filename='labels.csv'):
        """See base class."""
        if self.labels == None:
            self.labels = list(pd.read_csv(os.path.join(
                self.label_dir, filename), header=None)[0].astype('str').values)
        return self.labels

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def read_col_file(self, filename):
        '''
        read file
        return format :
        [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], 
        ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
        '''
        f = open(filename)
        data = []
        sentence = []
        label = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data


class TextProcessor(DataProcessor):

    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.labels = None

    def get_train_examples(self, filename='train.csv', text_col='text', label_col='label', size=-1):

        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))

            return self._create_examples(data_df, "train", text_col=text_col, label_col=label_col)
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
#             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df.sample(size), "train", text_col=text_col, label_col=label_col)

    def get_dev_examples(self, filename='val.csv', text_col='text', label_col='label', size=-1):

        if size == -1:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(data_df, "dev", text_col=text_col, label_col=label_col)
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, filename))
            return self._create_examples(data_df.sample(size), "dev", text_col=text_col, label_col=label_col)

    def get_test_examples(self, filename='val.csv', text_col='text', label_col='label', size=-1):
        data_df = pd.read_csv(os.path.join(self.data_dir, filename))
#         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
        if size == -1:
            return self._create_examples(data_df, "test",  text_col=text_col, label_col=None)
        else:
            return self._create_examples(data_df.sample(size), "test", text_col=text_col, label_col=None)

    def get_labels(self, filename='labels.csv'):
        """See base class."""
        if self.labels == None:
            self.labels = list(pd.read_csv(os.path.join(
                self.label_dir, filename), header=None)[0].astype('str').values)
        return self.labels

    def _create_examples(self, df, set_type, text_col, label_col):
        """Creates examples for the training and dev sets."""
        if label_col == None:
            return list(df.apply(lambda row: InputExample(guid=row.index, text_a=row[text_col], label=None), axis=1))
        else:
            return list(df.apply(lambda row: InputExample(guid=row.index, text_a=row[text_col], label=str(row[label_col])), axis=1))


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
        if label_col == None:
            return list(df.apply(lambda row: InputExample(guid=row.index, text_a=row[text_col], label=[]), axis=1))
        else:
            return list(df.apply(lambda row: InputExample(guid=row.index, text_a=row[text_col],
                                                          label=_get_labels(row, label_col)), axis=1))


class BertDataBunch(object):

    def get_dl_from_texts(self, texts):

        test_examples = []
        input_data = []

        for index, text in enumerate(texts):
            test_examples.append(InputExample(index, text, label=None))
            input_data.append({
                'id': index,
                'text': text
            })
        test_features = convert_examples_to_features(test_examples, label_list=self.labels,
                                                     tokenizer=self.tokenizer, max_seq_length=self.maxlen)

        all_input_ids = torch.tensor(
            [f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in test_features], dtype=torch.long)

        test_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)

        test_sampler = SequentialSampler(test_data)
        return DataLoader(test_data, sampler=test_sampler, batch_size=self.bs)

    def save(self, filename="databunch.pkl"):
        tmp_path = self.data_dir/'tmp'
        tmp_path.mkdir(exist_ok=True)
        with open(str(tmp_path/filename), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(data_dir, backend='nccl', filename="databunch.pkl"):

        try:
            torch.distributed.init_process_group(backend=backend,
                                                 init_method="tcp://localhost:23459",
                                                 rank=0, world_size=1)
        except:
            pass

        tmp_path = data_dir/'tmp'
        with open(str(tmp_path/filename), "rb") as f:
            databunch = pickle.load(f)

        return databunch

    def __init__(self, data_dir, label_dir, tokenizer, train_file='train.csv', val_file='val.csv', test_data=None,
                 label_file='labels.csv', text_col='text', label_col='label', bs=32, maxlen=512,
                 multi_gpu=True, multi_label=False, backend="nccl", model_type='bert'):
        
        if isinstance(tokenizer, str):
            _,_,tokenizer_class = MODEL_CLASSES[model_type]
            # instantiate the new tokeniser object using the tokeniser name
            tokenizer = tokenizer_class.from_pretrained(tokenizer, do_lower_case=('uncased' in tokenizer))

        self.tokenizer = tokenizer  
        self.data_dir = data_dir
        self.maxlen = maxlen
        self.bs = bs
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        self.multi_label = multi_label
        self.n_gpu = 0
        if multi_gpu:
            self.n_gpu = torch.cuda.device_count()

        if multi_label:
            processor = MultiLabelTextProcessor(data_dir, label_dir)
        else:
            processor = TextProcessor(data_dir, label_dir)

        self.labels = processor.get_labels(label_file)

        if train_file:
            # Train DataLoader
            train_examples = processor.get_train_examples(
                train_file, text_col=text_col, label_col=label_col)
            train_features = convert_examples_to_features(train_examples, label_list=self.labels,
                                                          tokenizer=tokenizer, max_seq_length=maxlen)

            all_input_ids = torch.tensor(
                [f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor(
                [f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in train_features], dtype=torch.long)
            if multi_label:
                all_label_ids = torch.tensor(
                    [f.label_id for f in train_features], dtype=torch.float)
            else:
                all_label_ids = torch.tensor(
                    [f.label_id for f in train_features], dtype=torch.long)

            train_data = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            train_batch_size = bs * max(1, self.n_gpu)

            if multi_gpu:
                train_sampler = RandomSampler(train_data)
            else:
                try:
#                    torch.distributed.init_process_group(backend='nccl')
                    torch.distributed.init_process_group(backend=backend,
                                                         init_method="tcp://localhost:23459",
                                                         rank=0, world_size=1)
                except:
                    pass
                # torch.distributed.init_process_group(backend='nccl')
                train_sampler = DistributedSampler(train_data)
            self.train_dl = DataLoader(
                train_data, sampler=train_sampler, batch_size=train_batch_size)

        if val_file:
            # Validation DataLoader
            val_examples = processor.get_dev_examples(
                val_file, text_col=text_col, label_col=label_col)
            val_features = convert_examples_to_features(val_examples, label_list=self.labels,
                                                        tokenizer=tokenizer, max_seq_length=maxlen)

            all_input_ids = torch.tensor(
                [f.input_ids for f in val_features], dtype=torch.long)
            all_input_mask = torch.tensor(
                [f.input_mask for f in val_features], dtype=torch.long)
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in val_features], dtype=torch.long)
            if multi_label:
                all_label_ids = torch.tensor(
                    [f.label_id for f in val_features], dtype=torch.float)
            else:
                all_label_ids = torch.tensor(
                    [f.label_id for f in val_features], dtype=torch.long)

            val_data = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            val_batch_size = bs * max(1, self.n_gpu)
            if multi_gpu:
                val_sampler = RandomSampler(val_data)
            else:
                try:
#                    torch.distributed.init_process_group(backend=backend)
                    torch.distributed.init_process_group(backend=backend,
                                                         init_method="tcp://localhost:23459",
                                                         rank=0, world_size=1)
                    
                except:
                    pass

                val_sampler = DistributedSampler(val_data)

            self.val_dl = DataLoader(
                val_data, sampler=val_sampler, batch_size=val_batch_size)

        if test_data:
            test_examples = []
            input_data = []

            for index, text in enumerate(test_data):
                test_examples.append(InputExample(index, text))
                input_data.append({
                    'id': index,
                    'text': text
                })

            test_features = convert_examples_to_features(test_examples, label_list=self.labels,
                                                         tokenizer=tokenizer, max_seq_length=maxlen)
            all_input_ids = torch.tensor(
                [f.input_ids for f in test_features], dtype=torch.long)
            all_input_mask = torch.tensor(
                [f.input_mask for f in test_features], dtype=torch.long)
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in test_features], dtype=torch.long)

            test_data = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids)

            test_sampler = SequentialSampler(test_data)
            self.test_dl = DataLoader(
                test_data, sampler=test_sampler, batch_size=bs)
