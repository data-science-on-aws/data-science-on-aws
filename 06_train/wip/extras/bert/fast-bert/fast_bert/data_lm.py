import re
import html
import logging
import pandas as pd
import os
import random
import torch
from pathlib import Path
import pickle
import shutil

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Dataset,
)
from torch.utils.data.distributed import DistributedSampler
import spacy
from tqdm import tqdm, trange
from fastprogress.fastprogress import master_bar, progress_bar

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
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    CamembertConfig,
    CamembertForSequenceClassification,
    CamembertTokenizer
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "camembert-base": (CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer)
}

# Create text corpus suitable for language model training


def create_corpus(text_list, target_path, logger=None):

#     nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner", "textcat"])

    with open(target_path, "w") as f:
        #  Split sentences for each document
        logger.info("Formatting corpus for {}".format(target_path))
        for text in progress_bar(text_list):

            text = fix_html(text)
            text = replace_multi_newline(text)
            text = spec_add_spaces(text)
            text = rm_useless_spaces(text)
            text = text.strip()

            f.write(text)


#            text_lines = [re.sub(r"\n(\s)*","",str(sent)) for i, sent in enumerate(nlp(str(text)).sents)]
#            text_lines = [text_line for text_line in text_lines if re.search(r'[a-zA-Z]', text_line)]

#            f.write('\n'.join(text_lines))
#            f.write("\n  \n")


def spec_add_spaces(t: str) -> str:
    "Add spaces around / and # in `t`. \n"
    return re.sub(r"([/#\n])", r" \1 ", t)


def rm_useless_spaces(t: str) -> str:
    "Remove multiple spaces in `t`."
    return re.sub(" {2,}", " ", t)


def replace_multi_newline(t: str) -> str:
    return re.sub(r"(\n(\s)*){2,}", "\n", t)


def fix_html(x: str) -> str:
    "List of replacements from html strings in `x`."
    re1 = re.compile(r"  +")
    x = (
        x.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(x))


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, cache_path, logger, block_size=512):
        assert os.path.isfile(file_path)

        if os.path.exists(cache_path):
            logger.info("Loading features from cached file %s", cache_path)
            with open(cache_path, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file %s", file_path)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            while len(tokenized_text) >= block_size:  # Truncate in block of block_size

                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[:block_size]
                    )
                )
                tokenized_text = tokenized_text[block_size:]
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cache_path)
            with open(cache_path, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


# DataBunch object for language models
class BertLMDataBunch(object):
    @staticmethod
    def from_raw_corpus(
        data_dir,
        text_list,
        tokenizer,
        batch_size_per_gpu=32,
        max_seq_length=512,
        multi_gpu=True,
        test_size=0.1,
        model_type="bert",
        logger=None,
        clear_cache=False,
        no_cache=False,
    ):

        train_file = "lm_train.txt"
        val_file = "lm_val.txt"

        train_list, val_list = train_test_split(
            text_list, test_size=test_size, shuffle=True
        )
        # Create train corpus
        create_corpus(train_list, str(data_dir / train_file), logger=logger)

        # Create val corpus
        create_corpus(val_list, str(data_dir / val_file), logger=logger)

        return BertLMDataBunch(
            data_dir,
            tokenizer,
            train_file=train_file,
            val_file=val_file,
            batch_size_per_gpu=batch_size_per_gpu,
            max_seq_length=max_seq_length,
            multi_gpu=multi_gpu,
            model_type=model_type,
            logger=logger,
            clear_cache=clear_cache,
            no_cache=no_cache,
        )

    def __init__(
        self,
        data_dir,
        tokenizer,
        train_file="lm_train.txt",
        val_file="lm_val.txt",
        batch_size_per_gpu=32,
        max_seq_length=512,
        multi_gpu=True,
        model_type="bert",
        logger=None,
        clear_cache=False,
        no_cache=False,
    ):

        # just in case someone passes string instead of Path
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        # Instantiate correct tokenizer if the tokenizer name is passed instead of object
        if isinstance(tokenizer, str):
            _, _, tokenizer_class = MODEL_CLASSES[model_type]
            # instantiate the new tokeniser object using the tokeniser name
            tokenizer = tokenizer_class.from_pretrained(
                tokenizer, do_lower_case=("uncased" in tokenizer)
            )

        # Bug workaround for RoBERTa
        if model_type == "roberta":
            tokenizer.max_len_single_sentence = tokenizer.max_len - 2

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size_per_gpu = batch_size_per_gpu
        self.train_dl = None
        self.val_dl = None
        self.data_dir = data_dir
        self.cache_dir = data_dir / "lm_cache"
        self.no_cache = no_cache
        self.model_type = model_type
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.n_gpu = 1
        if multi_gpu:
            self.n_gpu = torch.cuda.device_count()

        if clear_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

        # Create folder if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)

        if train_file:
            # Train DataLoader
            train_examples = None
            cached_features_file = os.path.join(
                self.cache_dir,
                "cached_{}_{}_{}".format(
                    self.model_type, "train", str(self.max_seq_length)
                ),
            )

            train_filepath = str(self.data_dir / train_file)
            train_dataset = TextDataset(
                self.tokenizer,
                train_filepath,
                cached_features_file,
                self.logger,
                block_size=self.tokenizer.max_len_single_sentence,
            )

            self.train_batch_size = self.batch_size_per_gpu * max(1, self.n_gpu)

            train_sampler = RandomSampler(train_dataset)
            self.train_dl = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=self.train_batch_size
            )

        if val_file:
            # Val DataLoader
            val_examples = None
            cached_features_file = os.path.join(
                self.cache_dir,
                "cached_{}_{}_{}".format(
                    self.model_type, "dev", str(self.max_seq_length)
                ),
            )

            val_filepath = str(self.data_dir / val_file)
            val_dataset = TextDataset(
                self.tokenizer,
                val_filepath,
                cached_features_file,
                self.logger,
                block_size=self.tokenizer.max_len_single_sentence,
            )

            self.val_batch_size = self.batch_size_per_gpu * 2 * max(1, self.n_gpu)

            val_sampler = RandomSampler(val_dataset)
            self.val_dl = DataLoader(
                val_dataset, sampler=val_sampler, batch_size=self.val_batch_size
            )

    # Mask tokens

    def mask_tokens(self, inputs, mlm_probability=0.15):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)

        masked_indices = torch.bernoulli(
            torch.full(labels.shape, mlm_probability)
        ).bool()
        # do not mask special tokens
        masked_indices[:, 0] = False
        masked_indices[:, -1] = False

        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def save(self, filename="databunch.pkl"):
        tmp_path = self.data_dir / "tmp"
        tmp_path.mkdir(exist_ok=True)
        with open(str(tmp_path / filename), "wb") as f:
            pickle.dump(self, f)

