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
from collections import deque, namedtuple
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

Batch = namedtuple(
    "Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"]
)


class SummarizationDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.
    The class will process the documents that are located in the specified
    folder. The preprocessing will work on any document that is reasonably
    formatted. On the CNN/DailyMail dataset it will extract both the story
    and the summary.
    CNN/Daily News:
    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].
    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, path="", prefix="train"):
        """ We initialize the class by listing all the documents to summarize.
        Files are not read in memory due to the size of some datasets (like CNN/DailyMail).
        """
        assert os.path.isdir(path)

        self.documents = []
        filenames_list = os.listdir(path)
        for filename in filenames_list:
            if "summary" in filename:
                continue
            path_to_text = os.path.join(path, filename)
            if not os.path.isfile(path_to_text):
                continue
            self.documents.append(path_to_text)

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.documents)

    def __getitem__(self, idx):
        document_path = self.documents[idx]
        document_name = document_path.split("/")[-1]
        with open(document_path, encoding="utf-8") as source:
            raw_doc = source.read()
            doc_lines = process_document(raw_doc)
        return document_name, doc_lines, []


class SummarizationInMemoryDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.
    The class will process the documents that are located in the specified
    folder. The preprocessing will work on any document that is reasonably
    formatted. On the CNN/DailyMail dataset it will extract both the story
    and the summary.
    CNN/Daily News:
    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].
    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, texts=[]):
        """ We initialize the class by listing all the documents to summarize.
        Files are not read in memory due to the size of some datasets (like CNN/DailyMail).
        """
        self.documents = texts

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.documents)

    def __getitem__(self, idx):
        raw_doc = self.documents[idx]
        doc_lines = process_document(raw_doc)

        return None, doc_lines, []


def process_document(raw_doc):
    """ Extract the story and summary from a story file.
    Attributes:
        raw_story (str): content of the story file as an utf-8 encoded string.
    Raises:
        IndexError: If the stoy is empty or contains no highlights.
    """
    nonempty_lines = list(
        filter(lambda x: len(x) != 0, [line.strip() for line in raw_doc.split("\n")])
    )

    # for some unknown reason some lines miss a period, add it
    nonempty_lines = [_add_missing_period(line) for line in nonempty_lines]

    # gather article lines
    doc_lines = []
    lines = deque(nonempty_lines)
    while True:
        try:
            element = lines.popleft()
            if element.startswith("@highlight"):
                break
            doc_lines.append(element)
        except IndexError:
            # if "@highlight" is absent from the file we pop
            # all elements until there is None, raising an exception.
            return doc_lines

    return doc_lines


def _add_missing_period(line):
    END_TOKENS = [".", "!", "?", "...", "'", "`", '"', u"\u2019", u"\u2019", ")"]
    if line.startswith("@highlight"):
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + "."


# Abstractive databunch
class BertAbsDataBunch(object):
    def __init__(
        self,
        tokenizer,
        device,
        data_dir=None,
        test_data=None,
        batch_size_per_gpu=16,
        max_seq_length=512,
        multi_gpu=True,
        multi_label=False,
        model_type="bert",
        logger=None,
        clear_cache=False,
        no_cache=False,
    ):

        # just in case someone passes string instead of Path
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if isinstance(tokenizer, str):
            # instantiate the new tokeniser object using the tokeniser name
            tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True
            )
        self.tokenizer = tokenizer

        if type(self.tokenizer) == BertWordPieceTokenizer:

            self.tokenizer.cls_token_id = self.tokenizer.token_to_id("[CLS]")
            self.tokenizer.pad_token_id = self.tokenizer.token_to_id("[PAD]")

        self.max_seq_length = max_seq_length
        self.batch_size_per_gpu = batch_size_per_gpu
        self.device = device
        if data_dir:
            self.data_dir = data_dir
            self.cache_dir = data_dir / "lm_cache"
            # Create folder if it doesn't exist
            self.cache_dir.mkdir(exist_ok=True)
            self.no_cache = no_cache
            if clear_cache:
                shutil.rmtree(self.cache_dir, ignore_errors=True)
        else:
            self.no_cache = True
            self.data_dir = None

        self.model_type = model_type
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.n_gpu = 1
        if multi_gpu:
            self.n_gpu = torch.cuda.device_count()

        # get dataset
        if self.data_dir:
            dataset = SummarizationDataset(self.data_dir)
        elif test_data:
            dataset = SummarizationInMemoryDataset(test_data)
        else:
            dataset = None

        if dataset:
            sampler = SequentialSampler(dataset)

            collate_fn = lambda data: collate(
                data, self.tokenizer, block_size=self.max_seq_length, device=self.device
            )

            self.test_dl = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.batch_size_per_gpu,
                collate_fn=collate_fn,
            )
        else:
            self.test_dl = None

    def get_dl_from_texts(self, texts):

        dataset = SummarizationInMemoryDataset(texts)

        sampler = SequentialSampler(dataset)

        collate_fn = lambda data: collate(
            data, self.tokenizer, block_size=self.max_seq_length, device=self.device
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size_per_gpu,
            collate_fn=collate_fn,
        )


def collate(data, tokenizer, block_size, device):
    """ Collate formats the data passed to the data loader.
    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    data = [x for x in data if not len(x[1]) == 0]  # remove empty_files
    names = [name for name, _, _ in data]
    summaries = [" ".join(summary_list) for _, _, summary_list in data]

    if type(tokenizer) == BertWordPieceTokenizer:
        encoded_text = [
            encode_for_summarization_new_tokenizer(story, summary, tokenizer)
            for _, story, summary in data
        ]
    else:
        encoded_text = [
            encode_for_summarization(story, summary, tokenizer)
            for _, story, summary in data
        ]
    encoded_stories = torch.tensor(
        [
            fit_to_block_size(story, block_size, tokenizer.pad_token_id)
            for story, _ in encoded_text
        ]
    )
    encoder_token_type_ids = compute_token_type_ids(
        encoded_stories, tokenizer.cls_token_id
    )
    encoder_mask = build_mask(encoded_stories, tokenizer.pad_token_id)

    batch = Batch(
        document_names=names,
        batch_size=len(encoded_stories),
        src=encoded_stories.to(device),
        segs=encoder_token_type_ids.to(device),
        mask_src=encoder_mask.to(device),
        tgt_str=summaries,
    )

    return batch


def encode_for_summarization(story_lines, summary_lines, tokenizer):
    """ Encode the story and summary lines, and join them
    as specified in [1] by using `[SEP] [CLS]` tokens to separate
    sentences.
    """
    story_lines_token_ids = [tokenizer.encode(line) for line in story_lines]
    story_token_ids = [
        token for sentence in story_lines_token_ids for token in sentence
    ]
    summary_lines_token_ids = [tokenizer.encode(line) for line in summary_lines]
    summary_token_ids = [
        token for sentence in summary_lines_token_ids for token in sentence
    ]

    return story_token_ids, summary_token_ids


def encode_for_summarization_new_tokenizer(story_lines, summary_lines, tokenizer):
    """ Encode the story and summary lines, and join them
    as specified in [1] by using `[SEP] [CLS]` tokens to separate
    sentences.
    """
    story_lines_token_ids = [tokenizer.encode(line).ids for line in story_lines]
    story_token_ids = [
        token for sentence in story_lines_token_ids for token in sentence
    ]
    summary_lines_token_ids = [tokenizer.encode(line).ids for line in summary_lines]
    summary_token_ids = [
        token for sentence in summary_lines_token_ids for token in sentence
    ]

    return story_token_ids, summary_token_ids


def fit_to_block_size(sequence, block_size, pad_token_id):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter we append padding token to the right of the sequence.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token_id] * (block_size - len(sequence)))
        return sequence


def build_mask(sequence, pad_token_id):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1. """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token_id
    mask[idx_pad_tokens] = 0
    return mask


def compute_token_type_ids(batch, separator_token_id):
    """ Segment embeddings as described in [1]
    The values {0,1} were found in the repository [2].
    Attributes:
        batch: torch.Tensor, size [batch_size, block_size]
            Batch of input.
        separator_token_id: int
            The value of the token that separates the segments.
    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    [2] https://github.com/nlpyang/PreSumm (/src/prepro/data_builder.py, commit fac1217)
    """
    batch_embeddings = []
    for sequence in batch:
        sentence_num = -1
        embeddings = []
        for s in sequence:
            if s == separator_token_id:
                sentence_num += 1
            embeddings.append(sentence_num % 2)
        batch_embeddings.append(embeddings)
    return torch.tensor(batch_embeddings)
