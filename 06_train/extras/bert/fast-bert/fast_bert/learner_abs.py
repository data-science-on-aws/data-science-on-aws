import os
from .data_abs import BertAbsDataBunch
from .learner_util import Learner
from torch import nn
from typing import List
import torch
from box import Box
from tokenizers import BertWordPieceTokenizer

from .summarisation import BertAbs, build_predictor
from .summarisation import BertAbsConfig
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np
import pandas as pd


from pathlib import Path


MODEL_CLASSES = {"bert": (BertAbsConfig, BertAbs)}


class BertAbsLearner(Learner):
    @staticmethod
    def from_pretrained_model(
        databunch,
        pretrained_path,
        device,
        logger,
        metrics=None,
        finetuned_wgts_path=None,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
    ):

        model_state_dict = None

        model_type = databunch.model_type

        config_class, model_class = MODEL_CLASSES[model_type]

        if finetuned_wgts_path:
            model_state_dict = torch.load(finetuned_wgts_path)
        else:
            model_state_dict = None

        model = model_class.from_pretrained(
            str(pretrained_path), state_dict=model_state_dict
        )

        model.to(device)

        return BertAbsLearner(
            databunch,
            model,
            str(pretrained_path),
            device,
            logger,
            metrics,
            multi_gpu,
            is_fp16,
            loss_scale,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
        )

    def __init__(
        self,
        data: BertAbsDataBunch,
        model: nn.Module,
        pretrained_model_path,
        device,
        logger,
        metrics=None,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
        alpha=0.95,
        beam_size=5,
        min_length=50,
        max_length=200,
        block_trigram=True,
    ):

        super(BertAbsLearner, self).__init__(
            data,
            model,
            pretrained_model_path,
            None,
            device,
            logger,
            multi_gpu,
            is_fp16,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
        )

        # Classification specific attributes
        self.metrics = metrics

        # Summarisation specific features
        if type(self.data.tokenizer) == BertWordPieceTokenizer:
            symbols = {
                "BOS": self.data.tokenizer.token_to_id("[unused0]"),
                "EOS": self.data.tokenizer.token_to_id("[unused1]"),
                "PAD": self.data.tokenizer.token_to_id("[PAD]"),
            }
        else:
            symbols = {
                "BOS": self.data.tokenizer.vocab["[unused0]"],
                "EOS": self.data.tokenizer.vocab["[unused1]"],
                "PAD": self.data.tokenizer.vocab["[PAD]"],
            }

        self.predictor_args = Box(
            {
                "alpha": alpha,
                "beam_size": beam_size,
                "min_length": min_length,
                "max_length": max_length,
                "block_trigram": block_trigram,
            }
        )

        # predictor object
        self.predictor = build_predictor(
            self.predictor_args, self.data.tokenizer, symbols, self.model
        )

    ### Train the model ###
    def fit(
        self,
        epochs,
        lr,
        validate=True,
        schedule_type="warmup_cosine",
        optimizer_type="lamb",
    ):
        self.logger.info(
            "Irony...fit is not implmented yet. This is a pretrained-only inference model"
        )

    ### Evaluate the model
    def validate(self):
        self.logger.info(
            "Irony...fit is not implmented yet. This is a pretrained-only inference model"
        )

    ### Return Predictions ###
    def predict_batch(self, texts=None):

        if texts:
            dl = self.data.get_dl_from_texts(texts)
        else:
            dl = self.data.test_dl

        all_summaries = []

        self.model.eval()
        for step, batch in enumerate(dl):
            # batch = tuple(t.to(self.device) for t in batch)

            batch_data = self.predictor.translate_batch(batch)
            translations = self.predictor.from_batch(batch_data)

            summaries = [format_summary(t) for t in translations]
            all_summaries.extend(summaries)

        return all_summaries


def format_summary(translation):
    """ Transforms the output of the `from_batch` function
    into nicely formatted summaries.
    """
    raw_summary, _, _ = translation
    summary = (
        raw_summary.replace("[unused0]", "")
        .replace("[unused3]", "")
        .replace("[PAD]", "")
        .replace("[unused1]", "")
        .replace(r" +", " ")
        .replace(" [unused2] ", ". ")
        .replace("[unused2]", "")
        .strip()
    )

    return summary
