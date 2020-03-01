import os
import torch
from transformers import BertTokenizer
from .data_cls import BertDataBunch
from .learner_cls import BertLearner
from .modeling import (
    BertForMultiLabelSequenceClassification,
    XLNetForMultiLabelSequenceClassification,
    RobertaForMultiLabelSequenceClassification,
    DistilBertForMultiLabelSequenceClassification,
    CamembertForMultiLabelSequenceClassification,
    AlbertForMultiLabelSequenceClassification,
)

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

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

MODEL_CLASSES = {
    "bert": (
        BertConfig,
        (BertForSequenceClassification, BertForMultiLabelSequenceClassification),
        BertTokenizer,
    ),
    "xlnet": (
        XLNetConfig,
        (XLNetForSequenceClassification, XLNetForMultiLabelSequenceClassification),
        XLNetTokenizer,
    ),
    "xlm": (
        XLMConfig,
        (XLMForSequenceClassification, XLMForSequenceClassification),
        XLMTokenizer,
    ),
    "roberta": (
        RobertaConfig,
        (RobertaForSequenceClassification, RobertaForMultiLabelSequenceClassification),
        RobertaTokenizer,
    ),
    "distilbert": (
        DistilBertConfig,
        (
            DistilBertForSequenceClassification,
            DistilBertForMultiLabelSequenceClassification,
        ),
        DistilBertTokenizer,
    ),
    "albert": (
        AlbertConfig,
        (AlbertForSequenceClassification, AlbertForMultiLabelSequenceClassification),
        AlbertTokenizer,
    ),
    "camembert-base": (
        CamembertConfig,
        (
            CamembertForSequenceClassification,
            CamembertForMultiLabelSequenceClassification,
        ),
        CamembertTokenizer,
    ),
}


class BertClassificationPredictor(object):
    def __init__(
        self,
        model_path,
        label_path,
        multi_label=False,
        model_type="bert",
        do_lower_case=True,
    ):
        self.model_path = model_path
        self.label_path = label_path
        self.multi_label = multi_label
        self.model_type = model_type
        self.do_lower_case = do_lower_case

        self.learner = self.get_learner()

    def get_learner(self):

        _, _, tokenizer_class = MODEL_CLASSES[self.model_type]
        # instantiate the new tokeniser object using the tokeniser name
        tokenizer = tokenizer_class.from_pretrained(
            self.model_path, do_lower_case=self.do_lower_case
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        databunch = BertDataBunch(
            self.label_path,
            self.label_path,
            tokenizer,
            train_file=None,
            val_file=None,
            batch_size_per_gpu=32,
            max_seq_length=512,
            multi_gpu=False,
            multi_label=self.multi_label,
            model_type=self.model_type,
            no_cache=True,
        )

        learner = BertLearner.from_pretrained_model(
            databunch,
            self.model_path,
            metrics=[],
            device=device,
            logger=None,
            output_dir=None,
            warmup_steps=0,
            multi_gpu=False,
            is_fp16=False,
            multi_label=self.multi_label,
            logging_steps=0,
        )

        return learner

    def predict_batch(self, texts):
        return self.learner.predict_batch(texts)

    def predict(self, text):
        predictions = self.predict_batch([text])[0]
        return predictions
