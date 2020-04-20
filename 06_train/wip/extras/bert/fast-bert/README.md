# Fast-Bert

[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/fast-bert.svg)](https://badge.fury.io/py/fast-bert)
![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)

**New Includes Summarisation using BERT Seq2Seq**

**New model architectures: ALBERT, CamemBERT, DistilRoberta**

**DistilBERT (from HuggingFace), Smaller, faster, cheaper, lighter**

**RoBERTa model support added to Fastbert**

**Now supports LAMB optimizer for faster training.**
Please refer to https://arxiv.org/abs/1904.00962 for the paper on LAMB optimizer.

**Now supports BERT and XLNet for both Multi-Class and Multi-Label text classification.**

Fast-Bert is the deep learning library that allows developers and data scientists to train and deploy BERT and XLNet based models for natural language processing tasks beginning with Text Classification.

The work on FastBert is built on solid foundations provided by the excellent [Hugging Face BERT PyTorch library](https://github.com/huggingface/pytorch-pretrained-BERT) and is inspired by [fast.ai](https://github.com/fastai/fastai) and strives to make the cutting edge deep learning technologies accessible for the vast community of machine learning practitioners.

With FastBert, you will be able to:

1. Train (more precisely fine-tune) BERT, RoBERTa and XLNet text classification models on your custom dataset.

2. Tune model hyper-parameters such as epochs, learning rate, batch size, optimiser schedule and more.

3. Save and deploy trained model for inference (including on AWS Sagemaker).

Fast-Bert will support both multi-class and multi-label text classification for the following and in due course, it will support other NLU tasks such as Named Entity Recognition, Question Answering and Custom Corpus fine-tuning.

1.  **[BERT](https://github.com/google-research/bert)** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

2)  **[XLNet](https://github.com/zihangdai/xlnet/)** (from Google/CMU) released with the paper [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.

3)  **[RoBERTa](https://arxiv.org/abs/1907.11692)** (from Facebook), a Robustly Optimized BERT Pretraining Approach by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du et al.

4)  **DistilBERT (from HuggingFace)**, released together with the blogpost [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5) by Victor Sanh, Lysandre Debut and Thomas Wolf.

## Installation

This repo is tested on Python 3.6+.

### With pip

PyTorch-Transformers can be installed by pip as follows:

```bash
pip install fast-bert
```

### From source

Clone the repository and run:

```bash
pip install [--editable] .
```

or

```bash
pip install git+https://github.com/kaushaltrivedi/fast-bert.git
```

You will also need to install NVIDIA Apex.

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Usage

## Text Classification

### 1. Create a DataBunch object

The databunch object takes training, validation and test csv files and converts the data into internal representation for BERT, RoBERTa, DistilBERT or XLNet. The object also instantiates the correct data-loaders based on device profile and batch_size and max_sequence_length.

```python

from fast_bert.data_cls import BertDataBunch

databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='label',
                          batch_size_per_gpu=16,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=False,
                          model_type='bert')
```

#### File format for train.csv and val.csv

| index | text                                                                                                                                                                                                                                                                                                                                | label |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| 0     | Looking through the other comments, I'm amazed that there aren't any warnings to potential viewers of what they have to look forward to when renting this garbage. First off, I rented this thing with the understanding that it was a competently rendered Indiana Jones knock-off.                                                | neg   |
| 1     | I've watched the first 17 episodes and this series is simply amazing! I haven't been this interested in an anime series since Neon Genesis Evangelion. This series is actually based off an h-game, which I'm not sure if it's been done before or not, I haven't played the game, but from what I've heard it follows it very well | pos   |
| 2     | his movie is nothing short of a dark, gritty masterpiece. I may be bias, as the Apartheid era is an area I've always felt for.                                                                                                                                                                                                      | pos   |

In case the column names are different than the usual text and labels, you will have to provide those names in the databunch text_col and label_col parameters.

**labels.csv** will contain a list of all unique labels. In this case the file will contain:

```csv
pos
neg
```

For multi-label classification, **labels.csv** will contain all possible labels:

```toxic
severe_toxic
obscene
threat
insult
identity_hate
```

The file **train.csv** will then contain one column for each label, with each column value being either 0 or 1. Don't forget to change `multi_label=True` for multi-label classification in `BertDataBunch`.

| id  | text                                                                       | toxic | severe_toxic | obscene | threat | insult | identity_hate |
| --- | -------------------------------------------------------------------------- | ----- | ------------ | ------- | ------ | ------ | ------------- |
| 0   | Why the edits made under my username Hardcore Metallica Fan were reverted? | 0     | 0            | 0       | 0      | 0      | 0             |
| 0   | I will mess you up                                                         | 1     | 0            | 0       | 1      | 0      | 0             |

label_col will be a list of label column names. In this case it will be:

```python
['toxic','severe_toxic','obscene','threat','insult','identity_hate']
```

#### Tokenizer

You can either create a tokenizer object and pass it to DataBunch or you can pass the model name as tokenizer and DataBunch will automatically download and instantiate an appropriate tokenizer object.

For example for using XLNet base cased model, set tokenizer parameter to 'xlnet-base-cased'. DataBunch will automatically download and instantiate XLNetTokenizer with the vocabulary for xlnet-base-cased model.

#### Model Type

Fast-Bert supports XLNet, RoBERTa and BERT based classification models. Set model type parameter value to **'bert'**, **roberta** or **'xlnet'** in order to initiate an appropriate databunch object.

### 2. Create a Learner Object

BertLearner is the ‘learner’ object that holds everything together. It encapsulates the key logic for the lifecycle of the model such as training, validation and inference.

The learner object will take the databunch created earlier as as input alongwith some of the other parameters such as location for one of the pretrained models, FP16 training, multi_gpu and multi_label options.

The learner class contains the logic for training loop, validation loop, optimiser strategies and key metrics calculation. This help the developers focus on their custom use-cases without worrying about these repetitive activities.

At the same time the learner object is flexible enough to be customised either via using flexible parameters or by creating a subclass of BertLearner and redefining relevant methods.

```python

from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging

logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=True,
						is_fp16=True,
						multi_label=False,
						logging_steps=50)
```

| parameter           | description                                                                                                                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| databunch           | Databunch object created earlier                                                                                                                                                                                               |
| pretrained_path     | Directory for the location of the pretrained model files or the name of one of the pretrained models i.e. bert-base-uncased, xlnet-large-cased, etc                                                                            |
| metrics             | List of metrics functions that you want the model to calculate on the validation set, e.g. accuracy, beta, etc                                                                                                                 |
| device              | torch.device of type _cuda_ or _cpu_                                                                                                                                                                                           |
| logger              | logger object                                                                                                                                                                                                                  |
| output_dir          | Directory for model to save trained artefacts, tokenizer vocabulary and tensorboard files                                                                                                                                      |
| finetuned_wgts_path | provide the location for fine-tuned language model (experimental feature)                                                                                                                                                      |
| warmup_steps        | number of training warms steps for the scheduler                                                                                                                                                                               |
| multi_gpu           | multiple GPUs available e.g. if running on AWS p3.8xlarge instance                                                                                                                                                             |
| is_fp16             | FP16 training                                                                                                                                                                                                                  |
| multi_label         | multilabel classification                                                                                                                                                                                                      |
| logging_steps       | number of steps between each tensorboard metrics calculation. Set it to 0 to disable tensor flow logging. Keeping this value too low will lower the training speed as model will be evaluated each time the metrics are logged |

### 3. Train the model

```python
learner.fit(epochs=6,
			lr=6e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="lamb")
```

Fast-Bert now supports LAMB optmizer. Due to the speed of training, we have set LAMB as the default optimizer. You can switch back to AdamW by setting optimizer_type to 'adamw'.

### 4. Save trained model artifacts

```python
learner.save_model()
```

Model artefacts will be persisted in the output_dir/'model_out' path provided to the learner object. Following files will be persisted:

| File name               | description                                      |
| ----------------------- | ------------------------------------------------ |
| pytorch_model.bin       | trained model weights                            |
| spiece.model            | sentence tokenizer vocabulary (for xlnet models) |
| vocab.txt               | workpiece tokenizer vocabulary (for bert models) |
| special_tokens_map.json | special tokens mappings                          |
| config.json             | model config                                     |
| added_tokens.json       | list of new tokens                               |

As the model artefacts are all stored in the same folder, you will be able to instantiate the learner object to run inference by pointing pretrained_path to this location.

### 5. Model Inference

If you already have a Learner object with trained model instantiated, just call predict_batch method on the learner object with the list of text data:

```python
texts = ['I really love the Netflix original movies',
		 'this movie is not worth watching']
predictions = learner.predict_batch(texts)
```

If you have persistent trained model and just want to run inference logic on that trained model, use the second approach, i.e. the predictor object.

```python
from fast_bert.prediction import BertClassificationPredictor

MODEL_PATH = OUTPUT_DIR/'model_out'

predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path=LABEL_PATH, # location for labels.csv file
				multi_label=False,
				model_type='xlnet',
				do_lower_case=False)

# Single prediction
single_prediction = predictor.predict("just get me result for this text")

# Batch predictions
texts = [
	"this is the first text",
	"this is the second text"
	]

multiple_predictions = predictor.predict_batch(texts)
```

## Language Model Fine-tuning

A useful approach to use BERT based models on custom datasets is to first finetune the language model task for the custom dataset, an apporach followed by fast.ai's ULMFit. The idea is to start with a pre-trained model and further train the model on the raw text of the custom dataset. We will use the masked LM task to finetune the language model.

This section will describe the usage of FastBert to finetune the language model.

### 1. Import the necessary libraries

The necessary objects are stored in the files with '\_lm' suffix.

```python
# Language model Databunch
from fast_bert.data_lm import BertLMDataBunch
# Language model learner
from fast_bert.learner_lm import BertLMLearner

from pathlib import Path
from box import Box
```

### 2. Define parameters and setup datapaths

```python
# Box is a nice wrapper to create an object from a json dict
args = Box({
    "seed": 42,
    "task_name": 'imdb_reviews_lm',
    "model_name": 'roberta-base',
    "model_type": 'roberta',
    "train_batch_size": 16,
    "learning_rate": 4e-5,
    "num_train_epochs": 20,
    "fp16": True,
    "fp16_opt_level": "O2",
    "warmup_steps": 1000,
    "logging_steps": 0,
    "max_seq_length": 512,
    "multi_gpu": True if torch.cuda.device_count() > 1 else False
})

DATA_PATH = Path('../lm_data/')
LOG_PATH = Path('../logs')
MODEL_PATH = Path('../lm_model_{}/'.format(args.model_type))

DATA_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)
LOG_PATH.mkdir(exist_ok=True)


```

### 3. Create DataBunch object

The BertLMDataBunch class contains a static method 'from_raw_corpus' that will take the list of raw texts and create DataBunch for the language model learner.

The method will at first preprocess the text list by removing html tags, extra spaces and more and then create files `lm_train.txt` and `lm_val.txt`. These files will be used for training and evaluating the language model finetuning task.

The next step will be to featurize the texts. The text will be tokenized, numericalized and split into blocks on 512 tokens (including special tokens).

```python
databunch_lm = BertLMDataBunch.from_raw_corpus(
					data_dir=DATA_PATH,
					text_list=texts,
					tokenizer=args.model_name,
					batch_size_per_gpu=args.train_batch_size,
					max_seq_length=args.max_seq_length,
                    multi_gpu=args.multi_gpu,
                    model_type=args.model_type,
                    logger=logger)
```

As this step can take some time based on the size of your custom dataset's text, the featurized data will be cached in pickled files in the data_dir/lm_cache folder.

The next time, instead of using from_raw_corpus method, you may want to directly instantiate the DataBunch object as shown below:

```python
databunch_lm = BertLMDataBunch(
						data_dir=DATA_PATH,
						tokenizer=args.model_name,
                        batch_size_per_gpu=args.train_batch_size,
                        max_seq_length=args.max_seq_length,
                        multi_gpu=args.multi_gpu,
                        model_type=args.model_type,
                        logger=logger)
```

### 4. Create the LM Learner object

BertLearner is the ‘learner’ object that holds everything together. It encapsulates the key logic for the lifecycle of the model such as training, validation and inference.

The learner object will take the databunch created earlier as as input alongwith some of the other parameters such as location for one of the pretrained models, FP16 training, multi_gpu and multi_label options.

The learner class contains the logic for training loop, validation loop, and optimizer strategies. This help the developers focus on their custom use-cases without worrying about these repetitive activities.

At the same time the learner object is flexible enough to be customized either via using flexible parameters or by creating a subclass of BertLearner and redefining relevant methods.

```python
learner = BertLMLearner.from_pretrained_model(
							dataBunch=databunch_lm,
							pretrained_path=args.model_name,
							output_dir=MODEL_PATH,
							metrics=[],
							device=device,
							logger=logger,
							multi_gpu=args.multi_gpu,
							logging_steps=args.logging_steps,
							fp16_opt_level=args.fp16_opt_level)
```

### 5. Train the model

```python
learner.fit(epochs=6,
			lr=6e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="lamb")
```

Fast-Bert now supports LAMB optmizer. Due to the speed of training, we have set LAMB as the default optimizer. You can switch back to AdamW by setting optimizer_type to 'adamw'.

### 6. Save trained model artifacts

```python
learner.save_model()
```

Model artefacts will be persisted in the output_dir/'model_out' path provided to the learner object. Following files will be persisted:

| File name               | description                                      |
| ----------------------- | ------------------------------------------------ |
| pytorch_model.bin       | trained model weights                            |
| spiece.model            | sentence tokenizer vocabulary (for xlnet models) |
| vocab.txt               | workpiece tokenizer vocabulary (for bert models) |
| special_tokens_map.json | special tokens mappings                          |
| config.json             | model config                                     |
| added_tokens.json       | list of new tokens                               |

The pytorch_model.bin contains the finetuned weights and you can point the classification task learner object to this file throgh the `finetuned_wgts_path` parameter.

## Amazon Sagemaker Support

The purpose of this library is to let you train and deploy production grade models. As transformer models require expensive GPUs to train, I have added support for training and deploying model on AWS SageMaker.

The repository contains the docker image and code for building BERT based classification models in Amazon SageMaker.

Please refer to my blog [Train and Deploy the Mighty BERT based NLP models using FastBert and Amazon SageMaker](https://towardsdatascience.com/train-and-deploy-mighty-transformer-nlp-models-using-fastbert-and-aws-sagemaker-cc4303c51cf3) that provides detailed explanation on using SageMaker with FastBert.

## Citation

Please include a mention of [this library](https://github.com/kaushaltrivedi/fast-bert) and HuggingFace [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) library and a link to the present repository if you use this work in a published or open-source project.

Also include my blogs on this topic:

- [Introducing FastBert — A simple Deep Learning library for BERT Models](https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384)
- [Multi-label Text Classification using BERT – The Mighty Transformer](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)

- [Train and Deploy the Mighty BERT based NLP models using FastBert and Amazon SageMaker](https://towardsdatascience.com/train-and-deploy-mighty-transformer-nlp-models-using-fastbert-and-aws-sagemaker-cc4303c51cf3)
