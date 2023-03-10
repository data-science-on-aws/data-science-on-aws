import argparse
import logging
import os
import sys

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1024)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=int, default=4)
    parser.add_argument("--teacher_id", type=str)
    parser.add_argument("--student_id", type=str)
    parser.add_argument("--dataset_id", type=str)
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--run_hpo", type=bool, default=True)
    parser.add_argument("--n_trials", type=int, default=50)

    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default="every_save")
    parser.add_argument("--hub_token", type=str, default=None)

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_id)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_id)

    # sample input
    sample = "This is a basic example, with different words to test."

    # assert results
    assert tokenizer(sample) == student_tokenizer(sample), "Tokenizers are not compatible"

    # load datasets
    dataset = load_dataset(args.dataset_id, args.dataset_config)

    # process dataset
    def process(examples):
        tokenized_inputs = tokenizer(examples["sentence"], truncation=True, max_length=512)
        return tokenized_inputs

    tokenized_datasets = dataset.map(process, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # define metrics and metrics function
    accuracy_metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        return {
            "accuracy": acc["accuracy"],
        }

    # create label2id, id2label dicts for nice outputs for the model
    labels = tokenized_datasets["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # define training args
    training_args = DistillationTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
        learning_rate=float(args.learning_rate),
        seed=33,
        # logging & evaluation strategies
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="epoch",  # to get more information to TB
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard",
        # push to hub parameters
        push_to_hub=args.push_to_hub,
        hub_strategy="every_save",
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        # distilation parameters
        alpha=args.alpha,
        temperature=args.temperature,
    )

    # define data_collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # define teach model
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # init method is needed when using hpo
    def student_init():
        return AutoModelForSequenceClassification.from_pretrained(
            args.student_id, num_labels=num_labels, id2label=id2label, label2id=label2id
        )

    student_model = student_init()

    trainer = DistillationTrainer(
        model_init=student_init,
        args=training_args,
        teacher_model=teacher_model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # run hpo if arg is provided
    if args.run_hpo:
        # to avoind unnecessary pushes of bad models
        training_args.push_to_hub = False
        training_args.output_dir = "./tmp/hpo"
        training_args.logging_dir = "./tmp/hpo/logs"

        # hpo space which replace the training_args
        def hp_space(trial):
            return {
                "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "alpha": trial.suggest_float("alpha", 0, 1),
                "temperature": trial.suggest_int("temperature", 2, 30),
            }

        best_run = trainer.hyperparameter_search(n_trials=args.n_trials, direction="maximize", hp_space=hp_space)

        # print best run
        print(best_run)
        # overwrite initial hyperparameters with from the best_run
        for k, v in best_run.hyperparameters.items():
            setattr(training_args, k, v)

        training_args.push_to_hub = args.push_to_hub
        training_args.output_dir = args.output_dir
        training_args.logging_dir = f"{args.output_dir}/logs"

    # train model with inital hyperparameters or hyperparameters from the best run
    trainer.train()

    # save best model, metrics and create model card
    trainer.create_model_card(model_name=args.hub_model_id)
    trainer.push_to_hub()

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])
