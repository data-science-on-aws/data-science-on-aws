import argparse
from pathlib import Path
from tqdm import tqdm
import requests
import urllib3

PRETRAINED_VOCAB_FILES_MAP = {
    # BERT
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
    "bert-base-german-cased": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
    # XLNet
    "xlnet-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-spiece.model",
    "xlnet-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-spiece.model",
    # ROBERTA
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-vocab.json",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
    # CamemBERT
    "camembert-base": "https://s3.amazonaws.com/models.huggingface.co/bert/camembert-base-sentencepiece.bpe.model",
    # ALBERT
    "albert-base-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-spiece.model",
    "albert-large-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-spiece.model",
    "albert-xlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-spiece.model",
    "albert-xxlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-spiece.model",
    "albert-base-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-spiece.model",
    "albert-large-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-spiece.model",
    "albert-xlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-spiece.model",
    "albert-xxlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-spiece.model",
    # DISTILBERT
    "distilbert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
}

PRETRAINED_VOCAB_MERGES_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-merges.txt",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-merges.txt",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
}

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    # BERT
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    # XLNet
    "xlnet-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin",
    "xlnet-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-pytorch_model.bin",
    # ROBERTA
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
    # CamemBERT
    "camembert-base": "https://s3.amazonaws.com/models.huggingface.co/bert/camembert-base-pytorch_model.bin",
    # ALBERT
    "albert-base-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-pytorch_model.bin",
    "albert-large-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-pytorch_model.bin",
    "albert-xlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-pytorch_model.bin",
    "albert-xxlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-pytorch_model.bin",
    "albert-base-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-pytorch_model.bin",
    "albert-large-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-pytorch_model.bin",
    "albert-xlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-pytorch_model.bin",
    "albert-xxlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-pytorch_model.bin",
    # DISTILBERT
    "distilbert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-pytorch_model.bin",
}

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # BERT
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
    # XLNet
    "xlnet-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json",
    "xlnet-large-cased": "https: // s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-config.json",
    # ROBERTA
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-config.json",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-config.json",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-config.json",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-config.json",
    # CamemBERT
    "camembert-base": "https://s3.amazonaws.com/models.huggingface.co/bert/camembert-base-config.json",
    # ALBERT
    "albert-base-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-config.json",
    "albert-large-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-config.json",
    "albert-xlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-config.json",
    "albert-xxlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-config.json",
    "albert-base-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-config.json",
    "albert-large-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json",
    "albert-xlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-config.json",
    "albert-xxlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-config.json",
    # DISTILBERT
    "distilbert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-config.json",
}


def http_get(url, target):
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    with open(target, "wb") as target_file:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                target_file.write(chunk)
    progress.close()


def download_pretrained_files(model_name, location):
    model_type = model_name.split("-")[0]
    print("model name is {}".format(model_name))
    location = location / model_name
    print("location is {}".format(location))
    location.mkdir(exist_ok=True)
    # download vocab files
    try:
        file_path = PRETRAINED_VOCAB_FILES_MAP[model_name]
        print("file path is {}".format(file_path))
        if model_type == "bert":
            file_name = "vocab.txt"
        if model_type == "distilbert":
            file_name = "vocab.txt"
        elif model_type == "xlnet":
            file_name = "spiece.model"
        elif model_type == "roberta":
            file_name = "vocab.json"

        target_path = location / file_name
        http_get(file_path, target_path)

    except:
        print(
            "error downloading model vocab {} for  model {}".format(
                file_path, model_name
            )
        )

    # download vocab merge file for Roberta
    if model_type == "roberta":
        try:
            file_path = PRETRAINED_VOCAB_MERGES_MAP[model_name]
            print(file_path)
            file_name = "merges.txt"
            target_path = location / file_name
            http_get(file_path, target_path)

        except:
            print("error downloading model merge file for {}".format(model_name))

    # download model files
    try:
        file_path = BERT_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]
        print(file_path)
        file_name = "pytorch_model.bin"
        target_path = location / file_name
        http_get(file_path, target_path)

    except:
        print("error downloading model file for {}".format(model_name))

    # download config files
    try:
        file_path = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name]
        print(file_path)
        file_name = "config.json"
        target_path = location / file_name
        http_get(file_path, target_path)

    except:
        print("error downloading model config for {}".format(model_name))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--location_dir",
        default=None,
        type=str,
        required=True,
        help="The location where pretrained model needs to be stored",
    )

    parser.add_argument(
        "--models",
        default=None,
        type=str,
        required=True,
        nargs="*",
        help="download the pretrained models",
    )

    args = parser.parse_args()
    print(args)
    Path(args.location_dir).mkdir(exist_ok=True)

    models = args.models
    #    [download_pretrained_files(k, location=Path(args.location_dir))
    #     for k, v in BERT_PRETRAINED_MODEL_ARCHIVE_MAP.items()]
    [
        download_pretrained_files(item, location=Path(args.location_dir))
        for item in args.models
    ]


if __name__ == "__main__":
    main()
