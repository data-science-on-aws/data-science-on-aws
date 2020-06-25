
import tensorflow_hub as hub

BERT_TFHUB_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

def load_bert_layer(model_url=BERT_TFHUB_URL):
    # Load the pre-trained BERT model as layer in Keras
    bert_layer = hub.KerasLayer(
        handle=model_url,
        trainable=True)
    return bert_layer