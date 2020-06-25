
import tensorflow as tf
import tensorflow_text as text

from bert import load_bert_layer

MAX_SEQ_LEN = 64  # max number is 512
do_lower_case = load_bert_layer().resolved_object.do_lower_case.numpy()

def preprocessing_fn(inputs):
    """Preprocess input column of text into transformed columns of.
        * input token ids
        * input mask
        * input type ids
    """

    CLS_ID = tf.constant(101, dtype=tf.int64)
    SEP_ID = tf.constant(102, dtype=tf.int64)
    PAD_ID = tf.constant(0, dtype=tf.int64)

    vocab_file_path = load_bert_layer().resolved_object.vocab_file.asset_path
    
    bert_tokenizer = text.BertTokenizer(vocab_lookup_table=vocab_file_path, 
                                        token_out_type=tf.int64, 
                                        lower_case=do_lower_case) 
    
    def tokenize_text(text, sequence_length=MAX_SEQ_LEN):
        """
        Perform the BERT preprocessing from text -> input token ids
        """

        # convert text into token ids
        tokens = bert_tokenizer.tokenize(text)
        
        # flatten the output ragged tensors 
        tokens = tokens.merge_dims(1, 2)[:, :sequence_length]
        
        # Add start and end token ids to the id sequence
        start_tokens = tf.fill([tf.shape(text)[0], 1], CLS_ID)
        end_tokens = tf.fill([tf.shape(text)[0], 1], SEP_ID)
        tokens = tokens[:, :sequence_length - 2]
        tokens = tf.concat([start_tokens, tokens, end_tokens], axis=1)

        # truncate sequences greater than MAX_SEQ_LEN
        tokens = tokens[:, :sequence_length]

        # pad shorter sequences with the pad token id
        tokens = tokens.to_tensor(default_value=PAD_ID)
        pad = sequence_length - tf.shape(tokens)[1]
        tokens = tf.pad(tokens, [[0, 0], [0, pad]], constant_values=PAD_ID)

        # and finally reshape the word token ids to fit the output 
        # data structure of TFT  
        return tf.reshape(tokens, [-1, sequence_length])

    def preprocess_bert_input(text):
        """
        Convert input text into the input_word_ids, input_mask, input_type_ids
        """
        input_word_ids = tokenize_text(text)
        input_mask = tf.cast(input_word_ids > 0, tf.int64)
        input_mask = tf.reshape(input_mask, [-1, MAX_SEQ_LEN])
        
        zeros_dims = tf.stack(tf.shape(input_mask))
        input_type_ids = tf.fill(zeros_dims, 0)
        input_type_ids = tf.cast(input_type_ids, tf.int64)

        return (
            input_word_ids, 
            input_mask,
            input_type_ids
        )

    input_word_ids, input_mask, input_type_ids = \
        preprocess_bert_input(tf.squeeze(inputs['text'], axis=1))

    return {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids,
        'label': inputs['label']
    }