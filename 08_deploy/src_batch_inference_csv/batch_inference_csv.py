import json

class RequestHandler(object):

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, data):
        transformed_instances = []

        for instance in instances:
            print('Instance {}'.format(instance))

            instance_split = instance.split('\t')
            print(instance_split)

            tokens_a = self.tokenizer.tokenize(instance_split[13])

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[0:(self.max_seq_length - 2)]

            tokens = []  
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)  
            tokens.append("[SEP]")
            segment_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            instance = {"input_ids": input_ids, 
                        "input_mask": input_mask, 
                        "segment_ids": segment_ids}

            transformed_instances.append(instance)

        transformed_data = {"instances": transformed_instances}

        return json.dumps(transformed_data)


class ResponseHandler(object):
    import tensorflow as tf

    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, response, accept_header):
        response_body = response.read().decode('utf-8')

        response_json = json.loads(response_body)

        log_probabilities = response_json["predictions"]

        predicted_classes = []

        for log_probability in log_probabilities:
            softmax = tf.nn.softmax(log_probability)    
            predicted_class_idx = tf.argmax(softmax, axis=-1, output_type=tf.int32)
            predicted_class = self.classes[predicted_class_idx]
            predicted_classes.append(predicted_class)

        return json.dumps(predicted_classes)


if __name__ == "__main__":

#    instances = ["""'US#01137107850#011R2C1DJSCC8UFS6#011B00EP7AP7C#011279360628#011Family Tree Maker Platinum#011Software#0113#0110#0110#011N#011Y#011Three Stars#011Didn't like having to get all of my old files redone.#0112015-08-31""",
#"""'US#01120193077#011R1XU1B93402SYJ#011B00N4OLCRO#011776572654#011Photoshop Elements 13#011Software#0111#0111#0111#011N#011Y#011Can't load to my computer.#011I have tried for 3 days to get an answer from both Amazon and Adobe, asking for the redemption number and/or serial number so that I can load this program into my computer.  I have not heard from either, so I will return it to Amazon for a refund.  Sorry it didn't work, I was excited to try this.#0112015-08-31"""]

#    instances = ["This is great!", 
#                 "This is terrible."]

    from transformers import DistilBertTokenizer

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    request_handler = RequestHandler(tokenizer=tokenizer,
                                     max_seq_length=128)

    response_handler = ResponseHandler(classes=[1, 2, 3, 4, 5])

    predicted_classes = request_handler(instances)

    print(predicted_classes)
