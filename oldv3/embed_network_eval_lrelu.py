from mrjob.job import MRJob
import json
import time
import logging
import re
from mrjob.protocol import JSONValueProtocol
import sys
import os
import random

def sigmoid(z):
    import numpy as np
    z = z.clip(-700, 700)
    result = 1 / (1+np.exp(-z))
    return result
def cross_entropy_loss(logits, labels):
    import numpy as np
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return -np.sum(labels * np.log(probabilities + 1e-12)) / logits.shape[0]
def sigmoid_der(z):
    z = sigmoid(z)
    return z * (1-z)
def printf(text):
    logging.debug(text)
def soft_max(z):
    import numpy as np
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def lrelu(z):
    import numpy as np
    return np.where(z > 0, z, z * 0.5)
def lrelu_der(z):
    import numpy as np
    return np.where(z > 0, 1, -0.5)
WEIGHT = {}
def extract(text: str) -> list[str]:
    return re.sub(r'[^a-z\s]', '', text.lower()).split()
def forward(X, embed, theta1, theta2, bias1, bias2):
    import numpy as np
    batch_size = X.shape[0]
    embedded: np.ndarray= embed[X]
    flattened = embedded.reshape(batch_size, -1)
    hidden = np.dot(flattened, theta1) + bias1
    activated_hidden = lrelu(hidden)
    logits = np.dot(activated_hidden, theta2) + bias2
    probabilities = soft_max(logits)
    return probabilities
class TextClassifierEvaluate(MRJob):
    FILES = ["util.py"]
    OUTPUT_PROTOCOL = JSONValueProtocol
    def __init__(self, *args, **kwargs):

        super(TextClassifierEvaluate, self).__init__(*args, **kwargs)
        from util import get_vocab, EMBED_DIM, SEQ_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE, MAPPER_SPLIT, FC1_LR, FC2_LR, EMBED_LR
        vocab = get_vocab()
        self.vocab_length = len(vocab)

        self.embedding_dim = EMBED_DIM
        self.seq_length = SEQ_LENGTH
        self.input_size = SEQ_LENGTH * EMBED_DIM
        self.hidden_size = HIDDEN_SIZE
        self.output_size = OUTPUT_SIZE

        self.mapper_split = MAPPER_SPLIT
        self.embed_lr = EMBED_LR
        self.fc1_lr = FC1_LR
        self.fc2_lr = FC2_LR

        self.word2idx = {w:i for i,w in enumerate(vocab)}
        self.idx2word = {i:w for i,w in enumerate(vocab)}

        logging.basicConfig(level=logging.DEBUG)
        self.load_weight()
    def load_weight(self):
        import numpy as np
        global WEIGHT
        if ("embed" in WEIGHT):
            # printf("Load old model weight")
            self.embed = WEIGHT["embed"]
            self.theta1 = WEIGHT["theta1"]
            self.bias1 = WEIGHT["bias1"]
            self.theta2 = WEIGHT["theta2"]
            self.bias2 = WEIGHT["bias2"]
            self.epoch = WEIGHT["epoch"]
        else:
            printf("*** *** *** *** *** *** ***")
            printf("Initialize new model weight")
            printf("*** *** *** *** *** *** ***")
            rand_range = 0
            self.embed = np.random.rand(self.vocab_length, self.embedding_dim) * rand_range - rand_range/2
            self.theta1 = np.random.rand(self.input_size, self.hidden_size) * rand_range - rand_range/2
            self.bias1 = np.random.rand(self.hidden_size) * rand_range - rand_range/2
            self.theta2 = np.random.rand(self.hidden_size, self.output_size) * rand_range - rand_range/2
            self.bias2 = np.random.rand(self.output_size) * rand_range - rand_range/2
            self.epoch = 0
            WEIGHT = {
                "epoch" : 0,
                "embed" : np.array(self.embed),
                "theta1" : np.array(self.theta1),
                "theta2" : np.array(self.theta2),
                "bias1" : np.array(self.bias1),
                "bias2" : np.array(self.bias2)
            }
    def mapper(self, key, line: str):
        data = line.split(",")
        if len(data) < 3: return
        # label_text = data[-1]
        label = data[-2]
        text = ",".join(data[:-2])
        if text == "text": return
        else:
            key = 1
            if self.mapper_split != 0:
                key = len(text) % self.mapper_split
            text = extract(text)
            if len(text) < self.seq_length:
                text.extend(["<pad>" for _ in range(self.seq_length - len(text))])
            text = text[:self.seq_length]
            input = [self.word2idx[w] for w in text]
            label = int(label)
            one_hot_label = [1 if i == label else 0 for i in range(self.output_size)]
            yield key, json.dumps({"X": input, "y": one_hot_label})
    def combiner(self, key, records):
        import numpy as np

        batch_X, batch_y = [], []
        # printf(key)
        for record in records:
            data = json.loads(record)
            batch_X.append(data["X"])
            batch_y.append(data["y"])
        
        if len(batch_X) > 0:
            X_np = np.array(batch_X)
            y_np = np.array(batch_y)
            probabilities = forward(X_np, self.embed, self.theta1, self.theta2, self.bias1, self.bias2)
            batch_size = y_np.shape[0]
            probabilities = np.clip(probabilities, 1e-12, 1e12)
            loss = -np.sum(y_np * np.log(probabilities)) / batch_size
            predicted = np.argmax(probabilities, axis=1)
            true_class = np.argmax(y_np, axis=1)
            accuracy = np.mean(predicted == true_class)
            yield 1, json.dumps({
                "loss" : loss,
                "accuracy" : accuracy
                })
    def reducer(self, key, grads):
        import numpy as np

        total_loss = 0
        total_accuracy = 0
        count = 0

        for grad in grads:

            metric_data = json.loads(grad)
            total_loss += metric_data["loss"]
            total_accuracy += metric_data["accuracy"]

            count += 1
        if (count > 0):
            self.metrics = {
                "loss" : total_loss / count,
                "accuracy" : total_accuracy / count
            }
            yield None, {
                "Status" : "Success"
            }
        else:
            yield None, {
                "Status" : "Failed"
            }
    def reducer_final(self):
        from util import get_abs_output_path
        file_path = os.path.join(get_abs_output_path(),"evaluate.json")
        data = {}
        if (os.path.exists(file_path)):
            with open(file_path, 'r') as file:
                data = json.loads(file.read())
        data[self.epoch] = self.metrics
        with open(file_path, 'w') as file:
            file.write(json.dumps(data))
def load_weight():
    global WEIGHT
    import numpy as np
    from util import get_abs_output_path
    path = os.path.join(get_abs_output_path(),"model_weight.json")
    if os.path.exists(path):
        with open(path, 'r') as file:
            weight = json.loads(file.read())
            if ("embed" in weight):
                WEIGHT = {
                    "epoch" : weight["epoch"],
                    "embed" : np.array(weight["embed"]),
                    "theta1" : np.array(weight["theta1"]),
                    "theta2" : np.array(weight["theta2"]),
                    "bias1" : np.array(weight["bias1"]),
                    "bias2" : np.array(weight["bias2"])
                }
if __name__ == "__main__":
    load_weight()
    start_time = time.time()
    TextClassifierEvaluate.run()
    printf(f"Finished {time.time() - start_time}s")
