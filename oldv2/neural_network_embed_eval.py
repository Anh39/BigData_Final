from mrjob.job import MRJob
import json
import time
import logging
import re
from mrjob.protocol import JSONValueProtocol
import sys
import os

def sigmoid(z):
    import numpy as np

    z = z.clip(-700, 700)
    result = 1 / (1+np.exp(-z))
    return result
    
def sigmoid_grad(z):
    return sigmoid(z) * (1-sigmoid(z))

def printf(text):
    logging.debug(text)

def compute_loss(X, y, theta1, theta2): # BCE
    import numpy as np
    m = X.shape[0]
    epsilon = 1e-5
    _, _, _, _, h = forward(X, theta1, theta2)
    loss = - (y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return np.sum(loss) / m
def soft_max(z):
    import numpy as np
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def forward(X, theta1, theta2):
    import numpy as np

    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, theta2.T)
    h = soft_max(z3)
    return a1, z2, a2, z3, h

def backward(X, y, theta1, theta2):
    import numpy as np

    m = X.shape[0]
    a1, z2, a2, z3, h = forward(X, theta1, theta2)

    d3 = h - y
    d2 = np.dot(d3, theta2[:, 1:]) * sigmoid_grad(z2)

    Delta1 = np.dot(d2.T, a1)
    Delta2 = np.dot(d3.T, a2)

    theta1_grad = Delta1 / m
    theta2_grad = Delta2 / m

    return theta1_grad, theta2_grad
def extract(text: str) -> list[str]:
    return re.sub(r'[^a-z\s]', '', text.lower()).split()

WEIGHT = {}
class TextClassifierEvaluate(MRJob):
    FILES = ["util.py"]
    OUTPUT_PROTOCOL = JSONValueProtocol
    def __init__(self, *args, **kwargs):
        import numpy as np

        super(TextClassifierEvaluate, self).__init__(*args, **kwargs)
        from util import get_vocab, EMBED_DIM, SEQ_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE, MAPPER_SPLIT
        vocab = get_vocab()
        self.vocab_length = len(vocab)
        self.embedding_dim = EMBED_DIM
        self.seq_length = SEQ_LENGTH
        self.input_layer_size = self.seq_length * self.embedding_dim
        self.hidden_layer_size = HIDDEN_SIZE
        self.output_layer_size = OUTPUT_SIZE
        self.mapper_split = MAPPER_SPLIT
        self.theta1 = WEIGHT['theta1']
        self.theta2 = WEIGHT['theta2']
        self.embed = WEIGHT["embed"]
        self.word2idx = {w:i for i,w in enumerate(vocab)}
        self.idx2word = {i:w for i,w in enumerate(vocab)}
        logging.basicConfig(level=logging.DEBUG)

    def mapper(self, key, line: str):
        data = line.split(",")
        if len(data) < 3: return
        label_text = data[-1]
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
            text = [self.word2idx.get(w, self.word2idx['<unk>']) for w in text]
            embed_text = []
            for word_idx in text:
                embed_text.extend(self.embed[word_idx].tolist())
            label = int(label)
            one_hot_label = [1 if i == label else 0 for i in range(self.output_layer_size)]
            yield key, json.dumps({"X": embed_text, "y": one_hot_label})
    def combiner(self, key, records):
        import numpy as np

        batch_X, batch_y = [], []
        for record in records:
            data = json.loads(record)
            batch_X.append(data["X"])
            batch_y.append(data["y"])
        
        if len(batch_X) > 0:
            X_np = np.array(batch_X)
            y_np = np.array(batch_y)
            loss = compute_loss(X_np, y_np, self.theta1, self.theta2)
            count = 0
            _, _, _, _, h = forward(X_np, self.theta1, self.theta2)
            for i in range(len(batch_X)):
                predicted = np.argmax(h[i])
                if (predicted == np.argmax(y_np[i])):
                    count += 1
            accuracy = count/len(batch_X)
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
            total_loss += metric_data['loss']
            total_accuracy += metric_data['accuracy']
            count += 1
        if (count > 0):
            self.loss = total_loss / count
            self.accuracy = total_accuracy / count
            yield None, {
                "Status" : "Sucess",
                "Metrics" : {
                    "Loss" : self.loss,
                    "Accuracy" : self.accuracy
                }
            }
        else:
            yield None, {
                "Status" : "Failed"
            }
    def reducer_final(self):
        # printf("-----------------")
        from util import get_abs_output_path
        metric = {
                "loss" : self.loss,
                "accuracy" : self.accuracy
            }
        printf(metric)
        file_path = os.path.join(get_abs_output_path(),"metric_result.json")
        with open(file_path, 'w') as file:
            file.write(json.dumps(metric))
        # printf(file_path)
def init():
    global WEIGHT
    import numpy as np
    from util import get_vocab, get_abs_output_path
    get_vocab()
    path = os.path.join(get_abs_output_path(),"model_weight.json")
    if os.path.exists(path):
        with open(path, 'r') as file:
            weight = json.loads(file.read())
            if ("theta1" in weight and "theta2" in weight):
                WEIGHT = {
                    "theta1" : np.array(weight["theta1"]),
                    "theta2" : np.array(weight["theta2"])
                }
    path = os.path.join(get_abs_output_path(),"embed_weight.json")
    if os.path.exists(path):
        with open(path, 'r') as file:
            weight = json.loads(file.read())
            if ("embed" in weight):
                WEIGHT["embed"] = np.array(weight["embed"])
if __name__ == "__main__":
    init()
    start_time = time.time()
    TextClassifierEvaluate.run()
    printf(f"Finished {time.time() - start_time}s")