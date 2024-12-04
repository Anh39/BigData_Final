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
VOCAB = None
def get_vocab(file_path: str = None):
    global VOCAB
    if VOCAB != None: return VOCAB
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [",".join(line.split(",")[:-2]) for line in lines]
    corpus = extract(" ".join(lines))
    vocab = list(set(corpus))
    vocab.sort()
    VOCAB = vocab
    return VOCAB
def extract(text: str) -> list[str]:
    return re.sub(r'[^a-z\s]', '', text.lower()).split()

WEIGHT = {}
OUTPUT_DIR = "/home/hd_user/storage/final/output"
class TextClassifierEvaluate(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol
    def __init__(self, *args, **kwargs):
        import numpy as np

        super(TextClassifierEvaluate, self).__init__(*args, **kwargs)
        self.max_length = 25
        self.input_layer_size = self.max_length
        self.hidden_layer_size = 1024
        self.output_layer_size = 4
        self.theta1 = WEIGHT['theta1']
        self.theta2 = WEIGHT['theta2']
        self.vocab = get_vocab()
        self.vocab.append("<unk>")
        self.vocab.append("<pad>")
        self.word2idx = {w:i for i,w in enumerate(self.vocab)}
        self.idx2word = {i:w for i,w in enumerate(self.vocab)}
        self.lr = 1e-1
        logging.basicConfig(level=logging.DEBUG)
        self.line_count = 0

    def mapper(self, key, line: str):
        # key = self.line_count
        # self.line_count += 1
        data = line.split(",")
        if len(data) < 3: return
        label_text = data[-1]
        label = data[-2]
        text = ",".join(data[:-2])
        if text == "text": return
        else:
            key = 1
            # key = len(text) % 4
            text = extract(text)
            text = [self.word2idx.get(w, self.word2idx['<unk>']) for w in text]
            if len(text) < self.max_length:
                text.extend([self.word2idx["<pad>"] for _ in range(self.max_length - len(text))])
            text = text[:self.max_length]
            label = int(label)
            one_hot_label = [1 if i == label else 0 for i in range(self.output_layer_size)]

            yield key, json.dumps({"X": text, "y": one_hot_label})
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
            self.finalize_custom()
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
    def finalize_custom(self):
        # printf("-----------------")
        metric = {
                "loss" : self.loss,
                "accuracy" : self.accuracy
            }
        printf(metric)
        file_path = os.path.join(OUTPUT_DIR,"metric_result.json")
        with open(file_path, 'w') as file:
            file.write(json.dumps(metric))
        # printf(file_path)
def init(data_path : str):
    import numpy as np
    global WEIGHT
    get_vocab(data_path)
    with open(os.path.join(OUTPUT_DIR, "model_weight.json"), 'r') as file:
        weight = json.loads(file.read())
        WEIGHT = {
            "theta1" : np.array(weight["theta1"]),
            "theta2" : np.array(weight["theta2"])
        }
if __name__ == "__main__":
    init("data/ag_news_data/train.csv")
    start_time = time.time()
    TextClassifierEvaluate.run()
    printf(f"Finished {time.time() - start_time}s")