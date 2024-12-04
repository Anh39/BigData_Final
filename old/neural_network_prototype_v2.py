import numpy as np
from mrjob.job import MRJob
import json
import time
import logging
import re
from mrjob.protocol import JSONValueProtocol
import sys

def sigmoid(z: np.ndarray):
    z = z.clip(-700, 700)
    result = 1 / (1+np.exp(-z))
    return result
    
def sigmoid_grad(z):
    return sigmoid(z) * (1-sigmoid(z))

def printf(text):
    logging.debug(text)

def forward(X: np.ndarray, theta1: np.ndarray, theta2: np.ndarray):
    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def backward(X: np.ndarray, y: np.ndarray, theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
def get_vocab(file_path: str):
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
class TextClassifier(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol
    DIRS = ["env/lib/python3.10/site-packages/numpy"]
    def __init__(self, *args, **kwargs):
        super(TextClassifier, self).__init__(*args, **kwargs)
        self.max_length = 100
        self.input_layer_size = self.max_length
        self.hidden_layer_size = 128
        self.output_layer_size = 4
        self.theta1 = np.random.rand(self.hidden_layer_size, self.input_layer_size + 1) * 0.1 - 0.05
        self.theta2 = np.random.rand(self.output_layer_size, self.hidden_layer_size + 1) * 0.1 - 0.05
        self.vocab = get_vocab("data/ag_news_data/train.csv")
        self.vocab.append("<unk>")
        self.vocab.append("<pad>")
        self.word2idx = {w:i for i,w in enumerate(self.vocab)}
        self.idx2word = {i:w for i,w in enumerate(self.vocab)}
        self.lr = 1e-1
        logging.basicConfig(level=logging.DEBUG)

    def mapper(self, _, line: str):
        printf("--------------Start mapper")
        data = line.split(",")
        if len(data) < 3: return
        label_text = data[-1]
        label = data[-2]
        text = ",".join(data[:-2])
        if text == "text": return
        else:
            text = extract(text)
            text = [self.word2idx[w] for w in text]
            if len(text) < self.max_length:
                text.extend([self.word2idx["<pad>"] for _ in range(self.max_length - len(text))])
            text = text[:self.max_length]
            label = int(label)
            one_hot_label = [1 if i == label else 0 for i in range(self.output_layer_size)]

            yield 1, json.dumps({"X": text, "y": one_hot_label})
    def combiner(self, key, records):
        batch_X, batch_y = [], []
        # printf(key)
        for record in records:
            data = json.loads(record)
            batch_X.append(data["X"])
            batch_y.append(data["y"])
        
        if len(batch_X) > 0:
            X_np = np.array(batch_X)
            y_np = np.array(batch_y)
            theta1_grad, theta2_grad = backward(X_np, y_np, self.theta1, self.theta2)
            yield 1, json.dumps({
                "theta1_grad" : theta1_grad.tolist(),
                "theta2_grad" : theta2_grad.tolist()
                })
    def reducer(self, key, grads):
        total_theta1_grads = np.zeros_like(self.theta1)
        total_theta2_grads = np.zeros_like(self.theta2)
        count = 0

        for grad in grads:
            grad_data = json.loads(grad)
            total_theta1_grads += np.array(grad_data["theta1_grad"])
            total_theta2_grads += np.array(grad_data["theta2_grad"])
            count += 1
        if (count > 0):
            avg_theta1_grad = total_theta1_grads / count
            avg_theta2_grad = total_theta2_grads / count

            self.theta1 -= self.lr * avg_theta1_grad
            self.theta2 -= self.lr * avg_theta2_grad

            yield None, {
                "Status" : "Sucess"
            }
        else:
            yield None, {
                "Status" : "Failed"
            }

if __name__ == "__main__":
    get_vocab("/home/hd_user/storage/final/data/ag_news_data/train.csv")
    with open("vocab.json", 'w') as file:
        file.write(json.dumps(VOCAB))
    start_time = time.time()
    TextClassifier.run()
    printf(f"Finished {time.time() - start_time}s")