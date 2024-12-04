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
OUTPUT_DIR = "/home/hd_user/storage/final/output"
WEIGHT = {}
LR = 0
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

class TextClassifierTrainer(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol
    def __init__(self, *args, **kwargs):

        super(TextClassifierTrainer, self).__init__(*args, **kwargs)
        self.max_length = 25
        self.input_layer_size = self.max_length
        self.hidden_layer_size = 1024
        self.output_layer_size = 4
        # self.initialize_weight()
        self.load_weight()
        self.vocab = get_vocab()
        self.vocab.append("<unk>")
        self.vocab.append("<pad>")
        self.word2idx = {w:i for i,w in enumerate(self.vocab)}
        self.idx2word = {i:w for i,w in enumerate(self.vocab)}
        self.lr = LR
        logging.basicConfig(level=logging.DEBUG)
        self.line_count = 0
    def initialize_weight(self):
        import numpy as np
        self.theta1 = np.zeros_like(np.random.rand(self.hidden_layer_size, self.input_layer_size + 1) * 0.1 - 0.05)
        self.theta2 = np.zeros_like(np.random.rand(self.output_layer_size, self.hidden_layer_size + 1) * 0.1 - 0.05)
        # self.theta1 = np.random.rand(self.hidden_layer_size, self.input_layer_size + 1) * 0.1 - 0.05
        # self.theta2 = np.random.rand(self.output_layer_size, self.hidden_layer_size + 1) * 0.1 - 0.05
    def load_weight(self):
        global WEIGHT
        if ("theta1" in WEIGHT):
            self.theta1 = WEIGHT["theta1"]
            self.theta2 = WEIGHT["theta2"]
        else:
            self.initialize_weight()
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
            key = len(text) % 4
            text = extract(text)
            text = [self.word2idx[w] for w in text]
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
            theta1_grad, theta2_grad = backward(X_np, y_np, self.theta1, self.theta2)
            yield 1, json.dumps({
                "theta1_grad" : theta1_grad.tolist(),
                "theta2_grad" : theta2_grad.tolist()
                })
    def reducer(self, key, grads):
        import numpy as np

        total_theta1_grads = np.zeros_like(self.theta1)
        total_theta2_grads = np.zeros_like(self.theta2)
        count = 0

        for grad in grads:
            # value = np.random.random()
            # if value < 0.75: continue
            grad_data = json.loads(grad)
            total_theta1_grads += np.array(grad_data["theta1_grad"])
            total_theta2_grads += np.array(grad_data["theta2_grad"])
            count += 1
        if (count > 0):
            avg_theta1_grad = total_theta1_grads / count
            avg_theta2_grad = total_theta2_grads / count
            # printf(avg_theta1_grad[0])
            # printf(avg_theta2_grad[0])
            self.theta1 -= self.lr * avg_theta1_grad
            self.theta2 -= self.lr * avg_theta2_grad
            self.finalize_custom()
            yield None, {
                "Status" : "Success"
            }
        else:
            yield None, {
                "Status" : "Failed"
            }
    def finalize_custom(self):
        # printf("-----------------")
        file_path = os.path.join(OUTPUT_DIR,"model_weight.json")
        with open(file_path, 'w') as file:
            file.write(json.dumps({
                "lr" : LR,
                "vocab_length" : len(VOCAB),
                "theta1_size" : self.theta1.shape,
                "theta2_size" : self.theta2.shape,
                "theta1" : self.theta1.tolist(),
                "theta2" : self.theta2.tolist()
            }))
        # printf(file_path)
def init(lr):
    global VOCAB, LR
    LR = lr
    with open("vocab.json", 'w') as file:
        file.write(json.dumps(VOCAB))
def load_weight():
    global WEIGHT
    import numpy as np
    get_vocab()
    with open(os.path.join(OUTPUT_DIR, "model_weight.json"), 'r') as file:
        weight = json.loads(file.read())
        if ("theta1" in weight and "theta2" in weight):
            WEIGHT = {
                "theta1" : np.array(weight["theta1"]),
                "theta2" : np.array(weight["theta2"])
            }
if __name__ == "__main__":
    init()
    load_weight()
    start_time = time.time()
    TextClassifierTrainer.run()
    printf(f"Finished {time.time() - start_time}s")
