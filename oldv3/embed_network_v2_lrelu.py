from mrjob.job import MRJob
import json
import time
import logging
import re
import os

def printf(text):
    logging.debug(text)
def xavier_init(fan_in, fan_out):
    import numpy as np
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))
def he_init(fan_in, fan_out):
    import numpy as np
    stddev = np.sqrt(2/fan_in)
    return np.random.normal(0, stddev, size=(fan_in, fan_out))
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
def sigmoid(z):
    import numpy as np
    z = np.clip(z, -700, 700)
    result = 1 / (1+np.exp(-z))
    return result
def lrelu(z):
    import numpy as np
    return np.where(z > 0, z, z * 0.5)
def lrelu_der(z):
    import numpy as np
    return np.where(z > 0, 1, -0.5)
def sigmoid_der(z):
    z = sigmoid(z)
    return z * (1-z)
def soft_max(z):
    import numpy as np
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def backward(X, y, embedding, theta1, theta2, bias1, bias2):
    import numpy as np
    X : np.ndarray = X
    y : np.ndarray = y
    embedding : np.ndarray = embedding
    theta1 : np.ndarray = theta1
    theta2 : np.ndarray = theta2
    bias1 : np.ndarray = bias1
    bias2 : np.ndarray = bias2

    # Forward
    batch_size = X.shape[0]
    seq_length = X.shape[1]
    embedded: np.ndarray= embedding[X]
    flattened = embedded.reshape(batch_size, -1)
    hidden = np.dot(flattened, theta1) + bias1
    activated_hidden = lrelu(hidden)
    logits = np.dot(activated_hidden, theta2) + bias2
    probabilities = soft_max(logits)

    # Backward
    # Total grad
    dlogits = probabilities - y
    # FC2 grad
    dfc2_weight = np.dot(activated_hidden.T, dlogits)
    dfc2_bias = np.sum(dlogits, axis=0)

    # FC1 grad
    dactivated_hidden = np.dot(dlogits, theta2.T)
    dhidden = dactivated_hidden * lrelu_der(hidden)
    dfc1_weight = np.dot(flattened.T, dhidden)
    dfc1_bias = np.sum(dhidden, axis=0)

    # Embed
    dflattened: np.ndarray = np.dot(dhidden, theta1.T)
    dembedded = dflattened.reshape(embedded.shape)

    dembedding: np.ndarray = np.zeros_like(embedding)
    for i in range(batch_size):
        for j in range(seq_length):
            dembedding[X[i][j]] += dembedded[i, j]
    return dembedding, dfc1_bias, dfc1_weight, dfc2_bias, dfc2_weight

class TextClassifierTrainer(MRJob):
    FILES = ["util.py"]
    def __init__(self, *args, **kwargs):

        super(TextClassifierTrainer, self).__init__(*args, **kwargs)
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

        import numpy as np

        self.total_fc1_weight_grads = np.zeros_like(self.theta1)
        self.total_fc2_weight_grads = np.zeros_like(self.theta2)
        self.total_fc1_bias_grads = np.zeros_like(self.bias1)
        self.total_fc2_bias_grads = np.zeros_like(self.bias2)
        self.total_embedding_grads = np.zeros_like(self.embed)
        self.count = 0
        self.line_count = 0
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
            self.embed = he_init(self.vocab_length, self.embedding_dim)
            self.theta1 = he_init(self.input_size, self.hidden_size)
            self.bias1 = np.zeros(self.hidden_size)
            self.theta2 = he_init(self.hidden_size, self.output_size)
            self.bias2 = np.zeros(self.output_size)
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
            key = None
            if self.mapper_split != 0:
                key = self.line_count
                self.line_count += 1
                key %= self.mapper_split
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
        for record in records:
            data = json.loads(record)
            batch_X.append(data["X"])
            batch_y.append(data["y"])
        if len(batch_X) > 0:
            X_np = np.array(batch_X)
            y_np = np.array(batch_y)
            embedding_grad, fc1_bias_grad, fc1_weight_grad, fc2_bias_grad, fc2_weight_grad = backward(X_np, y_np, self.embed, self.theta1, self.theta2, self.bias1, self.bias2)
            yield key, json.dumps({
                "embedding_grad" : embedding_grad.tolist(),
                "fc1_bias_grad" : fc1_bias_grad.tolist(),
                "fc1_weight_grad" : fc1_weight_grad.tolist(),
                "fc2_bias_grad" : fc2_bias_grad.tolist(),
                "fc2_weight_grad" : fc2_weight_grad.tolist()
                })
    def reducer(self, _, grads):
        import numpy as np
        for grad in grads:

            grad_data = json.loads(grad)
            self.total_fc1_weight_grads += np.array(grad_data["fc1_weight_grad"])
            self.total_fc1_bias_grads += np.array(grad_data["fc1_bias_grad"])
            self.total_fc2_weight_grads += np.array(grad_data["fc2_weight_grad"])
            self.total_fc2_bias_grads += np.array(grad_data["fc2_bias_grad"])
            self.total_embedding_grads += np.array(grad_data["embedding_grad"])

            self.count += 1
    def reducer_final(self):
        if (self.count > 0):
            self.theta1 -= self.fc1_lr / self.count * self.total_fc1_weight_grads
            self.theta2 -= self.fc2_lr / self.count * self.total_fc2_weight_grads
            self.bias1 -= self.fc1_lr / self.count * self.total_fc1_bias_grads
            self.bias2 -= self.fc2_lr / self.count * self.total_fc2_bias_grads
            self.embed -= self.embed_lr / self.count * self.total_embedding_grads
            self.log = {
                "fc1" : self.total_fc1_weight_grads.tolist(),
                "fc2" : self.total_fc2_weight_grads.tolist(),
                "embed" : self.total_embedding_grads.tolist()
            }

        from util import get_abs_output_path
        file_path = os.path.join(get_abs_output_path(),"model_weight.json")
        model_weight = {
                "epoch" : self.epoch + 1,
                "lrs" : [self.embed_lr, self.fc1_lr, self.fc2_lr],
                "vocab_length" : self.vocab_length,
                "embed" :self.embed.tolist(),
                "theta1" : self.theta1.tolist(),
                "theta2" : self.theta2.tolist(),
                "bias1" : self.bias1.tolist(),
                "bias2" : self.bias2.tolist()
            }
        json_data = json.dumps(model_weight)
        with open(file_path, 'w') as file:
            file.write(json_data)

        file_path = os.path.join(get_abs_output_path(),"grad_log.json")
        with open(file_path, 'w') as file:
            file.write(json.dumps(self.log))
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
    TextClassifierTrainer.run()
    printf(f"Finished {time.time() - start_time}s")
