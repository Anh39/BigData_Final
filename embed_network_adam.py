from mrjob.job import MRJob
import json
import time
import logging
import re
import os

def sigmoid(z):
    import numpy as np
    z = z.clip(-700, 700)
    result = 1 / (1+np.exp(-z))
    return result
def sigmoid_der(z):
    z = sigmoid(z)
    return z * (1-z)
def printf(text):
    logging.debug(text)
def soft_max(z):
    import numpy as np
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def xavier_init(fan_in, fan_out):
    import numpy as np
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))
WEIGHT = {}
def extract(text: str) -> list[str]:
    return re.sub(r'[^a-z\s]', '', text.lower()).split()
def forward(X, embed, theta1, theta2, bias1, bias2):
    import numpy as np
    batch_size = X.shape[0]
    embedded: np.ndarray= embed[X]
    flattened = embedded.reshape(batch_size, -1)
    hidden = np.dot(flattened, theta1) + bias1
    activated_hidden = sigmoid(hidden)
    logits = np.dot(activated_hidden, theta2) + bias2
    probabilities = soft_max(logits)
    return probabilities
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
    activated_hidden = sigmoid(hidden)
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
    dhidden = dactivated_hidden * sigmoid_der(hidden)
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

        import numpy as np


        logging.basicConfig(level=logging.DEBUG)
        self.load_weight()


        self.total_fc1_weight_grads = np.zeros_like(self.theta1)
        self.total_fc2_weight_grads = np.zeros_like(self.theta2)
        self.total_fc1_bias_grads = np.zeros_like(self.bias1)
        self.total_fc2_bias_grads = np.zeros_like(self.bias2)
        self.total_embedding_grads = np.zeros_like(self.embed)
        self.count = 0
        self.line_count = 0

    def adam_init(self):
        import numpy as np
        if ("m_embed" in WEIGHT):
            self.m_fc1_weight = WEIGHT["m_fc1_w"]
            self.m_fc2_weight = WEIGHT["m_fc2_w"]
            self.m_fc1_bias = WEIGHT["m_fc1_b"]
            self.m_fc2_bias = WEIGHT["m_fc2_b"]
            self.m_embed = WEIGHT["m_embed"]
            self.v_fc1_weight = WEIGHT["v_fc1_w"]
            self.v_fc2_weight = WEIGHT["v_fc2_w"]
            self.v_fc1_bias = WEIGHT["v_fc1_b"]
            self.v_fc2_bias = WEIGHT["v_fc2_b"]
            self.v_embed = WEIGHT["v_embed"]
        else:
            self.m_embed = np.zeros_like(self.embed)
            self.v_embed = np.zeros_like(self.embed)
            self.m_fc1_weight = np.zeros_like(self.theta1)
            self.v_fc1_weight = np.zeros_like(self.theta1)
            self.m_fc1_bias = np.zeros_like(self.bias1)
            self.v_fc1_bias = np.zeros_like(self.bias1)
            self.m_fc2_weight = np.zeros_like(self.theta2)
            self.v_fc2_weight = np.zeros_like(self.theta2)
            self.m_fc2_bias = np.zeros_like(self.bias2)
            self.v_fc2_bias = np.zeros_like(self.bias2)
            WEIGHT.update({
                "m_fc1_w" : self.m_fc1_weight,
                "m_fc2_w" : self.m_fc2_weight,
                "m_fc1_b" : self.m_fc1_bias,
                "m_fc2_b" : self.m_fc2_bias,
                "m_embed" : self.m_embed,
                "v_fc1_w" : self.v_fc1_weight,
                "v_fc2_w" : self.v_fc2_weight,
                "v_fc1_b" : self.v_fc1_bias,
                "v_fc2_b" : self.v_fc2_bias,
                "v_embed" : self.v_embed
            })

        self.time_step = self.epoch

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
            self.embed = xavier_init(self.vocab_length, self.embedding_dim)
            self.theta1 = xavier_init(self.input_size, self.hidden_size)
            self.bias1 = np.zeros(self.hidden_size)
            self.theta2 = xavier_init(self.hidden_size, self.output_size)
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
        self.adam_init()

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
    def sgd_update(self):
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
    def adam_update(self):
        import numpy as np
        if (self.count > 0):
            def adam(m, v, grad, lr, timestep):
                # printf("***********************************")
                # printf(timestep)
                # timestep = 1
                # bias_correction1 = 1 - 0.9 ** timestep
                # bias_correction2 = 1 - 0.999 *** timestep
                # bias_correction2_sqrt = bias_correction2 ** 0.5



                m = 0.9 * m + (1 - 0.9) * grad
                v = 0.999 * v + (1 - 0.999) * (grad ** 2)


                m_hat = m / (1 - 0.9 ** timestep)
                v_hat = v / (1 - 0.999 ** timestep)
                v_hat = np.maximum(v_hat, 0)
                # printf(np.sqrt(v_hat))
                update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                # printf(m_hat)

                return m, v, update
            # printf(self.total_fc1_weight_grads / self.count)
            self.time_step = self.epoch + 1
            # self.time_step = 1
            self.m_fc1_weight, self.v_fc1_weight, fc1_weight_update = adam(
                self.m_fc1_weight, self.v_fc1_weight, self.total_fc1_weight_grads / self.count, self.fc1_lr, self.time_step
            )
            # printf(fc1_weight_update)
            self.m_fc1_bias, self.v_fc1_bias, fc1_bias_update = adam(
                self.m_fc1_bias, self.v_fc1_bias, self.total_fc1_bias_grads / self.count, self.fc1_lr, self.time_step
            )
            self.m_fc2_weight, self.v_fc2_weight, fc2_weight_update = adam(
                self.m_fc2_weight, self.v_fc2_weight, self.total_fc2_weight_grads / self.count, self.fc2_lr, self.time_step
            )
            self.m_fc2_bias, self.v_fc2_bias, fc2_bias_update = adam(
                self.m_fc2_bias, self.v_fc2_bias, self.total_fc2_bias_grads / self.count, self.fc2_lr, self.time_step
            )
            self.m_embed, self.v_embed, embed_update = adam(
                self.m_embed, self.v_embed, self.total_embedding_grads / self.count, self.embed_lr, self.time_step
            )
            self.theta1 -= fc1_weight_update
            self.theta2 -= fc2_weight_update
            self.bias1 -= fc1_bias_update
            self.bias2 -= fc2_bias_update
            self.embed -= embed_update
            self.log = {
                "theta1" : fc1_weight_update.tolist(),
                "theta2" : fc2_weight_update.tolist(),
                "bias1" : fc1_bias_update.tolist(),
                "bias2" : fc2_bias_update.tolist(),
                "embed" : embed_update.tolist()
            }
    def reducer_final(self):
        self.adam_update()
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
                "bias2" : self.bias2.tolist(),
                "m_fc1_w" : self.m_fc1_weight.tolist(),
                "m_fc2_w" : self.m_fc2_weight.tolist(),
                "m_fc1_b" : self.m_fc1_bias.tolist(),
                "m_fc2_b" : self.m_fc2_bias.tolist(),
                "m_embed" : self.m_embed.tolist(),
                "v_fc1_w" : self.v_fc1_weight.tolist(),
                "v_fc2_w" : self.v_fc2_weight.tolist(),
                "v_fc1_b" : self.v_fc1_bias.tolist(),
                "v_fc2_b" : self.v_fc2_bias.tolist(),
                "v_embed" : self.v_embed.tolist(),
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
                    "bias2" : np.array(weight["bias2"]),
                    "m_fc1_w" : np.array(weight["m_fc1_w"]),
                    "m_fc2_w" : np.array(weight["m_fc2_w"]),
                    "m_fc1_b" : np.array(weight["m_fc1_b"]),
                    "m_fc2_b" : np.array(weight["m_fc2_b"]),
                    "m_embed" : np.array(weight["m_embed"]),
                    "v_fc1_w" : np.array(weight["v_fc1_w"]),
                    "v_fc2_w" : np.array(weight["v_fc2_w"]),
                    "v_fc1_b" : np.array(weight["v_fc1_b"]),
                    "v_fc2_b" : np.array(weight["v_fc2_b"]),
                    "v_embed" : np.array(weight["v_embed"])
                }
if __name__ == "__main__":
    load_weight()
    start_time = time.time()
    TextClassifierTrainer.run()
    printf(f"Finished {time.time() - start_time}s")
