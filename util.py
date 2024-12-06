import re, os, json, logging, pickle
from config import *
def printf(text):
    logging.debug(text)
def extract(text: str) -> list[str]:
    return re.sub(r'[^a-z\s]', '', text.lower()).split()
__VOCAB = None
def get_vocab() -> list[str]:
    global __VOCAB
    if __VOCAB != None: return __VOCAB
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'r') as file:
            __VOCAB = json.loads(file.read())
            return __VOCAB
    else:
        with open(RAW_TRAIN_DATA_PATH, 'r') as file:
            lines = file.readlines()
            lines = [",".join(line.split(",")[:-2]) for line in lines]
        for i in range(len(lines)):
            lines[i] = extract(lines[i])[:SEQ_LENGTH]
        vocab_set = set([])
        for line in lines:
            vocab_set.update(line)
        vocab = list(vocab_set)
        vocab.sort()
        __VOCAB = ["<unk>","<pad>"]
        __VOCAB.extend(vocab)
        with open(VOCAB_PATH, 'w') as file:
            file.write(json.dumps(__VOCAB))
        return __VOCAB
    
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
    # X : np.ndarray = X
    # y : np.ndarray = y
    # embedding : np.ndarray = embedding
    # theta1 : np.ndarray = theta1
    # theta2 : np.ndarray = theta2
    # bias1 : np.ndarray = bias1
    # bias2 : np.ndarray = bias2

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

def network_initializer():
    import numpy as np
    from config import ABS_OUTPUT_PATH
    weight_path = os.path.join(ABS_OUTPUT_PATH, f"model_weight.{FILE_EXTENSION}")
    vocab_len = len(get_vocab())
    if not os.path.exists(weight_path):
        data = {
            "embed" : xavier_init(vocab_len, EMBED_DIM),
            "fc1" : xavier_init(SEQ_LENGTH * EMBED_DIM, HIDDEN_SIZE),
            "fc1_b" : np.zeros(HIDDEN_SIZE),
            "fc2" : xavier_init(HIDDEN_SIZE, OUTPUT_SIZE),
            "fc2_b" : np.zeros(OUTPUT_SIZE)
        }
        writer(data, weight_path, True)
    grad_path = os.path.join(ABS_OUTPUT_PATH, f"grads.{FILE_EXTENSION}")
    if not os.path.exists(grad_path):
        data = {
            "embed_m" : np.zeros((vocab_len, EMBED_DIM)),
            "embed_v" : np.zeros((vocab_len, EMBED_DIM)),
            "fc1_m" : np.zeros((SEQ_LENGTH * EMBED_DIM, HIDDEN_SIZE)),
            "fc1_v" : np.zeros((SEQ_LENGTH * EMBED_DIM, HIDDEN_SIZE)),
            "fc2_m" : np.zeros((HIDDEN_SIZE, OUTPUT_SIZE)),
            "fc2_v" : np.zeros((HIDDEN_SIZE, OUTPUT_SIZE)),
            "fc1_b_m" : np.zeros(HIDDEN_SIZE),
            "fc1_b_v" : np.zeros(HIDDEN_SIZE),
            "fc2_b_m" : np.zeros(OUTPUT_SIZE),
            "fc2_b_v" : np.zeros(OUTPUT_SIZE)
        }
        writer(data, grad_path, True)
    metadata_path = os.path.join(ABS_OUTPUT_PATH, "metadata.json")
    if not os.path.exists(metadata_path):
        data = {
            "epoch" : 0,
            "beta1" : 0.9,
            "beta2" : 0.999,
            "epsilon" : 1e-8
        }
        writer(data, metadata_path)
def convert(data):
    import numpy as np
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    return data
def writer(data: object, file_path: str, auto_convert: bool = False):
    write_mode = 'wb' if file_path.endswith('pkl') else 'w'
    if write_mode == 'w':
        try:
            with open(file_path, write_mode) as file:
                file.write(json.dumps(convert(data) if auto_convert else data))
        except Exception as ex:
            print("Path" + file_path)
            print(ex)

    else:
        with open(file_path, write_mode) as file:
            pickle.dump(data, file)
def reader(file_path: str, internal_ndarray = False):
    read_mode = 'rb' if file_path.endswith('pkl') else 'r'
    if read_mode == 'r':
        with open(file_path, read_mode) as file:
            data = json.loads(file.read())
            if internal_ndarray:
                import numpy as np
                for key in data:
                    data[key] = np.array(data[key])
    else:
        with open(file_path, read_mode) as file:
            data = pickle.load(file)
    return data