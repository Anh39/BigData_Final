from mrjob.job import MRJob
import json
import time
import logging
import random
import os

def sigmoid(z):
    import numpy as np

    z = z.clip(-700, 700)
    result = 1 / (1+np.exp(-z))
    return result

def printf(text):
    logging.debug(text)
def soft_max(z):
    import numpy as np
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
WEIGHT = {}


class EmbeddingTrainer(MRJob):
    FILES = ["util.py"]
    def __init__(self, *args, **kwargs):

        super(EmbeddingTrainer, self).__init__(*args, **kwargs)
        from util import get_vocab, EMBED_DIM, SEQ_LENGTH, EMBED_WINDOW, EMBED_LR
        self.vocab = get_vocab()
        self.vocab_length = len(self.vocab)
        self.embedding_dim = EMBED_DIM
        self.seq_length = SEQ_LENGTH
        self.window = EMBED_WINDOW
        self.lr = EMBED_LR
        self.load_weight()
        self.word2idx = {w:i for i,w in enumerate(self.vocab)}
        self.idx2word = {i:w for i,w in enumerate(self.vocab)}
        logging.basicConfig(level=logging.DEBUG)
        self.line_count = 0
    def load_weight(self):
        global WEIGHT
        if ("embed" in WEIGHT):
            self.embed = WEIGHT["embed"]
            self.neg_embed = WEIGHT["neg_embed"]
        else:
            import numpy as np
            self.embed = np.random.rand(self.vocab_length, self.embedding_dim) * 0.001 - 0.0005
            self.neg_embed = np.random.rand(self.vocab_length, self.embedding_dim) * 0.001 - 0.0005
            self.reducer_final()
    def mapper(self, key, line: str):
        from util import extract
        # key = self.line_count
        # self.line_count += 1
        data = line.split(",")
        if len(data) < 3: return
        text = ",".join(data[:-2])
        if text == "text": return
        else:
            text = extract(text)
            if len(text) < self.seq_length:
                text.extend(["<pad>" for _ in range(self.seq_length - len(text))])
            text = text[:self.seq_length]
            text = [self.word2idx[w] for w in text]
            for i in range(self.window, self.seq_length-self.window):
                for offset in range(-self.window, self.window+1):
                    if (offset == 0): continue
                    target = text[i]
                    context = text[i+offset]
                    neg_context = context
                    while (neg_context == context):
                        neg_context = self.word2idx[self.vocab[random.randint(0,self.vocab_length-1)]]
                    yield target, context
                    yield -target, neg_context
    def combiner(self, key, records): 
        import numpy as np
        key = int(key)
        target_vector = self.embed[key]
        for record in records:
            if key > 0:
                context = int(record)
                neg_vector = self.neg_embed[context]
                pos_score = sigmoid(np.dot(target_vector, neg_vector))
                pos_grad = pos_score - 1
                yield key, pos_grad
            else:
                neg_context = int(record)
                neg_vector = self.neg_embed[neg_context]
                neg_score = sigmoid(np.dot(target_vector, neg_vector))
                neg_grad = 1 - neg_score
                yield key, neg_grad
    def reducer(self, key, grads):
        import numpy as np
        key = abs(int(key))
        total_grad = 0
        for grad_ in grads:
            total_grad += grad_
        self.embed[key] -= self.lr * grad_ * self.embed[key]            
    def reducer_final(self):
        # printf("-----------------")
        from util import get_abs_output_path
        file_path = os.path.join(get_abs_output_path(),"embed_weight.json")
        log = {
            "lr" : self.lr,
            "seq_length" : self.seq_length,
            "embed_dim" : self.embedding_dim,
            "vocab_length" : self.vocab_length,
            "embed_size" : list(self.embed.shape),
            "neg_embed_size" : list(self.neg_embed.shape),
            "embed" : self.embed.tolist(),
            "neg_embed" : self.neg_embed.tolist()
        }
        data = json.dumps(log)
        with open(file_path, 'w') as file:
            file.write(data)
        # printf(file_path)
def load_checkpoint():
    global WEIGHT
    import numpy as np
    from util import get_abs_output_path
    path = os.path.join(get_abs_output_path(), "embed_weight.json")
    if os.path.exists(path):
        with open(path, 'r') as file:
            weight = json.loads(file.read())
            if ("embed" in weight and "neg_embed" in weight):
                WEIGHT = {
                    "embed" : np.array(weight["embed"]),
                    "neg_embed" : np.array(weight["neg_embed"])
                }
if __name__ == "__main__":
    load_checkpoint()
    start_time = time.time()
    EmbeddingTrainer.run()
    printf(f"Finished {time.time() - start_time}s")
