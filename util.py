import re, os
def extract(text: str) -> list[str]:
    return re.sub(r'[^a-z\s]', '', text.lower()).split()
VOCAB = None
ABS_OUTPUT_PATH = "/home/hd_user/storage/final/output"
EMBED_DIM = 32
EMBED_WINDOW = 1
SEQ_LENGTH = 32
HIDDEN_SIZE = 512
OUTPUT_SIZE = 4
EMBED_LR = 1e-2
NN_LR = 1e-1
MAPPER_SPLIT = 0
def get_abs_output_path():
    return ABS_OUTPUT_PATH
def get_vocab():
    global VOCAB
    if VOCAB != None: return VOCAB
    with open(os.path.join("/home/hd_user/storage/final", "data/ag_news_data/train.csv"), 'r') as file:
        lines = file.readlines()
        lines = [",".join(line.split(",")[:-2]) for line in lines]
    for i in range(len(lines)):
        lines[i] = extract(lines[i])[:SEQ_LENGTH]
    vocab_set = set([])
    for line in lines:
        vocab_set.update(line)
    vocab = list(vocab_set)
    vocab.sort()

    VOCAB = ["<unk>","<pad>"]
    VOCAB.extend(vocab)
    return VOCAB