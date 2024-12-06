from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, JSONValueProtocol, RawValueProtocol, PickleProtocol, PickleValueProtocol
import time
import logging
import os
import util
import config as cf

class TextDataProcess(MRJob):
    FILES = ["util.py", "config.py"]
    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol if cf.MODE == "pickle" else JSONProtocol
    OUTPUT_PROTOCOL = PickleValueProtocol if cf.MODE == "pickle" else JSONValueProtocol
    def configure_args(self):
        super(TextDataProcess, self).configure_args()
        self.add_passthru_arg("--is_train", type=int, help="Is process train data")
    def __init__(self, *args, **kwargs):
        super(TextDataProcess, self).__init__(*args, **kwargs)
        vocab = util.get_vocab()
        self.vocab_length = len(vocab)
        self.seq_length = cf.SEQ_LENGTH
        self.split_factor = cf.CLUSTERS
        self.output_size = cf.OUTPUT_SIZE
        if self.options.is_train == 1:
            self.output_folder = cf.TRAIN_DATA_PATH
        elif self.options.is_train == 0:
            self.output_folder = cf.TEST_DATA_PATH
        else:
            raise Exception(f"Invalid or missing arg is_train {self.options.is_train}")
        self.extract_func = util.extract

        self.word2idx = {w:i for i,w in enumerate(vocab)}
        self.idx2word = {i:w for i,w in enumerate(vocab)}
        logging.basicConfig(level=logging.DEBUG)
        self.line_count = 0
    def mapper(self, key, line: str):
        data = line.split(",")
        if len(data) < 3: return
        # label_text = data[-1]
        label = data[-2]
        text = ",".join(data[:-2])
        if text == "text": return
        else:
            key = None
            if self.split_factor > 1:
                key = self.line_count
                self.line_count += 1
                key %= self.split_factor
            text = self.extract_func(text)
            if len(text) < self.seq_length:
                text.extend(["<pad>" for _ in range(self.seq_length - len(text))])
            text = text[:self.seq_length]
            input = [self.word2idx.get(w, self.word2idx['<unk>']) for w in text]
            label = int(label)
            one_hot_label = [1 if i == label else 0 for i in range(self.output_size)]
            yield key, [input, one_hot_label]
    def reducer(self, key, records):
        batch_X, batch_y = [], []
        for data in records:
            batch_X.append(data[0])
            batch_y.append(data[1])
        if len(batch_X) > 0:
            data = [key, batch_X, batch_y]
            if cf.FORCE_JSON_INPUT:
                file_path = os.path.join(self.output_folder, f"{key}.json")
            else:
                file_path = os.path.join(self.output_folder, f"{key}.{cf.FILE_EXTENSION}")
            util.writer(data, file_path)

if __name__ == "__main__":
    from util import printf
    start_time = time.time()
    TextDataProcess.run()
    printf(f"Finished {time.time() - start_time}s")
