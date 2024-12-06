from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, JSONValueProtocol, PickleProtocol, PickleValueProtocol
import time
import logging
import os
import config as cf
import util

class TextClassifierEvaluate(MRJob):
    FILES = ["util.py", "config.py"]
    INPUT_PROTOCOL = JSONValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol if cf.MODE == "pickle" else JSONProtocol
    OUTPUT_PROTOCOL = PickleValueProtocol if cf.MODE == "pickle" else JSONValueProtocol
    def __init__(self, *args, **kwargs):
        super(TextClassifierEvaluate, self).__init__(*args, **kwargs)
        vocab = util.get_vocab()
        self.vocab_length = len(vocab)

        self.embedding_dim = cf.EMBED_DIM
        self.seq_length = cf.SEQ_LENGTH
        self.input_size = cf.SEQ_LENGTH * cf.EMBED_DIM
        self.hidden_size = cf.HIDDEN_SIZE
        self.output_size = cf.OUTPUT_SIZE

        self.total_loss = 0
        self.total_accuracy = 0

        logging.basicConfig(level=logging.DEBUG)

        self.load_checkpoint()
        self.count = 0
    def load_checkpoint(self):
        weight_path = os.path.join(cf.ABS_OUTPUT_PATH, f"model_weight.{cf.FILE_EXTENSION}")
        if os.path.exists(weight_path):
            weight = util.reader(weight_path, True)
            self.embed = weight["embed"]
            self.fc1 = weight["fc1"]
            self.fc1_b = weight["fc1_b"]
            self.fc2 = weight["fc2"]
            self.fc2_b = weight["fc2_b"]
        metadata_path = os.path.join(cf.ABS_OUTPUT_PATH, "metadata.json")
        if os.path.exists(metadata_path):
            metadata = util.reader(metadata_path)
            self.epoch = metadata["epoch"]
            self.beta1 = metadata["beta1"]
            self.beta2 = metadata["beta2"]
            self.epsilon = metadata["epsilon"]
    def save_log(self):
        file_path = os.path.join(cf.ABS_OUTPUT_PATH, "evaluate.json")
        data = {}
        if (os.path.exists(file_path)):
            data = util.reader(file_path)
        data[self.epoch] = self.metrics
        util.writer(data, file_path)
    def mapper(self, _, data: object):
        import numpy as np
        if len(data) > 0:
            key = data[0]
            X_np = data[1]
            y_np = data[2]
            if cf.MODE == "json" or cf.FORCE_JSON_INPUT:
                X_np = np.array(X_np)
                y_np = np.array(y_np)
            probabilities = util.forward(X_np, self.embed, self.fc1, self.fc2, self.fc1_b, self.fc2_b)
            batch_size = y_np.shape[0]
            probabilities = np.clip(probabilities, 1e-12, 1e12)
            loss = -np.sum(y_np * np.log(probabilities)) / batch_size
            predicted = np.argmax(probabilities, axis=1)
            true_class = np.argmax(y_np, axis=1)
            accuracy = np.mean(predicted == true_class)
            yield key, {
                "loss" : loss,
                "accuracy" : accuracy
                }
    def combiner(self, _, metric_datas):
        total_loss = 0
        total_accuracy = 0
        count = 0
        for metric_data in metric_datas:
            total_loss += metric_data["loss"]
            total_accuracy += metric_data["accuracy"]
            count += 1
            yield None, {
                "loss" : total_loss / count,
                "accuracy" : total_accuracy / count
            }
    def reducer(self, _, metric_datas):
        for metric_data in metric_datas:
            self.total_loss += metric_data["loss"]
            self.total_accuracy += metric_data["accuracy"]
            self.count += 1

    def reducer_final(self):
        self.metrics = {
            "loss" : self.total_loss/self.count,
            "accuracy" : self.total_accuracy/self.count
        }
        self.save_log()
        import util
        util.printf("**********************END**********************")


if __name__ == "__main__":
    import util
    start_time = time.time()
    TextClassifierEvaluate.run()
    util.printf(f"Finished {time.time() - start_time}s")
