from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol, JSONValueProtocol, PickleProtocol, PickleValueProtocol
import time
import logging
import os
import config as cf
import util

class TextClassifierTrainer(MRJob):
    FILES = ["util.py", "config.py"]
    INPUT_PROTOCOL = JSONValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol if cf.MODE == "pickle" else JSONProtocol
    OUTPUT_PROTOCOL = PickleValueProtocol if cf.MODE == "pickle" else JSONValueProtocol
    def __init__(self, *args, **kwargs):
        super(TextClassifierTrainer, self).__init__(*args, **kwargs)
        vocab = util.get_vocab()
        self.vocab_length = len(vocab)

        self.embedding_dim = cf.EMBED_DIM
        self.seq_length = cf.SEQ_LENGTH
        self.input_size = cf.SEQ_LENGTH * cf.EMBED_DIM
        self.hidden_size = cf.HIDDEN_SIZE
        self.output_size = cf.OUTPUT_SIZE

        self.embed_lr = cf.EMBED_LR
        self.fc1_lr = cf.FC1_LR
        self.fc2_lr = cf.FC2_LR

        logging.basicConfig(level=logging.DEBUG)

        self.load_checkpoint()
        import numpy as np
        self.total_fc1_weight_grads = np.zeros_like(self.fc1)
        self.total_fc2_weight_grads = np.zeros_like(self.fc2)
        self.total_fc1_bias_grads = np.zeros_like(self.fc1_b)
        self.total_fc2_bias_grads = np.zeros_like(self.fc2_b)
        self.total_embedding_grads = np.zeros_like(self.embed)
        self.count = 0
        self.line_count = 0
    def load_checkpoint(self):
        weight_path = os.path.join(cf.ABS_OUTPUT_PATH, f"model_weight.{cf.FILE_EXTENSION}")
        if os.path.exists(weight_path):
            weight = util.reader(weight_path, True)
            self.embed = weight["embed"]
            self.fc1 = weight["fc1"]
            self.fc1_b = weight["fc1_b"]
            self.fc2 = weight["fc2"]
            self.fc2_b = weight["fc2_b"]
        grad_path = os.path.join(cf.ABS_OUTPUT_PATH, f"grads.{cf.FILE_EXTENSION}")
        if os.path.exists(grad_path):
            grads = util.reader(grad_path, True)
            self.embed_m = grads["embed_m"]
            self.embed_v = grads["embed_v"]
            self.fc1_m = grads["fc1_m"]
            self.fc1_v = grads["fc1_v"]
            self.fc2_m = grads["fc2_m"]
            self.fc2_v = grads["fc2_v"]
            self.fc1_b_m = grads["fc1_b_m"]
            self.fc1_b_v = grads["fc1_b_v"]
            self.fc2_b_m = grads["fc2_b_m"]
            self.fc2_b_v = grads["fc2_b_v"]
        metadata_path = os.path.join(cf.ABS_OUTPUT_PATH, "metadata.json")
        if os.path.exists(metadata_path):
            metadata = util.reader(metadata_path)
            self.epoch = metadata["epoch"]
            self.beta1 = metadata["beta1"]
            self.beta2 = metadata["beta2"]
            self.epsilon = metadata["epsilon"]
    def save_checkpoint(self):
        attr_name_map = {
            f"model_weight.{cf.FILE_EXTENSION}" : [
                "embed", "fc1", "fc1_b", "fc2", "fc2_b"
            ],
            f"grads.{cf.FILE_EXTENSION}" : [
                "embed_m", "embed_v", "fc1_m", "fc1_v", "fc2_m", "fc2_v", "fc1_b_m", "fc1_b_v", "fc2_b_m", "fc2_b_v"
            ],
            "metadata.json" : [
                "epoch", "beta1", "beta2", "epsilon"
            ]
        }
        for file_name in attr_name_map:
            attr_names = attr_name_map[file_name]
            file_data = {}
            for attr_name in attr_names:
                data = self.__getattribute__(attr_name)
                file_data[attr_name] = data
            file_path = os.path.join(cf.ABS_OUTPUT_PATH, file_name)
            util.writer(file_data, file_path, True)
    def mapper(self, _, data: str):
        if cf.MODE == "json" or cf.FORCE_JSON_INPUT:
            import numpy as np
            if len(data) > 0:
                key = data[0]
                X_np = np.array(data[1])
                y_np = np.array(data[2])
                embedding_grad, fc1_bias_grad, fc1_weight_grad, fc2_bias_grad, fc2_weight_grad = util.backward(X_np, y_np, self.embed, self.fc1, self.fc2, self.fc1_b, self.fc2_b)
                yield key, {
                    "embedding_grad" : embedding_grad.tolist(),
                    "fc1_bias_grad" : fc1_bias_grad.tolist(),
                    "fc1_weight_grad" : fc1_weight_grad.tolist(),
                    "fc2_bias_grad" : fc2_bias_grad.tolist(),
                    "fc2_weight_grad" : fc2_weight_grad.tolist()
                    }
        else:
            if len(data) > 0:
                key = data[0]
                X_np = data[1]
                y_np = data[2]
                embedding_grad, fc1_bias_grad, fc1_weight_grad, fc2_bias_grad, fc2_weight_grad = util.backward(X_np, y_np, self.embed, self.fc1, self.fc2, self.fc1_b, self.fc2_b)
                yield key, {
                    "embedding_grad" : embedding_grad,
                    "fc1_bias_grad" : fc1_bias_grad,
                    "fc1_weight_grad" : fc1_weight_grad,
                    "fc2_bias_grad" : fc2_bias_grad,
                    "fc2_weight_grad" : fc2_weight_grad
                    }
    def combiner(self, key, grads):
        import numpy as np
        total_fc1_weight_grads = np.zeros_like(self.fc1)
        total_fc2_weight_grads = np.zeros_like(self.fc2)
        total_fc1_bias_grads = np.zeros_like(self.fc1_b)
        total_fc2_bias_grads = np.zeros_like(self.fc2_b)
        total_embedding_grads = np.zeros_like(self.embed)
        count = 0
        if cf.MODE == "json":
            for grad_data in grads:
                total_fc1_weight_grads += np.array(grad_data["fc1_weight_grad"])
                total_fc1_bias_grads += np.array(grad_data["fc1_bias_grad"])
                total_fc2_weight_grads += np.array(grad_data["fc2_weight_grad"])
                total_fc2_bias_grads += np.array(grad_data["fc2_bias_grad"])
                total_embedding_grads += np.array(grad_data["embedding_grad"])
                count += 1
                yield None, {
                    "embedding_grad" : (total_embedding_grads/count).tolist(),
                    "fc1_bias_grad" : (total_fc1_bias_grads/count).tolist(),
                    "fc1_weight_grad" : (total_fc1_weight_grads/count).tolist(),
                    "fc2_bias_grad" : (total_fc2_bias_grads/count).tolist(),
                    "fc2_weight_grad" : (total_fc2_weight_grads/count).tolist()
                    }
        else:
            for grad_data in grads:
                total_fc1_weight_grads += grad_data["fc1_weight_grad"]
                total_fc1_bias_grads += grad_data["fc1_bias_grad"]
                total_fc2_weight_grads += grad_data["fc2_weight_grad"]
                total_fc2_bias_grads += grad_data["fc2_bias_grad"]
                total_embedding_grads += grad_data["embedding_grad"]
                count += 1
                yield None, {
                    "embedding_grad" : (total_embedding_grads/count),
                    "fc1_bias_grad" : (total_fc1_bias_grads/count),
                    "fc1_weight_grad" : (total_fc1_weight_grads/count),
                    "fc2_bias_grad" : (total_fc2_bias_grads/count),
                    "fc2_weight_grad" : (total_fc2_weight_grads/count)
                    }
    def reducer(self, _, grads):
        if cf.MODE == "json":
            import numpy as np
            for grad_data in grads:
                self.total_fc1_weight_grads += np.array(grad_data["fc1_weight_grad"])
                self.total_fc1_bias_grads += np.array(grad_data["fc1_bias_grad"])
                self.total_fc2_weight_grads += np.array(grad_data["fc2_weight_grad"])
                self.total_fc2_bias_grads += np.array(grad_data["fc2_bias_grad"])
                self.total_embedding_grads += np.array(grad_data["embedding_grad"])
                self.count += 1
        else:
            for grad_data in grads:
                self.total_fc1_weight_grads += grad_data["fc1_weight_grad"]
                self.total_fc1_bias_grads += grad_data["fc1_bias_grad"]
                self.total_fc2_weight_grads += grad_data["fc2_weight_grad"]
                self.total_fc2_bias_grads += grad_data["fc2_bias_grad"]
                self.total_embedding_grads += grad_data["embedding_grad"]
                self.count += 1

    def adam_update(self):
        import numpy as np
        if (self.count > 0):
            def adam(m, v, grad, lr, timestep):
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
            time_step = self.epoch + 1
            # self.time_step = 1
            self.fc1_m, self.fc1_v, fc1_weight_update = adam(
                self.fc1_m, self.fc1_v, self.total_fc1_weight_grads / self.count, self.fc1_lr, time_step
            )
            self.fc2_m, self.fc2_v, fc2_weight_update = adam(
                self.fc2_m, self.fc2_v, self.total_fc2_weight_grads / self.count, self.fc2_lr, time_step
            )
            self.fc1_b_m, self.fc1_b_v, fc1_bias_update = adam(
                self.fc1_b_m, self.fc1_b_v, self.total_fc1_bias_grads / self.count, self.fc1_lr, time_step
            )
            self.fc2_b_m, self.fc2_b_v, fc2_bias_update = adam(
                self.fc2_b_m, self.fc2_b_v, self.total_fc2_bias_grads / self.count, self.fc2_lr, time_step
            )
            self.embed_m, self.embed_v, embed_update = adam(
                self.embed_m, self.embed_v, self.total_embedding_grads / self.count, self.fc2_lr, time_step
            )
            self.fc1 -= fc1_weight_update
            self.fc2 -= fc2_weight_update
            self.fc1_b -= fc1_bias_update
            self.fc2_b -= fc2_bias_update
            self.embed -= embed_update
    def reducer_final(self):
        self.adam_update()
        self.epoch += 1
        self.save_checkpoint()
        import util
        util.printf("**********************END**********************")

if __name__ == "__main__":
    import util
    start_time = time.time()
    TextClassifierTrainer.run()
    util.printf(f"Finished {time.time() - start_time}s")
