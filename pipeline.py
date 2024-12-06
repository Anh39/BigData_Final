from util import network_initializer
from data_process import TextDataProcess
from neural_network import TextClassifierTrainer
from neural_network_eval import TextClassifierEvaluate
import logging
import os
import time
import config as cf

logging.getLogger('mrjob').setLevel(logging.WARNING)

extra_args = ["-cmrjob.conf", "-rlocal"]
n_epochs = 50
def run_new():
    import shutil
    folder_paths = [
        "output/train",
        "output/test"
    ]
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    file_paths = [
        f"output/grads.{cf.FILE_EXTENSION}",
        "output/vocab.json",
        f"output/model_weight.{cf.FILE_EXTENSION}",
        "output/metadata.json",
        "output/evaluate.json"
    ]
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
    network_initializer()
    args = [cf.RAW_TRAIN_DATA_PATH, "--is_train", "1"]
    args.extend(extra_args)
    job = TextDataProcess(args=args)
    with job.make_runner() as runner:
        runner.run()
    args = [cf.RAW_TEST_DATA_PATH, "--is_train", "0"]
    args.extend(extra_args)
    job = TextDataProcess(args=args)
    with job.make_runner() as runner:
        runner.run()
run_new()
for epoch in range(n_epochs):
    log_str = f"Epoch: {epoch+1}"
    print(log_str)
    args = [cf.TRAIN_DATA_PATH]
    args.extend(extra_args)
    start_time = time.time()
    job = TextClassifierTrainer(args=args)
    with job.make_runner() as runner:
        runner.run()
    print(f"Train time {time.time()-start_time} s")
    args = [cf.TEST_DATA_PATH]
    args.extend(extra_args)
    job = TextClassifierEvaluate(args=args)
    with job.make_runner() as runner:
        runner.run()


