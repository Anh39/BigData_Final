from embed_network_adam import TextClassifierTrainer
from embed_network_adam import load_weight as train_load_weight
from embed_network_eval import TextClassifierEvaluate
from embed_network_eval import load_weight as test_load_weight
import logging
import os
import time

logging.getLogger('mrjob').setLevel(logging.ERROR)


train_data_path = "data/ag_news_data/train.csv"
test_data_path = "data/ag_news_data/test.csv"
extra_args = ["-cmrjob.conf", "-rlocal"]
n_epochs = 10
def base_eval():
    from util import ABS_OUTPUT_PATH
    if not os.path.exists(os.path.join(ABS_OUTPUT_PATH, "evaluate.json")):
        test_load_weight()
        args = [train_data_path]
        args.extend(extra_args)
        job = TextClassifierEvaluate(args=args)
        with job.make_runner() as runner:
            runner.run()
base_eval()
for epoch in range(n_epochs):
    log_str = f"Epoch: {epoch+1}"
    print(log_str)
    train_load_weight()
    args = [train_data_path]
    args.extend(extra_args)
    start_time = time.time()
    job = TextClassifierTrainer(args=args)
    with job.make_runner() as runner:
        runner.run()
    print(f"Train time {time.time()-start_time} s")
    test_load_weight()
    args = [test_data_path]
    args.extend(extra_args)
    job = TextClassifierEvaluate(args=args)
    with job.make_runner() as runner:
        runner.run()


