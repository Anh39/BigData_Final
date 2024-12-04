from neural_network import TextClassifierTrainer, load_weight
from neural_network import init as train_init
from neural_network import get_vocab as init_vocab
import logging

logging.getLogger('mrjob').setLevel(logging.ERROR)

train_data_path = "data/ag_news_data/test.csv"
test_data_path = "data/ag_news_data/test.csv"

from neural_network_eval import TextClassifierEvaluate
from neural_network_eval import init as eval_init

init_vocab(train_data_path)
train_init()
eval_init(train_data_path)


n_epoch = 10
for epoch in range(n_epoch):
    job = TextClassifierTrainer(args=[train_data_path])
    if epoch != 0:
        load_weight(train_data_path)
        # job.load_weight()
    with job.make_runner() as runner:
        runner.run()
    job = TextClassifierEvaluate(args=[test_data_path])
    with job.make_runner() as runner:
        runner.run()




