from neural_network import TextClassifierTrainer, load_weight
from neural_network import init as train_init
from neural_network import get_vocab as init_vocab
import logging
import math

logging.getLogger('mrjob').setLevel(logging.ERROR)

train_data_path = "data/ag_news_data/train.csv"
test_data_path = "data/ag_news_data/test.csv"

from neural_network_eval import TextClassifierEvaluate
from neural_network_eval import init as eval_init

init_vocab(train_data_path)
train_init(lr=1e0)

n_epoch = 10
full_data = []
with open(train_data_path, 'r') as file:
    full_data = file.readlines()
batch_size = 1024
n_batch = math.ceil(len(full_data)/batch_size)
temp_data_path = "data/ag_news_data/temp.csv" 
print(n_batch)
def printf(text):
    logging.debug(text)
for epoch in range(n_epoch):
    for batch in range(n_batch):
        log__str = f"Epoch: {epoch+1} | Batch: {batch+1}/{n_batch+1}"
        batch_data = full_data[batch*batch_size:min((batch+1)*batch_size, len(full_data))]
        with open(temp_data_path, 'w') as file:
            file.writelines(batch_data)
        if not (epoch == 0 and batch == 0):
            load_weight()
            printf(log__str)
        else:
            print(log__str)

        job = TextClassifierTrainer(args=[temp_data_path])
        # printf(job.theta2[0])
        with job.make_runner() as runner:
            runner.run()
        
        printf("---Train")
        eval_init(train_data_path)
        job = TextClassifierEvaluate(args=[train_data_path])
        printf(job.theta2[0])
        with job.make_runner() as runner:
            runner.run()
        printf("---Test")
        job = TextClassifierEvaluate(args=[test_data_path])
        with job.make_runner() as runner:
            runner.run()




