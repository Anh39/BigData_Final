from neural_network import TextClassifierTrainer, init, load_weight

init()
load_weight()

job = TextClassifierTrainer(args=['data/ag_news_data/test.csv'])
n_epoch = 2
for epoch in range(n_epoch):
    load_weight()
    with job.make_runner() as runner:
        job.load_weight()
        runner.run()
