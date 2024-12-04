from neural_network_eval import TextClassifierEvaluate, init

init()
job = TextClassifierEvaluate(args=['data/ag_news_data/train.csv'])
with job.make_runner() as runner:
    runner.run()