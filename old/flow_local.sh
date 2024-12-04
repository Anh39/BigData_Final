python3 embedding.py -c mrjob.conf -r local data/ag_news_data/train.csv
python3 neural_network_embed.py -c mrjob.conf -r local data/ag_news_data/train.csv
python3 neural_network_embed_eval.py -c mrjob.conf -r local data/ag_news_data/train.csv