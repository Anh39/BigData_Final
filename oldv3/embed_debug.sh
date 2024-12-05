# python3 embed_network_adam.py --mapper < data/ag_news_data/train.csv > mapper_output.txt
# python3 embed_network_adam.py --combiner < mapper_output.txt > combiner_output.txt
python3 embed_network_adam.py --reducer < combiner_output.txt
