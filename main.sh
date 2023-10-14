python3 create_graph.py PL 
python3 create_graph.py ALL 
mkdir random_walks_pl random_walks_all
sh random_walks.sh
cat random_walks_pl/* > data/random_walks_pl.txt
cat random_walks_all/* > data/random_walks_all.txt
mkdir wordnet_embeddings
fasttext skipgram -input data/random_walks_pl.txt -output wordnet_embeddings/random_walks_pl -maxn 0 -ws 7 -thread 24 -dim 300 -neg 10
fasttext skipgram -input data/random_walks_all.txt -output wordnet_embeddings/random_walks_all -maxn 0 -ws 7 -thread 32 -dim 300 -neg 10
python3 get_emotions.py
python3 generate_nn_dict.py
python3 generate_folds.py
sh run_all.sh
