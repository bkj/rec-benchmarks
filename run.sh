#!/bin/bash

# run.sh

mkdir -p {data,models,results}


# --
# Installation

conda create -n rec_env python==3.6 pip -y
source activate rec_env

pip install implicit
pip install lightfm
conda install -c maciejkula -c pytorch spotlight=0.1.5 -y
# Also need to install `https://github.com/quora/qmf`
# Also need to install `https://github.com/bkj/basenet`

# --
# IO

# Movielens
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip && mv ml-20m data && rm ml-20m.zip
python prep_ml.py --inpath data/ml-20m/ratings.csv --outpath data/ml-20m

# Netflix
# wget from HIVE server

python prep_netflix.py --inpath data/netflix/netflix.tsv --outpath data/netflix