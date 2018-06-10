#!/bin/bash

# run.sh

mkdir -p {data,models,results}

# --
# IO

wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip && mv ml-20m data && rm ml-20m.zip

python prep.py --inpath data/ml-20m/ratings.csv --outpath data
