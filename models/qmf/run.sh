#!/bin/bash

# run.sh

# --
# Install QMF

# git clone https://github.com/quora/qmf
# cd qmf
# cmake .
# make

mkdir -p results/qmf
/home/bjohnson/software/qmf/bin/wals \
    --train_dataset=data/edgelist-train.tsv \
    --test_dataset=data/edgelist-test.tsv \
    --user_factors=results/qmf/user_factors \
    --item_factors=results/qmf/item_factors \
    --regularization_lambda=0.05 \
    --confidence_weight=40 \
    --nepochs=10 \
    --nfactors=30 \
    --nthreads=12 \
    --test_avg_metrics=p@1,p@5,p@10

head results/qmf/user_factors