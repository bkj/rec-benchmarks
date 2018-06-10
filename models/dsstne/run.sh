#!/bin/bash

# run.sh

mkdir -p {data,models}

export PATH="/home/bjohnson/software/amazon-dsstne/src/amazon/dsstne/bin/:$PATH"

export TRAIN_PATH='../../data/dsstne-train.txt'
export TEST_PATH='../../data/dsstne-train.txt'

function prep_data {
    generateNetCDF -d gl_input -i $TRAIN_PATH -o data/gl_input.nc -f data/features_input -s data/samples_input -c
    generateNetCDF -d gl_output -i $TRAIN_PATH -o data/gl_output.nc -f data/features_output -s data/samples_input -c    
}
function run {
    NUM_EPOCHS=50
    BATCH_SIZE=1024

    rm models/*
    train -b $BATCH_SIZE -e $NUM_EPOCHS -n models/network.nc \
        -d gl \
        -i data/gl_input.nc \
        -o data/gl_output.nc \
        -c config.json

    predict -b 2048 -k 10 -n models/network.nc \
        -d gl \
        -i data/features_input \
        -o data/features_output \
        -f $TRAIN_PATH \
        -r $TRAIN_PATH \
        -s results/recs

    head results/recs
    python inspect-results.py $TEST_PATH results/recs
}

# prep_data # Nee to run this first
run