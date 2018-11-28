#!/usr/bin/env python

"""
    prep.py
"""

from __future__ import print_function, division

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/netflix/netflix.tsv')
    parser.add_argument('--outpath', type=str, default='data/netflix')
    parser.add_argument('--seed', type=int, default=456)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    print('loading %s' % args.inpath, file=sys.stderr)
    edges = pd.read_csv(args.inpath, header=None, sep='\t')
    edges.columns = ['userId', 'movieId', 'movieRating']
    
    edges.userId      = pd.Categorical(edges.userId).codes
    edges.movieId     = pd.Categorical(edges.movieId).codes
    edges.movieRating = 1
    
    train, test = train_test_split(edges, train_size=0.8, stratify=edges.userId)
    
    train = train.sort_values(['userId', 'movieId']).reset_index(drop=True)
    test  = test.sort_values(['userId', 'movieId']).reset_index(drop=True)
    
    train_users = set(train.userId)
    test_users  = set(test.userId)
    
    no_train = test_users.difference(train_users)
    if len(no_train):
        sel = test.userId.isin(no_train)
        test = test[~sel]
    
    train.to_csv(os.path.join(args.outpath, 'edgelist-train.tsv'), sep='\t', header=None)
    test.to_csv(os.path.join(args.outpath, 'edgelist-test.tsv'), sep='\t', header=None)