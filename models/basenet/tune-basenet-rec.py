#!/usr/bin/env python

"""
    basenet-rec.py
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet, HPSchedule
from basenet.helpers import to_numpy, set_seeds
from basenet.text.data import SortishSampler

from torch.utils.data import Dataset, DataLoader

import dlib

# --
# Helpers

class RaggedAutoencoderDataset(Dataset):
    def __init__(self, X, n_toks):
        self.X = [torch.LongTensor(xx) for xx in X]
        self.n_toks = n_toks
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.zeros((n_toks,))
        y[x] += 1
        return self.X[idx], y
    
    def __len__(self):
        return len(self.X)


def pad_collate_fn(batch, pad_value=0):
    X, y = zip(*batch)
    
    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value).data for xx in X]
    
    X = torch.stack(X, dim=-1).t().contiguous()
    y = torch.stack(y, dim=0)
    return X, y


class DestinyModel(BaseNet):
    def __init__(self, n_toks, emb_dim, dropout, bias_offset):
        
        def _loss_fn(x, y):
            return F.binary_cross_entropy_with_logits(x, y)
        
        super().__init__(loss_fn=_loss_fn)
        
        self.emb = nn.Embedding(n_toks, emb_dim, padding_idx=0)
        
        self.emb_bias   = nn.Parameter(torch.zeros(emb_dim))
        self.bn1        = nn.BatchNorm1d(emb_dim)
        self.dropout1   = nn.Dropout(p=dropout)
        self.hidden     = nn.Linear(emb_dim, emb_dim)
        self.bn2        = nn.BatchNorm1d(emb_dim)
        self.dropout2   = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(emb_dim, n_toks)
        
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01)
        self.emb.weight.data[0] = 0
        
        torch.nn.init.normal_(self.hidden.weight.data, 0, 0.01)
        self.hidden.bias.data.zero_()
        
        torch.nn.init.normal_(self.classifier.weight.data, 0, 0.01)
        self.classifier.bias.data.zero_()
        self.classifier.bias.data += bias_offset
    
    def forward(self, x):
        x = self.emb(x).sum(dim=1) + self.emb_bias
        x = self.bn1(F.relu(x))
        x = self.dropout1(x)
        
        x = self.hidden(x)
        x = self.bn2(F.relu(x))
        x = self.dropout2(x)
        
        x = self.classifier(x)
        return x


def precision(act, preds):
    return len(act.intersection(preds)) / preds.shape[0]

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-path', type=str, default='../../data/edgelist-train.tsv')
    parser.add_argument('--test-path', type=str, default='../../data/edgelist-test.tsv')
    
    parser.add_argument('--use-cache', action="store_true")
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-verbose', action="store_true")
    parser.add_argument('--seed', type=int, default=456)
    
    return parser.parse_args()


def run_one(batch_size, emb_dim, dropout, bias_offset, lr, lr_type):
    epochs = 6
    lr = 10 ** lr
    
    dataloaders = {
        "train" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=int(batch_size),
            collate_fn=pad_collate_fn,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        ),
        "valid" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=int(batch_size),
            collate_fn=pad_collate_fn,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
    }
    
    destiny_params = {
        "emb_dim"     : int(emb_dim),
        "dropout"     : dropout,
        "bias_offset" : bias_offset,
    }
    
    model = DestinyModel(n_toks=n_toks, **destiny_params).to(torch.device('cuda'))
    # print(model, file=sys.stderr)
    
    if lr_type == 0:
        lr_scheduler = HPSchedule.constant(hp_max=lr)
    elif lr_type == 1:
        lr_scheduler = HPSchedule.linear(hp_max=lr, epochs=epochs)
    elif lr_type == 2:
        lr_scheduler = HPSchedule.linear_cycle(hp_max=lr, epochs=epochs, low_hp=0.0, extra=0)
    
    model.init_optimizer(
        opt=torch.optim.Adam,
        params=model.parameters(),
        hp_scheduler={
            "lr" : lr_scheduler,
        },
    )
    
    t = time()
    for epoch in range(epochs):
        train_hist = model.train_epoch(dataloaders, mode='train', compute_acc=False)
    
    preds, _ = model.predict(dataloaders, mode='valid')
    
    for i in range(preds.shape[0]):
        preds[i][X_train[i]] = -1
    
    top_k = to_numpy(preds.topk(k=10, dim=-1)[1])
    
    p_at_10 = np.mean([precision(X_test[i], top_k[i][:10]) for i in range(len(X_test))])
    print(json.dumps({
        "lr"             : lr,
        "lr_type"        : lr_type,
        "destiny_params" : destiny_params,
        "p_at_10"        : p_at_10,
    }))
    
    return p_at_10

def dlib_find_max_global(f, bounds, int_vars=[], **kwargs):
    varnames = f.__code__.co_varnames[:f.__code__.co_argcount]
    bound1_, bound2_, int_vars_ = [], [], []
    for varname in varnames:
        bound1_.append(bounds[varname][0])
        bound2_.append(bounds[varname][1])
        int_vars_.append(1 if varname in int_vars else 0)
    
    return dlib.find_max_global(f, bound1=bound1_, bound2=bound2_, is_integer_variable=int_vars_, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # IO
    
    if not args.use_cache:
        train_edges = pd.read_csv(args.train_path, header=None, sep='\t')
        test_edges  = pd.read_csv(args.test_path, header=None, sep='\t')
        
        X_train = train_edges.groupby(0)[1].apply(lambda x: list(x + 1)).values # Increment by 1 for padding_idx
        X_test  = test_edges.groupby(0)[1].apply(lambda x: list(x + 1)).values  # Increment by 1 for padding_idx
        
        # Reorder for efficiency
        o = np.argsort([len(t) for t in X_test])[::-1]
        X_train, X_test = X_train[o], X_test[o]
        
        np.save('.X_train_cache.npy', X_train)
        np.save('.X_test_cache.npy', X_test)
    else:
        print('loading cache', file=sys.stderr)
        X_train = np.load('.X_train_cache.npy')
        X_test = np.load('.X_test_cache.npy')
        print('len(X_train)=%s' % str(len(X_train)), file=sys.stderr)
        print('len(X_test)=%s' % str(len(X_test)), file=sys.stderr)
    
    n_toks = max([max(x) for x in X_train]) + 1
    X_test = [set(x) for x in X_test]
    
    dlib_find_max_global(run_one, {
        "emb_dim"     : [50, 800],
        "dropout"     : [0.1, 0.9],
        "bias_offset" : [-40, -1],
        "batch_size"  : [16, 256],
        "lr"          : [-4, -2],
        "lr_type"     : [0, 2],
    }, int_vars=["emb_dim", "batch_size", "lr_type"], num_function_calls=256, solver_epsilon=0.002)