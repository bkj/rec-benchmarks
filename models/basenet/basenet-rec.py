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
from joblib import Parallel, delayed

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet, HPSchedule
from basenet.helpers import to_numpy, set_seeds
from basenet.text.data import SortishSampler

from torch.utils.data import Dataset, DataLoader

# --
# Helpers

def __filter_and_rank(pred, X_filter, k=10):
    for i in range(pred.shape[0]):
        pred[i][X_filter[i]] = -1
    
    return np.argsort(-pred, axis=-1)[:,:k]

def fast_topk(preds, X_train, n_jobs=32):
    offsets = np.cumsum([p.shape[0] for p in preds])
    offsets -= preds[0].shape[0]
    
    jobs = [delayed(__filter_and_rank)(
        to_numpy(pred),
        to_numpy(X_train[offset:(offset + pred.shape[0])])
    ) for pred, offset in zip(preds, offsets)]
    top_k = Parallel(n_jobs=n_jobs, backend='threading')(jobs)
    top_k = np.vstack(top_k)
    
    return top_k

# def slow_topk(preds, X_train):
#     preds, _ = model.predict(dataloaders, mode='valid')
#     for i in range(preds.shape[0]):
#         preds[i][X_train[i]] = -1
    
#     top_k = to_numpy(preds.topk(k=10, dim=-1)[1])
    
#     return top_k

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



class EmbeddingSum(nn.Module):
    def __init__(self, n_toks, emb_dim):
        super().__init__()
        
        self.emb      = nn.Embedding(n_toks, emb_dim, padding_idx=0)
        self.emb_bias = nn.Parameter(torch.zeros(emb_dim))
    
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01)
        self.emb.weight.data[0] = 0
    
    def forward(self, x):
        return self.emb(x).sum(dim=1) + self.emb_bias


class DestinyLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias_offset):
        super().__init__()
        
        self.linear = nn.Linear(in_channels, out_channels)
        
        torch.nn.init.normal_(self.linear.weight.data, 0, 0.01)
        self.linear.bias.data.zero_()
        self.linear.bias.data += bias_offset
    
    def forward(self, x):
        return self.linear(x)


class WTALayer(nn.Module):
    def __init__(self, dim, p):
        super().__init__()
        
        self.dim = dim
        self.p   = p
        self.k   = int(np.ceil(dim * p))
    
    def forward(self, x):
        if self.p < 1:
            # nnz_before = float((x > 0).float().mean())
            topk = x.topk(k=self.k, dim=-1)[0]
            topk = topk[:,-1:]
            x = x * (x >= topk).float()
            # nnz_after = float((x > 0).float().mean())
        
        return x
    
    def __repr__(self):
        return 'WTALayer(dim=%d | p=%f | k=%d)' % (self.dim, self.p, self.k)


class DestinyModel(BaseNet):
    def __init__(self, n_toks, emb_dim, dropout, bias_offset):
        def _loss_fn(x, y):
            return F.binary_cross_entropy_with_logits(x, y)
        
        super().__init__(loss_fn=_loss_fn)
        
        self.layers = nn.Sequential(
            EmbeddingSum(n_toks, emb_dim),
            
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
            
            DestinyLinear(emb_dim, emb_dim, bias_offset=0),
            
            WTALayer(emb_dim, p=0.1),
            
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
            
            DestinyLinear(emb_dim, n_toks, bias_offset=bias_offset),
        )
    
    def forward(self, x):
        return self.layers(x)


def precision(act, preds):
    return len(act.intersection(preds)) / preds.shape[0]

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-path', type=str, default='../../data/edgelist-train.tsv')
    parser.add_argument('--test-path', type=str, default='../../data/edgelist-test.tsv')
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bias-offset', type=float, default=-10)
    
    parser.add_argument('--emb-dim', type=int, default=800)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    parser.add_argument('--use-cache', action="store_true")
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-verbose', action="store_true")
    parser.add_argument('--seed', type=int, default=456)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # IO
    
    if not args.use_cache:
        train_edges = pd.read_csv(args.train_path, header=None, sep='\t')[[0,1]]
        test_edges  = pd.read_csv(args.test_path, header=None, sep='\t')[[0,1]]
        
        X_train = train_edges.groupby(0)[1].apply(lambda x: list(x + 1)).values # Increment by 1 for padding_idx
        X_test  = test_edges.groupby(0)[1].apply(lambda x: list(x + 1)).values  # Increment by 1 for padding_idx
        
        # Reorder for efficiency
        o = np.argsort([len(t) for t in X_test])[::-1]
        X_train, X_test = X_train[o], X_test[o]
        
        print('saving cache', file=sys.stderr)
        np.save('.X_train_cache.npy', X_train)
        np.save('.X_test_cache.npy', X_test)
    else:
        print('loading cache', file=sys.stderr)
        X_train = np.load('.X_train_cache.npy')
        X_test = np.load('.X_test_cache.npy')
        print('X_train.shape=%s' % str(X_train.shape))
        print('X_test.shape=%s' % str(X_test.shape))
    
    n_toks = max([max(x) for x in X_train]) + 1
    X_test = [set(x) for x in X_test]
    
    # --
    # Dataloaders
    
    dataloaders = {
        "train" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=pad_collate_fn,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        ),
        "valid" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=pad_collate_fn,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )
    }
    
    model = DestinyModel(
        n_toks=n_toks,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        bias_offset=args.bias_offset
    ).to(torch.device('cuda'))
    
    model.verbose = not args.no_verbose
    print(model, file=sys.stderr)
    
    model.init_optimizer(
        opt=torch.optim.Adam,
        params=model.parameters(),
        # lr=args.lr,
    )
    
    t = time()
    for epoch in range(args.epochs):
        train_hist = model.train_epoch(dataloaders, mode='train')
        
        if epoch % args.eval_interval == 0:
            
            preds, _ = model.predict(dataloaders, mode='valid', no_cat=True) # no_cat=False if using `slow_topk`
            top_k = fast_topk(preds, X_train)
            
            p_at_01 = np.mean([precision(X_test[i], top_k[i][:1]) for i in range(len(X_test))])
            p_at_05 = np.mean([precision(X_test[i], top_k[i][:5]) for i in range(len(X_test))])
            p_at_10 = np.mean([precision(X_test[i], top_k[i][:10]) for i in range(len(X_test))])
            
            print(json.dumps({
                "epoch"   : epoch,
                "p_at_01" : p_at_01,
                "p_at_05" : p_at_05,
                "p_at_10" : p_at_10,
                "elapsed" : time() - t,
            }))
