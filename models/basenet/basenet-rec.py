#!/usr/bin/env python

"""
    basenet-rec.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from collections import OrderedDict
from joblib import Parallel, delayed

import faiss
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet, HPSchedule
from basenet.helpers import to_numpy, to_device, set_seeds

from torch.utils.data import Dataset, DataLoader

# --
# Helpers

def precision(act, preds):
    return len(act.intersection(preds)) / preds.shape[0]


def predict_precisions(model, X, dataloaders, mode='val', do_topk=True):
    _ = model.eval()
    
    loader = dataloaders[mode]
    gen = enumerate(loader)
    if model.verbose:
        gen = tqdm(gen, total=len(loader), desc='predict:%s' % mode)
    
    offset = 0
    p_at_01, p_at_05, p_at_10 = [], [], []
    for _, (data, target) in gen:
        with torch.no_grad():
            pred = model(to_device(data, model.device) % 10000)
            
            # dummy
            # model(to_device(data, model.device))
            # torch.cuda.synchronize()
            
            if do_topk:
                topk = to_numpy(pred.topk(dim=-1, k=10)[1])
            else:
                topk = to_numpy(pred)
            
            X_tmp = to_numpy(X[offset:(offset + pred.shape[0])])
            
            p_at_01 += [precision(set(X_tmp[i]), topk[i][:1]) for i in range(len(X_tmp))]
            p_at_05 += [precision(set(X_tmp[i]), topk[i][:5]) for i in range(len(X_tmp))]
            p_at_10 += [precision(set(X_tmp[i]), topk[i][:10]) for i in range(len(X_tmp))]
            
            offset += pred.shape[0]
    
    return np.mean(p_at_01), np.mean(p_at_05), np.mean(p_at_10)


def __filter_and_rank(pred, X_filter, k=10):
    # for i in range(pred.shape[0]):
    #     pred[i][X_filter[i]] = -1
    
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
    def __init__(self, n_toks, emb_dim, bag=False):
        super().__init__()
        
        self._bag = bag # !! Faster at inference time, waay slower at training
        
        self.emb     = nn.Embedding(n_toks, emb_dim, padding_idx=0)
        self.emb_bag = nn.EmbeddingBag(n_toks, emb_dim, mode='sum') 
        
        self.emb_bias = nn.Parameter(torch.zeros(emb_dim))
        
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01) # !! Slows down approx. _a lot_ (at high dimensions?)
        self.emb.weight.data[0] = 0
    
    def set_bag(self, val):
        self._bag = val
        if val:
            self.emb_bag.weight.data.set_(self.emb.weight.data.clone())
    
    def forward(self, x):
        
        if not self._bag:
            out = self.emb(x).sum(dim=1) + self.emb_bias
        else:
            out = self.emb_bag(x) + self.emb_bias
        
        return out


class DestinyLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias_offset):
        super().__init__(in_channels, out_channels, bias=False) # !! bias not handled by approx yet
        torch.nn.init.normal_(self.weight.data, 0, 0.01)
        # self.bias.data.zero_()        # !!
        # self.bias.data += bias_offset # !!


class ApproxLinear(nn.Module):
    def __init__(self, linear, batch_size, topk, nprobe, npartitions):
        super().__init__()
        
        self.weights = linear.weight.detach().cpu().numpy()
        
        self.cpu_index = faiss.index_factory(
            self.weights.shape[1],
            f"IVF{npartitions},Flat",
            # "Flat", # Exact query, for testing
            
            faiss.METRIC_INNER_PRODUCT # This appears to be slower -- why? And can we get away w/ L2 at inference time?
            # faiss.METRIC_L2
        )
        self.cpu_index.train(self.weights)
        self.cpu_index.add(self.weights)
        self.cpu_index.nprobe = nprobe
        
        self.res   = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        
        self.topk       = topk
        self.batch_size = batch_size
        
        self.I    = torch.LongTensor(batch_size, topk).cuda()
        self.D    = torch.FloatTensor(batch_size, topk).cuda()
        self.Dptr = faiss.cast_integer_to_float_ptr(self.D.storage().data_ptr())
        self.Iptr = faiss.cast_integer_to_long_ptr(self.I.storage().data_ptr())
        
    def forward(self, x):
        self.I.zero_()
        self.D.zero_()
        torch.cuda.synchronize()
        xptr = faiss.cast_integer_to_float_ptr(x.storage().data_ptr())
        self.index.search_c(
            x.shape[0],
            xptr,
            self.topk,
            self.Dptr,
            self.Iptr,
        )
        torch.cuda.synchronize()
        self.res.syncDefaultStreamCurrentDevice()
        
        if x.shape[0] == self.batch_size:
            return self.I
        else:
            return self.I[:self.batch_size]


class DestinyModel(BaseNet):
    def __init__(self, n_toks, emb_dim, out_dim, topk, dropout, bias_offset):
        super().__init__(loss_fn=F.binary_cross_entropy_with_logits)
        
        self.emb = EmbeddingSum(n_toks, emb_dim)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
            
            DestinyLinear(emb_dim, emb_dim, bias_offset=0),
            
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
        )
        
        # <<
        # Made this bigger, because `n_toks` on this dataset isn't big enough to see gains
        self.linear = DestinyLinear(emb_dim, n_toks, bias_offset=bias_offset)
        # --
        # self.linear = DestinyLinear(emb_dim, out_dim, bias_offset=bias_offset)
        # >>
        
        self.approx_linear = None # Init later
        self.topk          = topk
        self.exact         = None
    
    def set_approx_linear(self, batch_size, nprobe, npartitions):
        self.approx_linear = ApproxLinear(self.linear, batch_size, self.topk, nprobe, npartitions)
    
    def forward(self, x):
        if self.training:
            x = self.emb(x)
            x = self.layers(x)
            x = self.linear(x)
            return x
        else:
            assert self.exact is not None
        
            x = self.emb(x)
            x = self.layers(x)
            x = self.linear(x) if self.exact else self.approx_linear(x)
        
        return x

# --
# Run

out_dim          = 400000
topk             = 32
nprobe           = 32
npartitions      = 8192

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-path', type=str)
    parser.add_argument('--test-path', type=str)
    parser.add_argument('--cache-path', type=str)
    
    parser.add_argument('--batch-size', type=int, default=256) # !!
    parser.add_argument('--emb-dim', type=int, default=800)    # !!
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bias-offset', type=float, default=-10)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-verbose', action="store_true")
    parser.add_argument('--seed', type=int, default=456)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # IO
    
    if args.train_path is not None:
        assert args.test_path is not None
        
        print('load', file=sys.stderr)
        args.train_path = '../../data/netflix/edgelist-train.tsv'
        args.test_path = '../../data/netflix/edgelist-test.tsv'
        
        train_edges = pd.read_csv(args.train_path, header=None, sep='\t')[[0,1]]
        test_edges  = pd.read_csv(args.test_path, header=None, sep='\t')[[0,1]]
        
        print('group', file=sys.stderr)
        X_train = train_edges.groupby(0)[1].apply(lambda x: list(x + 1)).values # Increment by 1 for padding_idx
        X_test  = test_edges.groupby(0)[1].apply(lambda x: list(x + 1)).values  # Increment by 1 for padding_idx
        
        print('sort', file=sys.stderr)
        o = np.argsort([len(t) for t in X_test])[::-1]
        X_train, X_test = X_train[o], X_test[o]
        
        print('saving cache', file=sys.stderr)
        np.save('%s_train.npy' % args.cache_path, X_train)
        np.save('%s_test.npy' % args.cache_path, X_test)
    else:
        print('loading cache: start', file=sys.stderr)
        X_train = np.load('%s_train.npy' % args.cache_path)
        X_test  = np.load('%s_test.npy' % args.cache_path)
        print('loading cache: done', file=sys.stderr)
    
    n_toks = max([max(x) for x in X_train]) + 1
    X_test = [set(x) for x in X_test]
    
    # --
    # Dataloaders
    
    print('define dataloaders', file=sys.stderr)
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
            batch_size=8 * args.batch_size,
            collate_fn=pad_collate_fn,
            num_workers=4,
            pin_memory=False,
            shuffle=False,
        )
    }
    
    print('define model', file=sys.stderr)
    model = DestinyModel(
        n_toks=10000,
        emb_dim=args.emb_dim,
        out_dim=out_dim,
        topk=topk,
        dropout=args.dropout,
        bias_offset=args.bias_offset,
    ).to(torch.device('cuda'))
    model.verbose = not args.no_verbose
    print(model, file=sys.stderr)
    
    model.init_optimizer(
        opt=torch.optim.Adam,
        params=model.parameters(),
        # lr=args.lr,
    )
    
    print('preloading dataloaders["valid"] + warming up', file=sys.stderr)
    dataloaders['valid'] = list(dataloaders['valid'])
    warmup_batch = dataloaders['valid'][0][0]
    
    t = time()
    for epoch in range(args.epochs):
        _ = model.train()
        model.emb.set_bag(False)
        
        # train_hist = model.train_epoch(dataloaders, mode='train', num_batches=100)
        
        if epoch % args.eval_interval == 0:
            _ = model.eval()
            model.emb.set_bag(True)
            
            # --
            # Exact
            
            model.exact = True
            _ = model(warmup_batch.cuda()).cpu()
            
            t = time()
            p_at_01, p_at_05, p_at_10 = predict_precisions(model, X_train, dataloaders, mode='valid', do_topk=True)
            print(json.dumps({
                "epoch"   : epoch,
                "p_at_01" : p_at_01,
                "p_at_05" : p_at_05,
                "p_at_10" : p_at_10,
                "elapsed" : time() - t,
            }))
            
            # --
            # Approximate
            
            print('set_approx_linear + warm', file=sys.stderr)
            model.set_approx_linear(batch_size=8 * args.batch_size, nprobe=nprobe, npartitions=npartitions)
            model.exact = False
            _ = model(warmup_batch.cuda()).cpu()
            
            t = time()
            p_at_01, p_at_05, p_at_10 = predict_precisions(model, X_train, dataloaders, mode='valid', do_topk=False)
            print(json.dumps({
                "epoch"   : epoch,
                "p_at_01" : p_at_01,
                "p_at_05" : p_at_05,
                "p_at_10" : p_at_10,
                "elapsed" : time() - t,
            }))
            
            os._exit(0)

