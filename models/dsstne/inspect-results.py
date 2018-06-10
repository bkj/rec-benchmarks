#!/usr/bin/env python

"""
    inspect-results.py
"""

from __future__ import division, print_function

import sys
import json
import numpy as np
import pandas as pd

if __name__ == "__main__":
    test_path = sys.argv[1]
    recs_path = sys.argv[2]
    
    print('loading text: %s' % test_path, file=sys.stderr)
    
    test = pd.read_csv(test_path, sep='\t', header=None)
    test = test.set_index(test[0], drop=True)
    test = test[1].apply(lambda x: set([int(xx) for xx in x.split(':')])).to_dict()
    
    print('loading recs: %s' % recs_path, file=sys.stderr)
    recs = pd.read_csv(recs_path, sep='\t', header=None)
    recs = recs.set_index(recs[0], drop=True)
    recs = recs[1].apply(lambda x: [int(xx.split(',')[0]) for xx in x.split(':')[:-1]]).to_dict()
    
    assert len(recs) == len(test)
    
    res = {}
    for top_k in [1, 5, 10]:
        precisions = [len(test[k].intersection(v[:top_k])) / top_k for k,v in recs.items()]
        print('p@%0.2d -> %f' % (top_k, np.mean(precisions)))