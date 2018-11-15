from rsub import *
from matplotlib import pyplot as plt

import json
import numpy as np
import pandas as pd

x = list(map(json.loads, open('tmp.jl').read().splitlines()[:-1]))

p = [xx['p_at_10'] for xx in x]
print(max(p))
_ = plt.plot(p)
_ = plt.plot(pd.Series(p).cummax())
show_plot()