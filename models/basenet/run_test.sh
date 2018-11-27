#!/bin/bash

# run_test.sh

python synth.py --emb-dim 64
# {'exact_time': 1.376793384552002, 'approx_time': 0.5546786785125732, 'approx_speedup': 2.4821458582904468}
python synth.py --emb-dim 128
# {'exact_time': 2.356105089187622, 'approx_time': 0.7755272388458252, 'approx_speedup': 3.038068775887336}
python synth.py --emb-dim 256
# {'exact_time': 4.437787771224976, 'approx_time': 1.0572614669799805, 'approx_speedup': 4.197436405113028}
python synth.py --emb-dim 512
# {'exact_time': 8.645541191101074, 'approx_time': 1.2734665870666504, 'approx_speedup': 6.7889815711737915}
python synth.py --emb-dim 800
# {'exact_time': 13.332379341125488, 'approx_time': 2.2359402179718018, 'approx_speedup': 5.962761988877839}
python synth.py --emb-dim 1024
# {'exact_time': 17.026272773742676, 'approx_time': 2.2736849784851074, 'approx_speedup': 7.488404477689255}

