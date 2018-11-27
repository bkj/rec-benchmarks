import numpy as np
import faiss
import torch
from time import time
import h5py

# --

nq           = 2048
d            = 128
nb           = 400000
k            = 32
nprobe       = 32
npartitions  = 8192

sift         = True

if sift:
    d = 128

# --
# IO

if sift:
    f  = h5py.File('sift-128-euclidean.hdf5')
    xq = f['test'].value[:nq]
    xb = f['train'].value[:nb]
    
    neibs = f['neighbors'].value[:xq.shape[0]]
    neibs[neibs >= xb.shape[0]] = -1
    for i in range(neibs.shape[1] - 1)[::-1]:
        sel = neibs[:,i] == -1
        neibs[sel,i] = neibs[sel,i+1]

else:
    xb = np.random.uniform(0, 1, (nb, d)).astype(np.float32)
    xq = np.random.uniform(0, 1, (nq, d)).astype(np.float32)

# --
# Faiss

cpu_index = faiss.index_factory(d, f"IVF{npartitions},Flat")
cpu_index.train(xb)
cpu_index.add(xb)
cpu_index.nprobe = nprobe

res   = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

xq_torch = torch.FloatTensor(xq).cuda()
xb_torch = torch.FloatTensor(xb).cuda()

I = torch.LongTensor(nq, k).cuda()
D = torch.FloatTensor(nq, k).cuda()

Iptr = I.storage().data_ptr()
Dptr = D.storage().data_ptr()
Dptr = faiss.cast_integer_to_float_ptr(Dptr)
Iptr = faiss.cast_integer_to_long_ptr(Iptr)

# warmup
torch.cuda.synchronize()
xptr = xq_torch.storage().data_ptr()
index.search_c(
    nq,
    faiss.cast_integer_to_float_ptr(xptr),
    k,
    Dptr,
    Iptr,
)
torch.cuda.synchronize()
res.syncDefaultStreamCurrentDevice()

t = time()
for _ in range(10):
    torch.cuda.synchronize()
    xptr = xq_torch.storage().data_ptr()
    index.search_c(
        nq,
        faiss.cast_integer_to_float_ptr(xptr),
        k,
        Dptr,
        Iptr,
    )
    
    torch.cuda.synchronize()
    res.syncDefaultStreamCurrentDevice()
    
    a = I[:,0].cpu()

faiss_time = time() - t

if sift:
    faiss_acc  = (a.cpu().numpy() == neibs[:,0]).mean()
    print('faiss_acc ', faiss_acc)
    print('----------')

print('faiss_time', faiss_time)

# --
# Torch (1 chunk)

try:
    t = time()
    for _ in range(10):
        z = torch.mm(xq_torch, xb_torch.t())
        # z = z.topk(k=k, dim=-1)[1]
        a = z[:,0].cpu()
        # a = z.max(dim=-1)[1].cpu()

    torch_time = time() - t
except:
    print('torch_error!')
    torch_time = -1

print('torch_time', torch_time)
