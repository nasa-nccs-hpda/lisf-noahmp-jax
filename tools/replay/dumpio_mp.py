"""Loader for the Noah-MP 4.0.1 dump (jaxdump_mp.bin + layout files written by lis_patches/jaxdump_mp_mod.F90)."""
import numpy as np


def read_layout(path):
    names, sizes = [], []
    for line in open(path):
        a, b = line.split()
        names.append(a); sizes.append(int(b))
    return names, sizes


def load(dirpath, max_records=None, start=0, idx=None, prefix="jaxdump_mp"):
    pn, ps = read_layout(f"{dirpath}/{prefix}_pre.layout")
    qn, qs = read_layout(f"{dirpath}/{prefix}_post.layout")
    npre, npost = sum(ps), sum(qs)
    dt = np.dtype([("w1", ">i4"), ("n1", ">i4"), ("pre", ">f4", (npre,)), ("w2", ">i4"), ("n2", ">i4"),
                   ("post", ">f4", (npost,))])
    import os
    nrec = os.path.getsize(f"{dirpath}/{prefix}.bin") // dt.itemsize
    mm = np.memmap(f"{dirpath}/{prefix}.bin", dtype=dt, mode="r", shape=(nrec,))
    n = len(mm)
    stop = n if max_records is None else min(n, start + max_records)
    sub = mm[start:stop] if idx is None else mm[np.asarray(idx)]
    assert (sub["w1"] == 1).all() and (sub["w2"] == 2).all()

    def split(arr, names, sizes):
        out = {}; o = 0
        for nm, sz in zip(names, sizes):
            out[nm] = np.array(arr[:, o] if sz == 1 else arr[:, o:o + sz]).astype(np.float32); o += sz
        return out
    return split(sub["pre"], pn, ps), split(sub["post"], qn, qs)
