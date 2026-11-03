"""Readers for Noah parameter tables (VEGPARM/SOILPARM/GENPARM) following noah39_setvegparms / setsoils."""
import numpy as np

f32 = np.float32


def _section_start(lines, name):
    # first token of a line equal to name (LIS compares the first 4 characters)
    for i, ln in enumerate(lines):
        if ln.strip()[:4] == name[:4] and not ln.strip()[:1].isdigit():
            return i
    raise KeyError(name)


def read_veg(path, lutype4):
    lines = open(path).read().splitlines()
    i = _section_start(lines, lutype4)
    n = int(lines[i + 1].split(",")[0].split()[0])
    rows = []
    for ln in lines[i + 2:i + 2 + n]:
        p = ln.split("'")[0].replace(",", " ").split()
        rows.append([float(x) for x in p[1:18]])
    a = np.array(rows, f32)
    names = ["shdfac", "nroot", "rs", "rgl", "hs", "snup", "maxalb", "laimin", "laimax", "emissmin", "emissmax",
             "albmin", "albmax", "z0min", "z0max", "ztopv", "zbotv"]
    out = {k: a[:, j] for j, k in enumerate(names) if j < a.shape[1]}
    # scalar tail after the table: TOPT, CMCMAX, CFACTR, RSMAX, BARE, NATURAL (label line then value line)
    tail = lines[i + 2 + n:]
    vals = {}
    k = 0
    while k + 1 < len(tail) and len(vals) < 6:
        lab = tail[k].strip().upper()
        if lab.endswith("_DATA") or lab in ("BARE", "NATURAL"):
            vals[lab.replace("_DATA", "").lower()] = float(tail[k + 1].split()[0])
            k += 2
        else:
            k += 1
    out.update(vals)
    return out


def read_soil(path, sltype4="STAS", fxsb=False):
    lines = open(path).read().splitlines()
    i = _section_start(lines, sltype4)
    n = int(lines[i + 1].split(",")[0].split()[0])
    rows = []
    for ln in lines[i + 2:i + 2 + n]:
        p = ln.split("'")[0].replace(",", " ").split()
        rows.append([float(x) for x in p[1:11]])
    a = np.array(rows, f32)
    names = ["bb", "drysmc", "f11", "maxsmc", "refsmc", "satpsi", "satdk", "satdw", "wltsmc", "qtz"]
    out = {k: a[:, j] for j, k in enumerate(names)}
    if fxsb:   # Noah 3.3 soil table has two more columns (FXEXP, SBETA)
        rows2 = [[float(x) for x in ln.split("'")[0].replace(",", " ").split()[11:13]] for ln in lines[i + 2:i + 2 + n]]
        b = np.array(rows2, f32)
        out["fxexp"], out["sbeta"] = b[:, 0], b[:, 1]
    return out


def read_gen(path):
    lines = [ln.strip() for ln in open(path).read().splitlines()]
    out = {}
    i = 0
    ns = int(lines[2].split()[0])
    out["slope_data"] = np.array([float(lines[3 + k].split()[0]) for k in range(ns)], f32)
    i = 3 + ns
    while i + 1 < len(lines):
        lab = lines[i].upper()
        if lab.endswith("_DATA"):
            out[lab[:-5].lower()] = float(lines[i + 1].split()[0])
            i += 2
        else:
            i += 1
    return out
