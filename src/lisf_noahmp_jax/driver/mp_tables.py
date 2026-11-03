"""Readers for the Noah-MP 4.0.1 parameter tables (MPTABLE.TBL Fortran namelists, SOILPARM.TBL, GENPARM.TBL),
mirroring read_mp_veg_parameters / read_mp_soil_parameters / read_mp_rad_parameters / read_mp_global_parameters."""
import re
import numpy as np

f32 = np.float32


def _tok(v):
    v = v.strip()
    if not v:
        return None
    if v[0] in "\"'":
        return v.strip("\"'")
    low = v.lower()
    if low in (".true.", ".t.", "t"):
        return True
    if low in (".false.", ".f.", "f"):
        return False
    v = v.replace("d", "e").replace("D", "e")
    try:
        return int(v)
    except ValueError:
        return float(v)


def read_namelists(path):
    """{group: {KEY: [values]}} for a file of Fortran namelist groups."""
    out, cur, name = {}, None, None
    for raw in open(path):
        line = raw.split("!")[0].strip()
        if not line:
            continue
        if line.startswith("&"):
            name = line[1:].split()[0].lower()
            out[name] = {}
            cur = name
            line = line[len(name) + 1:].strip()
            if not line:
                continue
        if line == "/":
            cur = None
            continue
        if cur is None:
            continue
        end = line.endswith("/")
        line = line.rstrip("/").strip()
        out[cur].setdefault("__text__", "")
        out[cur]["__text__"] += " " + line
        if end:
            cur = None
    for g in out.values():
        text = g.pop("__text__", "")
        parts = re.split(r"(\b[A-Za-z_][A-Za-z0-9_]*)\s*=", text)
        # parts: [pre, key, val, key, val, ...]
        for i in range(1, len(parts) - 1, 2):
            key, val = parts[i].upper(), parts[i + 1]
            vals = []
            for t in re.findall(r"\"[^\"]*\"|'[^']*'|[^\s,]+", val.strip()):
                m = re.match(r"^(\d+)\*(.+)$", t)
                if m:
                    vals += [_tok(m.group(2))] * int(m.group(1))
                else:
                    vals.append(_tok(t))
            g[key] = vals
    return out


def read_mp_veg(path, scheme):
    """scheme: USGS | MODIFIED_IGBP_MODIS_NOAH | UMD -> dict of per-class arrays (index 0 = class 1)"""
    g = read_namelists(path)
    key = {"USGS": "usgs", "MODIFIED_IGBP_MODIS_NOAH": "modis", "UMD": "umd"}[scheme]
    cat = g[f"noahmp_{key}_veg_categories"]
    par = g[f"noahmp_{key}_parameters"]
    nveg = cat["NVEG"][0]
    d = {}
    for k in ("ISURBAN ISWATER ISBARREN ISICE ISCROP EBLFOREST NATURAL LOW_DENSITY_RESIDENTIAL "
              "HIGH_DENSITY_RESIDENTIAL HIGH_INTENSITY_INDUSTRIAL").split():
        d[k] = par[k][0]
    arr = lambda k: np.array(par[k][:nveg], f32)
    for k in ("CH2OP DLEAF Z0MVT HVT HVB DEN RC MFSNO XL CWPVT C3PSN KC25 AKC KO25 AKO AVCMX AQE LTOVRC DILEFC DILEFW "
              "RMF25 SLA FRAGR TMIN VCMX25 TDLEF BP MP QE25 RMS25 RMR25 ARM FOLNMX WDPOOL WRRAT MRP NROOT RGL RS HS "
              "TOPT RSMAX").split():
        d[k] = arr(k)
    months = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
    d["SAIM"] = np.stack([arr(f"SAI_{m}") for m in months], 1)
    d["LAIM"] = np.stack([arr(f"LAI_{m}") for m in months], 1)
    for a in ("RHOL", "RHOS", "TAUL", "TAUS"):
        d[a] = np.stack([arr(f"{a}_VIS"), arr(f"{a}_NIR")], 1)
    return d


def read_mp_rad_global(path):
    g = read_namelists(path)
    r = g["noahmp_rad_parameters"]
    d = {"ALBSAT": np.stack([np.array(r["ALBSAT_VIS"], f32), np.array(r["ALBSAT_NIR"], f32)], 1),
         "ALBDRY": np.stack([np.array(r["ALBDRY_VIS"], f32), np.array(r["ALBDRY_NIR"], f32)], 1)}
    for k in ("ALBICE", "ALBLAK", "OMEGAS", "EG"):
        d[k] = np.array(r[k], f32)
    d["BETADS"] = f32(r["BETADS"][0]); d["BETAIS"] = f32(r["BETAIS"][0])
    gl = g["noahmp_global_parameters"]
    for k in ("CO2", "O2", "TIMEAN", "FSATMX", "Z0SNO", "SSI", "SWEMX", "RSURF_SNOW"):
        d[k] = f32(gl[k][0])
    return d


def read_mp_soil(soil_path, gen_path):
    L = open(soil_path).read().splitlines()
    ncat = int(L[2].split(",")[0].split()[0])
    rows = []
    for i in range(ncat):
        p = [x.strip() for x in L[3 + i].split(",")]
        rows.append([float(x) for x in p[1:11]])
    a = np.array(rows, f32)
    s = dict(zip("BEXP SMCDRY F1 SMCMAX SMCREF PSISAT DKSAT DWSAT SMCWLT QUARTZ".split(), a.T))
    G = [l for l in open(gen_path).read().splitlines()]
    nslope = int(G[2].split()[0])
    slope = np.array([float(G[3 + i].split()[0]) for i in range(nslope)], f32)
    j = 3 + nslope
    val = lambda idx: f32(float(G[idx].split()[0]))
    csoil, refdk, refkdt, frzk, zbot, czil = (val(j + k) for k in (5, 9, 11, 13, 15, 17))
    return s, dict(SLOPE=slope, CSOIL=csoil, REFDK=refdk, REFKDT=refkdt, FRZK=frzk, ZBOT=zbot, CZIL=czil)
