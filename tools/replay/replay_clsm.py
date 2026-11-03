"""Open-loop replay of the CLSM F2.5 core (clsmf25) on the sampled calls recorded from the real LIS."""
import sys, numpy as np
import jax, jax.numpy as jnp
import dumpio_clsm as D
from lisf_noahmp_jax.physics import clsm_jax as C

def build_x(d):
    x = {k: jnp.asarray(v) for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim == 1}
    x["ityp"] = jnp.asarray(d["ityp"].astype(np.int32))
    x["ght"] = [jnp.asarray(d["ghtcnt"][i]) for i in range(6)]
    x["wesn"] = [jnp.asarray(d["wesnn"][i]) for i in range(3)]
    x["htsn"] = [jnp.asarray(d["htsnnn"][i]) for i in range(3)]
    x["sndz"] = [jnp.asarray(d["sndzn"][i]) for i in range(3)]
    return x

def compare(out, post):
    pairs = {"tc1": "tc1_0", "tc2": "tc2_0", "tc4": "tc4_0", "qa1": "qa1_0", "qa2": "qa2_0", "qa4": "qa4_0", "capac": "capac",
             "catdef": "catdef", "rzexc": "rzexc", "srfexc": "srfexc", "evap": "evap", "shflux": "shflux", "runoff": "runoff",
             "eint": "eint", "esoi": "esoi", "eveg": "eveg", "esno": "esno", "bflow": "bflow", "runsrf": "runsrf",
             "smelt": "smelt", "hlwup": "hlwup", "swland": "swland", "hlatn": "hlatn", "qinfil": "qinfil", "ar1": "ar1",
             "ar2": "ar2", "rzeq": "rzeq", "ghflux": "ghflux", "tpsn1": "tpsn1", "asnow0": "asnow0", "sfmc": "sfmc",
             "rzmc": "rzmc", "prmc": "prmc", "tsurf": "tsurf"}
    res = {}
    for a, b in pairs.items():
        res[a] = (np.asarray(out[a]), post[b])
    for i in range(6):
        res[f"ght{i}"] = (np.asarray(out["ght"][i]), post["ghtcnt"][i]); res[f"tp{i+1}"] = (np.asarray(out["tp"][i]), post[f"tp{i+1}"])
    for i in range(3):
        res[f"wesn{i}"] = (np.asarray(out["wesn"][i]), post["wesnn"][i]); res[f"htsn{i}"] = (np.asarray(out["htsn"][i]), post["htsnnn"][i])
        res[f"sndz{i}"] = (np.asarray(out["sndz"][i]), post["sndzn"][i])
    return res

if __name__ == "__main__":
    dirname = sys.argv[1]
    steps = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 10, 50, 96]
    idx, recs, lay, dn = D.load(dirname)
    fn = jax.jit(lambda x: C.clsmf25(jnp.float32(recs[0][1]["dtstep"]), jnp.float32(recs[0][1]["sfrac"]), x))
    acc = {}
    for s in steps:
        x = build_x(recs[0][s])
        out = fn(x)
        for k, (a, b) in compare(out, recs[1][s]).items():
            e = np.abs(a.astype(np.float64) - b.astype(np.float64))
            bad = ~np.isfinite(e)
            acc.setdefault(k, []).append((np.nanmax(e) if (~bad).any() else np.nan, np.percentile(e[~bad], 99), bad.sum(), np.abs(b).max()))
    print(f"{'var':10s} {'maxabs(ref)':>11s} {'p99':>10s} {'max':>10s} {'nonfinite':>9s}")
    for k, l in acc.items():
        print(f"{k:10s} {max(x[3] for x in l):11.4g} {max(x[1] for x in l):10.3g} {np.nanmax([x[0] for x in l]):10.3g} {sum(x[2] for x in l):9d}")
