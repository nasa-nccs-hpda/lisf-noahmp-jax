"""Builds the small regression-test data sets (tests/data/<case>/) from the full-domain call dumps of the real LIS (GNU 14, serial):
a seeded random sample of the recorded SFLX calls (inputs and the outputs of the Fortran), stored in the same binary layout as the dump.
usage: python tests/make_test_data.py        (needs the dumps under $LISF_RUN, see DATA_AND_PATHS.md; they are not in the repository)"""
import os, sys, shutil
import numpy as np

RUN = os.environ.get("LISF_RUN", "/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N = int(os.environ.get("N_TEST_RECORDS", "600"))


def noah_family(case, src):
    """jaxdump_noah39.bin layout: >i4 (NHDR, NPRE, NPOST), then records."""
    with open(src, "rb") as f:
        nh, npre, npost = (int(x) for x in np.fromfile(f, ">i4", 3))
    dt = np.dtype([("hdr", ">i4", nh), ("pre", ">f4", npre), ("post", ">f4", npost)])
    mm = np.memmap(src, dtype=dt, mode="r", offset=12)
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(len(mm), N, replace=False))
    d = os.path.join(OUT, case); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "jaxdump_noah39.bin"), "wb") as f:
        np.array([nh, npre, npost], ">i4").tofile(f)
        np.array(mm[idx]).tofile(f)
    print(case, len(mm), "->", len(idx))


def mp_family(case, srcdir, prefix="jaxdump_mp", n=None, ice=False, wet=False, crop=False):
    n = n or N
    names = {}
    for k in ("pre", "post"):
        names[k] = [l.split()[0] for l in open(f"{srcdir}/{prefix}_{k}.layout")]
    sizes = {k: sum(int(l.split()[1]) for l in open(f"{srcdir}/{prefix}_{k}.layout")) for k in ("pre", "post")}
    dt = np.dtype([("w1", ">i4"), ("n1", ">i4"), ("pre", ">f4", (sizes["pre"],)), ("w2", ">i4"), ("n2", ">i4"), ("post", ">f4", (sizes["post"],))])
    size = os.path.getsize(f"{srcdir}/{prefix}.bin")
    nrec = size // dt.itemsize
    mm = np.memmap(f"{srcdir}/{prefix}.bin", dtype=dt, mode="r", shape=(nrec,))
    rng = np.random.default_rng(7)
    # all records with snow on the ground first, then random ones
    swe_i = [i for i, nme in enumerate(names["pre"]) if nme == "SWE"]
    off = 0; swe_off = None
    for l in open(f"{srcdir}/{prefix}_pre.layout"):
        a, b = l.split()
        if a == "SWE": swe_off = off
        off += int(b)
    pick = set(rng.choice(nrec, n, replace=False).tolist())
    if swe_off is not None:
        sn = np.nonzero(np.asarray(mm["pre"][:, swe_off]) > 0)[0]
        if len(sn) > 0:
            pick |= set(rng.choice(sn, min(len(sn), n // 3), replace=False).tolist())
    if wet:                                     # records with rain (RAINBL > 0): the surface runoff schemes only act there
        offs = {}; off = 0
        for l in open(f"{srcdir}/{prefix}_pre.layout"):
            a_, b_ = l.split(); offs[a_] = off; off += int(b_)
        rain = np.nonzero(np.asarray(mm["pre"][:, offs["RAINBL"]]) > 0)[0]
        pick |= set(rng.choice(rain, min(len(rain), 2 * n), replace=False).tolist())
    if ice:                                     # Noah-MP 5.0 land-ice records only (VEGTYP == ISICE)
        offs = {}; off = 0
        for l in open(f"{srcdir}/{prefix}_pre.layout"):
            a, b = l.split(); offs[a] = off; off += int(b)
        isice = np.nonzero(np.asarray(mm["pre"][:, offs["VEGTYP"]] if "VEGTYP" in offs else mm["pre"][:, offs["ivgtyp"]]) == np.asarray(mm["pre"][:, offs["p_ISICE"]]))[0]
        pick = set(rng.choice(isice, min(len(isice), n), replace=False).tolist())
    if crop:                                    # Noah-MP 5.0 crop-model records (cropland tiles): n random ones plus every change of the growth stage
        offs = {}; off = 0
        for l in open(f"{srcdir}/{prefix}_pre.layout"):
            a, b = l.split(); offs[a] = off; off += int(b)
        poffs = {}; off = 0
        for l in open(f"{srcdir}/{prefix}_post.layout"):
            a, b = l.split(); poffs[a] = off; off += int(b)
        isc = np.nonzero(np.asarray(mm["pre"][:, offs["ivgtyp"]]) == np.asarray(mm["pre"][:, offs["ISCROP_TABLE"]]))[0]
        chg = isc[np.asarray(mm["pre"][isc, offs["PGSXY"]]) != np.asarray(mm["post"][isc, poffs["PGSXY"]])]
        pick = set(rng.choice(isc, min(len(isc), n), replace=False).tolist()) | set(chg.tolist())
    idx = np.array(sorted(pick))
    d = os.path.join(OUT, case); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{prefix}.bin"), "wb") as f:
        np.array(mm[idx]).tofile(f)
    for k in ("pre", "post"):
        shutil.copy(f"{srcdir}/{prefix}_{k}.layout", os.path.join(d, f"{prefix}_{k}.layout"))
    print(case, nrec, "->", len(idx))


def snow_family(case, srcdir, prefix="jaxdump_mp", n_each=150, n_rand=100):
    """Noah-MP 5.0 records on a deep, wet, layered snowpack (CONUS 1 Mar 2017 00-04 UTC, dump of a Fortran run restarted from the 1 Mar state): fixed numbers of records per
    snow-layer count (ISNOW -1, -2, -3), with liquid water in the pack, with a thick pack, plus snow-free random records. Closes the gap found by the independent review: the 3 h
    CONUS test set has no layered snow, so snow-hydrology and snow-conductivity edits passed every test."""
    names, sizes, offs = {}, {}, {}
    for k in ("pre", "post"):
        names[k] = [l.split() for l in open(f"{srcdir}/{prefix}_{k}.layout")]
        sizes[k] = sum(int(b) for _, b in names[k])
    off = 0
    for a, b in names["pre"]:
        offs[a] = (off, int(b)); off += int(b)
    dt = np.dtype([("w1", ">i4"), ("n1", ">i4"), ("pre", ">f4", (sizes["pre"],)), ("w2", ">i4"), ("n2", ">i4"), ("post", ">f4", (sizes["post"],))])
    nrec = os.path.getsize(f"{srcdir}/{prefix}.bin") // dt.itemsize
    mm = np.memmap(f"{srcdir}/{prefix}.bin", dtype=dt, mode="r", shape=(nrec,))
    col = lambda nme, j=0: np.asarray(mm["pre"][:, offs[nme][0] + j])
    isn = col("ISNOWXY")
    liq = sum(col("SNLIQXY", j) for j in range(offs["SNLIQXY"][1]))
    ice = sum(col("SNICEXY", j) for j in range(offs["SNICEXY"][1]))
    veg, isice = col("VEGTYP") if "VEGTYP" in offs else col("ivgtyp"), col("p_ISICE")
    land = veg != isice
    rng = np.random.default_rng(7)
    pick = set()
    for mask, m in (((isn == -1), n_each), ((isn == -2), n_each), ((isn == -3), n_each), ((liq > 1.0), n_each), ((ice > 100.0), n_each), ((isn == 0), n_rand)):
        idx = np.nonzero(mask & land)[0]
        pick |= set(rng.choice(idx, min(len(idx), m), replace=False).tolist())
    idx = np.array(sorted(pick))
    d = os.path.join(OUT, case); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{prefix}.bin"), "wb") as f:
        np.array(mm[idx]).tofile(f)
    for k in ("pre", "post"):
        shutil.copy(f"{srcdir}/{prefix}_{k}.layout", os.path.join(d, f"{prefix}_{k}.layout"))
    print(case, nrec, "->", len(idx))


def clsm_family(case, srcdir, steps=(2, 12), ntile=1500):
    sys.path.insert(0, os.path.join(os.path.dirname(OUT), "..", "tools", "replay"))
    import dumpio_clsm as D
    idx, recs, lay, dn = D.load(srcdir, maxstep=max(steps))
    ns = len(idx)
    rng = np.random.default_rng(7)
    sel = np.sort(rng.choice(ns, ntile, replace=False))
    d = os.path.join(OUT, case); os.makedirs(d, exist_ok=True)
    arrs = {}
    for s in steps:
        for tag, nm in ((0, "in"), (1, "out")):
            for k, v in recs[tag][s].items():
                v = np.asarray(v)
                arrs[f"{nm}{s}_{k}"] = v if v.ndim == 0 else (v[sel] if v.ndim == 1 else v[..., sel])
    np.savez_compressed(os.path.join(d, "clsm_calls.npz"), steps=np.array(steps), **arrs)
    print(case, ns, "->", len(sel), "tiles, steps", steps)


if __name__ == "__main__":
    noah_family("noah33", f"{RUN}/conus_n33_dump/jaxdump_noah39.bin")
    noah_family("noah271", f"{RUN}/conus_n271_dump/jaxdump_noah39.bin")
    mp_family("noahmp401", RUN)
    mp_family("noahmp36", f"{RUN}/conus_mp36_dump")
    # one small data set per Noah-MP option variant (recorded with the option changed in lis.config, 3 h CONUS dumps)
    V401 = dict(run2="conus_mp401_dump_r2", run3="conus_mp401_dump_r3", run4="conus_mp401_dump_r4", frz2="conus_mp401_dump_frz2",
                inf2="conus_mp401_dump_inf2", crs2="conus_mp401_dump_crs2", sfc2="conus_mp401_dump_sfc2", stc2="conus_mp401_dump_stc2",
                stc3="conus_mp401_dump_stc3", dveg3="conus_mp401_dump_dveg3", dveg1="conus_mp401_dump_dveg1")
    V36 = dict(run2="conus_mp36_dump_r2", run4="conus_mp36_dump_r4", frz2="conus_mp36_dump_frz2", inf2="conus_mp36_dump_inf2",
               crs2="conus_mp36_dump_crs2", sfc2="conus_mp36_dump_sfc2", stc2="conus_mp36_dump_stc2", dveg3="conus_mp36_dump_dveg3", dveg1="conus_mp36_dump_dveg1")
    for tag, d in V401.items():
        mp_family(f"noahmp401_{tag}", f"{RUN}/{d}", n=200)
    for tag, d in V36.items():
        mp_family(f"noahmp36_{tag}", f"{RUN}/{d}", n=200)
    # Noah-MP 5.0 (CONUS 3 h dump, snow tiles first + random; land-ice records separately); dump layout of lis_patches/make_noahmp50_hook.py
    mp_family("noahmp50", f"{RUN}/mp50_dump_conus", n=600)
    mp_family("noahmp50_sgp", f"{RUN}/mp50_dump_sgp", n=400)
    mp_family("noahmp50_ice", f"{RUN}/mp50_dump_conus", n=300, ice=True)
    snow_family("noahmp50_snow", f"{RUN}/mp50_rst_mar1")        # deep, wet, layered snowpack (1 Mar 2017 restart)
    # Noah-MP 5.0 surface runoff options 6 (VIC), 7 (Xinanjiang), 8 (dynamic VIC), SGP Feb 14 2017 (10 mm of rain), dumps of lis/variants50.py with VAR_DAY=14
    for r in (6, 7, 8):
        mp_family(f"noahmp50_run{r}", f"{RUN}/mp50v_sgp_runoff{r}w", n=80, wet=True)
    # Noah-MP 5.0 dynamic vegetation options 2, 5, 6 (carbon model, LAI predicted) and 7, 8, 9 (LAI from an LDT file, synthetic monthly map), SGP Jul 9 2017,
    # dumps of lis/variants50.py with VAR_DAY=9 VAR_MONTH=7 (7-9 with "@LAI data source=LDT" and a copy of the LDT file that holds a LAI variable)
    for d in (2, 5, 6, 7, 8, 9):
        mp_family(f"noahmp50_dveg{d}", f"{RUN}/mp50v_sgp_dveg{d}j", n=150)
    # Noah-MP 5.0 tile drainage 1 (simple) and 2 (Hooghoudt), SGP Feb 14 2017, synthetic TD_FRACTION map (0.05-0.95) in a copy of the LDT file
    for t in (1, 2):
        mp_family(f"noahmp50_tdrn{t}", f"{RUN}/mp50v_sgp_tdrn{t}w", n=150)
    # Noah-MP 5.0 irrigation (SGP Jul 9 2017, synthetic IRFRACT/SIFRACT/MIFRACT/FIFRACT maps in a copy of the LDT file; dumps of a LIS build that sets LLANDUSE, because the
    # stock 7.8 driver never does and so never irrigates): maps (trigger 1, method 0), flood (1, 3), LAI trigger + sprinkler (3, 1)
    for tag in ("m0", "m3", "l1"):
        mp_family(f"noahmp50_irr{tag}", f"{RUN}/mp50v_sgp_irr{tag}", n=150)
    # Noah-MP 5.0 dynamic crop model (crop option 1, DEFAULT_CROP = 1 corn in a copy of the table; synthetic PLANTING / HARVEST / SEASON_GDD maps in a copy of the LDT file),
    # SGP Jul 9 2017, all eight growth stages; the table copy keeps only the crop group (lis/dump50.py reads NoahmpTable.TBL next to the dump)
    mp_family("noahmp50_crop1", f"{RUN}/mp50v_sgp_dveg1crop1j", n=200, crop=True)
    import re
    tbl = open(f"{RUN}/NoahmpTable_crop1.TBL").read()
    grp = re.search(r"&noahmp_crop_parameters.*?\n/\n", tbl, re.S).group(0)
    open(os.path.join(OUT, "noahmp50_crop1", "NoahmpTable.TBL"), "w").write(grp)
    # irrigation trigger 2 (crop season from the PLANTING / HARVEST maps), flood method, patched LIS build as above
    mp_family("noahmp50_irrt2", f"{RUN}/mp50v_sgp_irrt2m3", n=150)
    clsm_family("clsm", f"{RUN}/conus_clsm_jul_d")
