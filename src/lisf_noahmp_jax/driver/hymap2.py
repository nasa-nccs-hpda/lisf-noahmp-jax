"""HyMAP2 river routing (lis/routing/HYMAP2_router) in JAX.

Supported options: routing method kinematic / diffusive / local inertia (flowtype 1, 2, 3), time step constant or adaptive,
floodplain dynamics on/off, linear runoff/baseflow reservoir on/off. Not ported: evaporation, deep water infiltration,
reservoir operation, direct insertion, ocean tides, urban drainage, hybrid method, 2-way coupling, ensembles, DA.
All cells of the river network advance together within a (sub)step, using the values at the start of the substep for the
downstream neighbours, exactly as the Fortran code does (it gathers the global arrays before the cell loop); the inflow to a
cell is the sum of the outflows of its upstream cells (scatter-add).
"""
import numpy as np
import netCDF4 as nc
import jax
import jax.numpy as jnp

f32 = np.float32
IMIS = -9999
MIS = f32(IMIS)


class HyMAP2Params:
    """River network and parameters from the LDT output file, ordered as LIS orders its routing vector (column-major)."""

    def __init__(self, ldt_file, cfg, nz=10):
        d = nc.Dataset(ldt_file)
        g = lambda k: np.array(d[k][:]).astype(f32)
        nextx = np.rint(g("HYMAP_flow_direction_x")).astype(np.int64)
        nexty = np.rint(g("HYMAP_flow_direction_y")).astype(np.int64)
        mask = g("HYMAP_basin_mask")
        nr, ncol = nextx.shape
        self.nr, self.nc = nr, ncol
        # sequence: ix (column) outer, iy (row) inner, cells with a flow direction inside the basin mask
        valid = (nextx != IMIS) & (mask > 0)
        cc, rr = np.nonzero(valid.T)                         # column-major order
        self.cols, self.rows = cc, rr
        n = len(cc)
        self.nseq = n
        sindex = np.full((nr, ncol), IMIS, np.int64)
        sindex[rr, cc] = np.arange(n)
        self.sindex = sindex
        nx_ = nextx[rr, cc]
        self.outlet = (nx_ <= 0).astype(np.int32)           # 0: flows to a neighbour, 1: outlet
        nxt = np.zeros(n, np.int64)
        m0 = self.outlet == 0
        nxt[m0] = sindex[nexty[rr, cc][m0] - 1, nx_[m0] - 1]
        assert (nxt[m0] >= 0).all(), "downstream cell outside the routing sequence"
        self.next = nxt
        v = lambda k: g(k)[rr, cc].astype(f32)
        self.rivwth, self.rivlen, self.rivhgt, self.rivman = v("HYMAP_river_width"), v("HYMAP_river_length"), \
            v("HYMAP_river_height"), v("HYMAP_river_roughness")
        self.fldman = v("HYMAP_floodplain_roughness")
        fh = g("HYMAP_floodplain_height")
        self.fldhgt = np.stack([fh[k][rr, cc] for k in range(nz)], 1).astype(f32)         # (n, nz)
        el = g("HYMAP_grid_elevation"); el = np.where(el < 0, f32(0), el)
        self.elevtn = el[rr, cc].astype(f32)
        self.nxtdst, self.grarea = v("HYMAP_grid_distance"), v("HYMAP_grid_area")
        cn = v("HYMAP_runoff_time_delay")
        mn = cn[cn > 0].min() if (cn > 0).any() else f32(1)
        self.cntime = np.where(cn == 0, mn, cn).astype(f32)
        self.trnoff = v("HYMAP_runoff_time_delay_multiplier")
        tb = v("HYMAP_baseflow_time_delay")
        self.tbsflw = np.where((tb < self.cntime / f32(86400.0)) & (self.cntime > 0), self.cntime / f32(86400.0), tb).astype(f32)
        for k in ("rivwth", "rivlen", "rivhgt", "rivman", "fldman", "elevtn", "nxtdst", "grarea", "cntime", "trnoff", "tbsflw"):
            assert (getattr(self, k) != MIS).all(), f"missing values in {k}"
        assert (self.fldhgt != MIS).all()
        self.nz = nz
        # derived (HYMAP2_routingInit / HYMAP2_set_fldstg)
        self.rivstomax = ((self.rivlen * self.rivwth).astype(f32) * self.rivhgt).astype(f32)
        self.rivelv = (self.elevtn - self.rivhgt).astype(f32)
        self.rivare = (self.rivwth * self.rivlen).astype(f32)
        wthinc = ((self.grarea - self.rivare) / self.rivlen / f32(nz)).astype(f32)
        stopre = self.rivstomax.copy(); hgtpre = np.zeros(n, f32)
        fsm = np.zeros((n, nz), f32)
        for i in range(nz):
            stonow = ((self.rivlen * (self.rivwth + wthinc * (f32(i + 1) - f32(0.5))).astype(f32)).astype(f32)
                      * (self.fldhgt[:, i] - hgtpre).astype(f32)).astype(f32)
            fsm[:, i] = stopre + stonow
            stopre = fsm[:, i].copy(); hgtpre = self.fldhgt[:, i].copy()
        self.fldstomax = fsm

        self.dt = cfg["dt"]
        self.method = cfg["method"]                        # 'kinematic' | 'diffusive' | 'local inertia'
        self.flowmap = {"kinematic": 1, "diffusive": 2, "local inertia": 3}[self.method]
        self.steptype = cfg["steptype"]                    # 'constant' | 'adaptive'
        self.cadp = f32(cfg.get("cadp", 0.7))
        self.floodflag = cfg["floodflag"]
        self.linres = cfg["linres"]
        self.slpmin = f32(1e-5)
        self.grv = f32(9.81)


def hymap2_config(cfg, parse_interval):
    c = cfg
    assert c.s("Routing model") == "HYMAP2 router", c.s("Routing model")
    assert c.i("HYMAP2 run in ensemble mode") == 0
    assert c.s("HYMAP2 routing model evaporation option").lower() in ("none", "0"), "evaporation not ported"
    assert c.i("HYMAP2 routing model dwi flag") == 0, "deep water infiltration not ported"
    assert c.i("HYMAP2 reservoir operation option") == 0, "reservoir operation not ported"
    assert c.i("HYMAP2 enable 2-way coupling") == 0 if "HYMAP2 enable 2-way coupling" in c.d else True
    assert c.s("HYMAP2 routing method") in ("kinematic", "diffusive", "local inertia"), c.s("HYMAP2 routing method")
    return dict(dt=f32(parse_interval(c.s("HYMAP2 routing model time step"))), method=c.s("HYMAP2 routing method"),
                steptype=c.s("HYMAP2 routing model time step method"),
                cadp=c.f("HYMAP2 routing model adaptive time step alfa coefficient")
                if "HYMAP2 routing model adaptive time step alfa coefficient" in c.d else 0.7,
                floodflag=c.i("HYMAP2 floodplain dynamics"), linres=c.i("HYMAP2 routing model linear reservoir flag"),
                out_interval=parse_interval(c.s("HYMAP2 routing model output interval")))


STATE_KEYS = ("rivsto", "fldsto", "roffsto", "basfsto", "rivdph", "rivout_pre", "rivdph_pre", "fldout_pre", "flddph_pre",
              "fldelv1")


def initial_state(P):
    return {k: jnp.zeros(P.nseq, jnp.float32) for k in STATE_KEYS}


def make_router(P):
    """Returns step(state, runoff, baseflow) -> (state, outputs); runoff/baseflow in kg m-2 s-1 on the routing vector
    (LIS undefined = -9999 allowed)."""
    j = lambda a: jnp.asarray(a)
    outlet0 = j(P.outlet == 0)
    nxt = j(P.next)
    rivlen, rivwth, rivman, fldman = j(P.rivlen), j(P.rivwth), j(P.rivman), j(P.fldman)
    elevtn, nxtdst, grarea = j(P.elevtn), j(P.nxtdst), j(P.grarea)
    cntime, trnoff, tbsflw = j(P.cntime), j(P.trnoff), j(P.tbsflw)
    rivstomax, rivelv, rivare = j(P.rivstomax), j(P.rivelv), j(P.rivare)
    fldhgt, fldstomax = j(P.fldhgt), j(P.fldstomax)
    nz = P.nz
    slpmin, grv, cadp = jnp.float32(P.slpmin), jnp.float32(P.grv), jnp.float32(P.cadp)
    dt_in = jnp.float32(P.dt)
    n = P.nseq
    kine = P.flowmap == 1
    f32j = jnp.float32

    def down(a, own):
        return jnp.where(outlet0, a[nxt], own)

    def calc_fldstg(rivsto, fldsto):
        """HYMAP2_calc_fldstg: rstoall, h1, h2, v1, v2 are real*8 in the Fortran, everything else single precision."""
        d = jnp.float64
        rsto32 = rivsto + fldsto                                      # single-precision sum, then widened
        rstoall = rsto32.astype(d)
        cond = (rstoall > rivstomax.astype(d)) & (grarea > rivare)
        fsm64, fh64 = fldstomax.astype(d), fldhgt.astype(d)
        gt = rstoall[:, None] > fsm64
        i0 = jnp.sum(jnp.cumprod(gt.astype(jnp.int32), axis=1), axis=1)       # 0-based bracketing level
        over = i0 >= nz
        ii = jnp.clip(i0, 0, nz - 1)
        take = lambda a, k: jnp.take_along_axis(a, k[:, None], 1)[:, 0]
        h2, v2 = take(fh64, ii), take(fsm64, ii)
        prev = jnp.clip(ii - 1, 0, nz - 1)
        h1 = jnp.where(ii > 0, take(fh64, prev), 0.0)
        v1 = jnp.where(ii > 0, take(fsm64, prev), rivstomax.astype(d))
        flddph_in = (h1 + (h2 - h1) * (rstoall - v1) / (v2 - v1)).astype(f32j)
        a32 = (rivare / grarea + ii.astype(f32j) / f32j(nz)).astype(f32j)
        fldfrc_in = jnp.fmin(jnp.fmax(a32.astype(d) + f32j(1.0 / nz).astype(d) * (rstoall - v1) / (v2 - v1), 0.0),
                                1.0).astype(f32j)
        flddph_over = (fh64[:, nz - 1] + (rstoall - fsm64[:, nz - 1]) / grarea.astype(d)).astype(f32j)
        flddph = jnp.where(over, flddph_over, flddph_in)
        fldfrc = jnp.where(over, f32j(1.0), fldfrc_in)
        rivsto_n = rivstomax + rivlen * rivwth * flddph
        ok_geom = (rivwth > 0) & (rivlen > 0)
        rivdph_n = jnp.where(ok_geom, rivsto_n / rivlen / rivwth, f32j(0))
        fldsto_n = jnp.fmax(rstoall - rivsto_n.astype(d), 0.0).astype(f32j)
        fldare_n = grarea * fldfrc
        fldwth_n = fldare_n / rivlen
        flddph1_n = fldsto_n / fldwth_n / rivlen
        rivdph_e = jnp.fmax(0.0, rstoall / rivare.astype(d)).astype(f32j)
        fldfrc_e = jnp.where(rsto32 > 0, jnp.fmin(jnp.fmax(rivare / grarea, f32j(0)), f32j(1)), f32j(0))
        out = dict(
            rivsto=jnp.where(cond, rivsto_n, rsto32),
            fldsto=jnp.where(cond, fldsto_n, f32j(0)),
            rivdph=jnp.where(cond, rivdph_n, rivdph_e),
            flddph=jnp.where(cond, flddph, f32j(0)),
            flddph1=jnp.where(cond, flddph1_n, f32j(0)),
            fldfrc=jnp.where(cond, fldfrc, fldfrc_e),
        )
        out["fldare"] = jnp.where(cond, fldare_n, out["fldfrc"] * grarea)
        out["fldwth"] = jnp.where(cond, fldwth_n, f32j(0))
        out["fldelv1"] = rivelv + out["rivdph"] - out["flddph1"]
        return out

    def rivout_kine(is_out, dt, elv, elv_down, wth, man, sto, dph):
        slope = jnp.where(is_out, slpmin, jnp.fmax((elv - elv_down) / nxtdst, slpmin))
        vel = (f32j(1.0) / man) * jnp.sqrt(slope) * jnp.power(dph, f32j(2.0 / 3.0))
        area = wth * dph
        out = jnp.fmin(area * vel, sto / dt)
        return out, vel

    def rivout_iner(is_out, dt, elv, elv_down, wth, sto, sto_down, dph, dph_down, man, out_pre, dph_pre, dph_pre_down):
        sfc = elv + dph
        sfc_down = elv_down + dph_down
        sfc_pre = elv + dph_pre
        sfc_pre_down = elv_down + dph_pre_down
        dslope = (sfc - sfc_down) / nxtdst
        dflw = jnp.fmax(sfc, sfc_down) - elv
        darea = wth * dflw
        dflw_pre = jnp.fmax(sfc_pre, sfc_pre_down) - elv
        dflw_imp = jnp.sqrt(dflw * dflw_pre)
        dflw_imp = jnp.where(dflw_imp <= 0, dflw, dflw_imp)
        act = (dflw_imp > 0) & (darea > 0)
        safe = jnp.where(act, dflw_imp, f32j(1))
        dout_pre = out_pre / wth
        o = wth * (dout_pre + grv * dt * safe * dslope) / (f32j(1) + grv * dt * man ** 2 * jnp.abs(dout_pre)
                                                         * jnp.power(safe, f32j(-7.0 / 3.0)))
        o = jnp.where(o >= 0, jnp.fmin(o, sto / dt), -jnp.fmin(jnp.abs(o), sto_down / dt))
        vel = o / jnp.where(act, darea, f32j(1))
        o = jnp.where(act, o, f32j(0)); vel = jnp.where(act, vel, f32j(0))
        return o, vel, sfc

    def core(carry, dt, ro1, bf1):
        s = dict(carry)
        # runoff -> linear reservoirs
        if P.linres == 1:
            roffsto = s["roffsto"] + ro1 * dt
            qrun = roffsto / (trnoff * cntime)
            v = roffsto - qrun * dt
            qrun = jnp.where(v < 0, roffsto / dt, qrun)
            roffsto = jnp.where(v < 0, f32j(0), v)
            basfsto = s["basfsto"] + bf1 * dt
            qbas = basfsto / (tbsflw * f32j(86400.0))
            v = basfsto - qbas * dt
            qbas = jnp.where(v < 0, basfsto / dt, qbas)
            basfsto = jnp.where(v < 0, f32j(0), v)
            runoff = qrun + qbas
            s["roffsto"], s["basfsto"] = roffsto, basfsto
        else:
            runoff = ro1 + bf1
            s["roffsto"] = jnp.zeros(n, f32j); s["basfsto"] = jnp.zeros(n, f32j)
        fs = calc_fldstg(s["rivsto"], s["fldsto"])
        rivsto, fldsto, rivdph, flddph1, fldelv1 = fs["rivsto"], fs["fldsto"], fs["rivdph"], fs["flddph1"], fs["fldelv1"]
        fldwth = fs["fldwth"]
        out = {}
        rivelv_d = jnp.where(outlet0, rivelv[nxt], rivelv - slpmin * nxtdst)
        fldelv1_d = jnp.where(outlet0, fldelv1[nxt], fldelv1 - slpmin * nxtdst)
        if kine:
            rivout, rivvel = rivout_kine(~outlet0, dt, rivelv, rivelv_d, rivwth, rivman, rivsto, rivdph)
            if P.floodflag == 1:
                fldout, fldvel = rivout_kine(~outlet0, dt, fldelv1, fldelv1_d, fldwth, fldman, fldsto, flddph1)
                sfcelv = fldelv1 + flddph1                  # the floodplain call overwrites sfcelv
            else:
                fldout = jnp.zeros(n, f32j); fldvel = jnp.zeros(n, f32j)
                sfcelv = rivelv + rivdph                    # HYMAP2_no_floodplain_flow writes the other variable
            new_pre = {k: s[k] for k in ("rivout_pre", "rivdph_pre", "fldout_pre", "flddph_pre")}
            if P.floodflag != 1:
                new_pre["fldout_pre"] = jnp.zeros(n, f32j); new_pre["flddph_pre"] = flddph1
        else:
            rivsto_d = jnp.where(outlet0, rivsto[nxt], f32j(1e20))
            rivdph_d = jnp.where(outlet0, rivdph[nxt], rivdph - slpmin * nxtdst)
            rivdph_pre_d = jnp.where(outlet0, s["rivdph_pre"][nxt], s["rivdph_pre"] - slpmin * nxtdst)
            rivout, rivvel, sfcelv = rivout_iner(~outlet0, dt, rivelv, rivelv_d, rivwth, rivsto, rivsto_d, rivdph, rivdph_d,
                                                 rivman, s["rivout_pre"], s["rivdph_pre"], rivdph_pre_d)
            new_pre = {"rivout_pre": rivout, "rivdph_pre": rivdph, "fldout_pre": s["fldout_pre"],
                       "flddph_pre": s["flddph_pre"]}
            if P.floodflag == 1:
                fldsto_d = jnp.where(outlet0, fldsto[nxt], f32j(1e20))
                flddph1_d = jnp.where(outlet0, flddph1[nxt], flddph1 - slpmin * nxtdst)
                flddph_pre_d = jnp.where(outlet0, s["flddph_pre"][nxt], s["flddph_pre"] - slpmin * nxtdst)
                fldout, fldvel, _ = rivout_iner(~outlet0, dt, fldelv1, fldelv1_d, fldwth, fldsto, fldsto_d, flddph1, flddph1_d,
                                                fldman, s["fldout_pre"], s["flddph_pre"], flddph_pre_d)
                new_pre["fldout_pre"] = fldout; new_pre["flddph_pre"] = flddph1
            else:
                fldout = jnp.zeros(n, f32j); fldvel = jnp.zeros(n, f32j)
                new_pre["fldout_pre"] = jnp.zeros(n, f32j); new_pre["flddph_pre"] = flddph1
        rivinf = jax.ops.segment_sum(jnp.where(outlet0, rivout, f32j(0)), nxt, num_segments=n)
        fldinf = jax.ops.segment_sum(jnp.where(outlet0, fldout, f32j(0)), nxt, num_segments=n)
        rivsto = jnp.fmax(f32j(0), rivsto + (rivinf - rivout) * dt)
        fldsto = jnp.fmax(f32j(0), fldsto + (fldinf - fldout) * dt)
        fldfrc = fs["fldfrc"]
        neg = runoff < 0
        rfld = jnp.where(neg, jnp.fmax(-fldsto, runoff * fldfrc * dt), runoff * fldfrc * dt)
        rriv = jnp.where(neg, jnp.fmax(-rivsto, runoff * dt - rfld), runoff * (f32j(1) - fldfrc) * dt)
        rivsto = jnp.fmax(f32j(0), rivsto + rriv)
        fldsto = jnp.fmax(f32j(0), fldsto + rfld)
        surfws = f32j(1e3) * (rivsto + fldsto) / grarea
        s.update(rivsto=rivsto, fldsto=fldsto, rivdph=rivdph, fldelv1=fldelv1, **new_pre)
        out.update(rivout=rivout, rivvel=rivvel, fldout=fldout, fldvel=fldvel, flddph=fs["flddph"], fldfrc=fldfrc,
                   fldare=fs["fldare"], sfcelv=sfcelv, surfws=surfws)
        return s, out

    def step(state, runoff0, basflw0):
        ok = (runoff0 != MIS) & (grarea != MIS)
        ro1 = jnp.where(ok, runoff0 * grarea / f32j(1e3), f32j(0))
        bf1 = jnp.where(ok, basflw0 * grarea / f32j(1e3), f32j(0))
        if P.linres != 1:
            ro1 = jnp.where(ok, runoff0, f32j(0)); bf1 = jnp.where(ok, basflw0, f32j(0))
        s = dict(state)
        s["rivdph"] = jnp.where(jnp.max(s["rivdph"]) <= 0, s["rivdph_pre"], s["rivdph"])
        if P.steptype == "constant":
            dta = jnp.full(n, dt_in, f32j); nt = jnp.ones(n, jnp.int32)
        else:
            zdph = jnp.fmax(s["rivdph"], f32j(0.01))
            dt0 = cadp * nxtdst / jnp.sqrt(grv * zdph)
            nt = (dt_in / dt0 - f32j(0.001)).astype(jnp.int32) + 1
            dta = dt_in / nt.astype(f32j)
            big = dta > dt_in
            dta = jnp.where(big, dt_in, dta); nt = jnp.where(big, 1, nt)
        i = jnp.argmin(dta)
        maxi, mindt = nt[i], dta[i]
        zero = jnp.zeros(n, f32j)
        acc0 = dict(rivout=zero, rivvel=zero, fldout=zero, fldvel=zero)
        out0 = dict(rivout=zero, rivvel=zero, fldout=zero, fldvel=zero, flddph=zero, fldfrc=zero, fldare=zero, sfcelv=zero,
                    surfws=zero)

        def body(it, c):
            s_, acc, _ = c
            s_, o = core(s_, mindt, ro1, bf1)
            mi = maxi.astype(f32j)
            acc = dict(rivout=acc["rivout"] + (o["rivout"] + o["fldout"]) / mi, rivvel=acc["rivvel"] + o["rivvel"] / mi,
                       fldout=acc["fldout"] + o["fldout"] / mi, fldvel=acc["fldvel"] + o["fldvel"] / mi)
            return s_, acc, o
        s, acc, o = jax.lax.fori_loop(0, maxi, body, (s, acc0, out0))
        outs = dict(o)
        outs.update(acc)
        outs["rivsto"], outs["fldsto"], outs["rivdph"] = s["rivsto"], s["fldsto"], s["rivdph"]
        outs["roffsto"], outs["basfsto"] = s["roffsto"], s["basfsto"]
        return s, outs

    return jax.jit(step)
