"""Soil-moisture ensemble Kalman filter as configured in LIS (Noah-MP.4.0.1 + 'Synthetic SM' observations, EnKF).
Mirrors lis/dataassim/algorithm/enkf/{enkf_Mod,enkf_general}.F90, lis/core/LIS_lsmMod.F90 (applyLSMPert),
LIS_DAobservationsMod.F90 (observation perturbation) and lis/surfacemodels/land/noahmp.4.0.1/da_soilm/*.F90.
All arrays are float32 and the member/layer loops keep the Fortran summation order. Tile axis layout: cell*nens + member."""
import os
import numpy as np
import netCDF4 as nc
from . import pert as pertmod

f32 = np.float32
UDEF = f32(-9999.0)
TKFRZ = f32(273.16)
MIN_THRESHOLD = f32(0.02)


def _seq_sum(a, axis):
    """sequential float32 sum along an axis (Fortran intrinsic sum order)."""
    a = np.moveaxis(a, axis, 0)
    s = a[0].astype(f32).copy()
    for k in range(1, a.shape[0]):
        s = (s + a[k]).astype(f32)
    return s


def read_state_attribs(path):
    """noah(mp)_sm_attribs.txt: (min, max) for each state variable."""
    L = [l.rstrip("\n") for l in open(path) if not l.startswith("#")]
    n = int(L[0]); out = []
    for i in range(n):
        v = L[2 + 2 * i].split()
        out.append((f32(v[0]), f32(v[1])))
    return out


def setsoilm(soilm, smc, sh2o, tslb, smcmax, nens, bias_corr=True):
    """NoahMP401_setsoilm. soilm/smc/sh2o/tslb: (ncell, nens, 4); smcmax: (ncell, nens) table value.
    Returns new (smc, sh2o)."""
    smc = smc.copy(); sh2o = sh2o.copy()
    smth = (smcmax - f32(0.02)).astype(f32)[:, :, None]                     # sm_threshold
    delta = (soilm - smc).astype(f32)
    t = (sh2o + delta).astype(f32)
    ok = (t > MIN_THRESHOLD) & (t < smth) & (tslb > TKFRZ)
    flag_t = ok.all(axis=2)                                                 # (ncell, nens)
    upd = flag_t.all(axis=1)
    pc = flag_t.sum(axis=1)
    upd_ens = ~(pc < nens * 0.5)
    upd_new = upd | upd_ens
    mx = np.where(flag_t[:, :, None], soilm, f32(-10000.0)).max(axis=1)     # (ncell, 4)
    mn = np.where(flag_t[:, :, None], soilm, f32(10000.0)).min(axis=1)
    mid = ((mx - mn) / f32(2) + mn).astype(f32)
    sel = upd_new[:, None, None]
    good = sel & flag_t[:, :, None]
    bad = sel & (~flag_t[:, :, None])
    sh2o = np.where(good, (sh2o + (soilm - smc)).astype(f32), sh2o)
    smc = np.where(good, soilm, smc)
    sh2o = np.where(bad, mid[:, None, :], sh2o)
    smc = np.where(bad, mid[:, None, :], smc)
    if bias_corr:
        for i in np.nonzero(~upd_new)[0]:
            _bias_correct_cell(smc[i], sh2o[i], tslb[i], smcmax[i], nens)
    return smc.astype(f32), sh2o.astype(f32)


def _bias_correct_cell(smc, sh2o, tslb, smcmax, nens):
    """The 'do while (bounds_violation)' loop of setsoilm for one cell; smc/sh2o (nens, 4) modified in place."""
    u = nens - 1
    ens_flag = np.ones((4, nens), bool)
    smcmax = smcmax.astype(f32)
    nit = 0
    viol = True

    def mean_dep():
        d = np.zeros(4, f32)
        for j in range(4):
            for m in range(nens - 1):
                if tslb[m, 0] > TKFRZ:
                    d[j] = f32(d[j] + f32(sh2o[m, j] - sh2o[u, j]))
                else:
                    d[j] = f32(d[j] + f32(smc[m, j] - smc[u, j]))
        return d
    while viol:
        nit += 1
        ens_count = np.zeros(4, f32)
        delta = mean_dep()
        for j in range(4):
            delta[j] = f32(delta[j] / f32(nens - 1))
            for m in range(nens - 1):
                smth = f32(smcmax[m] - f32(0.02))
                tmpval = f32(sh2o[m, j] - delta[j])
                if tmpval <= MIN_THRESHOLD:
                    sh2o[m, j] = max(sh2o[u, j], MIN_THRESHOLD)
                    smc[m, j] = max(smc[u, j], MIN_THRESHOLD)
                    ens_flag[j, m] = False; ens_count[j] += 1
                elif tmpval >= smth:
                    sh2o[m, j] = min(sh2o[u, j], smth)
                    smc[m, j] = min(smc[u, j], smth)
                    ens_flag[j, m] = False; ens_count[j] += 1
        delta = mean_dep()
        for j in range(4):
            if ens_count[j] < (nens - 1):
                delta[j] = f32(delta[j] / f32(nens - 1 - ens_count[j]))
            else:
                delta[j] = f32(delta[j] / f32(nens - 1))
            for m in range(nens - 1):
                if ens_flag[j, m]:
                    tmpval = f32(sh2o[m, j] - delta[j])
                    if not (tmpval <= 0.0 or tmpval > smcmax[m]):
                        smc[m, j] = f32(smc[m, j] - delta[j])
                        sh2o[m, j] = f32(sh2o[m, j] - delta[j])
                        viol = False
                tmpval = sh2o[m, j]
                viol = bool(tmpval <= 0.0 or tmpval > smcmax[m])
        if nit > 10 and viol:
            for j in range(4):
                for m in range(nens):
                    if sh2o[m, j] > smcmax[m] or smc[m, j] > smcmax[m]:
                        sh2o[m, j] = smcmax[m]; smc[m, j] = smcmax[m]
                    if sh2o[m, j] < MIN_THRESHOLD or smc[m, j] < MIN_THRESHOLD:
                        sh2o[m, j] = MIN_THRESHOLD; smc[m, j] = MIN_THRESHOLD
            if nit > 100:
                break


def apply_state_perturbation(smc, pert, nens):
    """applyLSMPert (additive, perturbation bias correction on). smc/pert: (ncell, nens, 4)."""
    s = smc.copy()
    for m in range(nens - 1):
        s[:, m, :] = (s[:, m, :] + pert[:, m, :]).astype(f32)
    delta = np.zeros((smc.shape[0], 4), f32)
    for m in range(nens - 1):
        delta = (delta + (s[:, m, :] - s[:, nens - 1, :])).astype(f32)
    delta = (delta / f32(nens - 1)).astype(f32)
    for m in range(nens - 1):
        s[:, m, :] = (s[:, m, :] - delta).astype(f32)
    return s


def enkf_increments(state, obs_pred, obs, obs_err, obs_std, assim):
    """enkf_analysis with one observation per cell. state: (ncell, nens, 4); obs_pred/obs_err: (ncell, nens);
    obs/obs_std: (ncell,). Returns increments (ncell, nens, 4), zero where assim is False."""
    ncell, nens, nst = state.shape
    sbar = (_seq_sum(state, 1) / f32(nens)).astype(f32)
    sp = (state - sbar[:, None, :]).astype(f32)
    obar = (_seq_sum(obs_pred, 1) / f32(nens)).astype(f32)
    op = (obs_pred - obar[:, None]).astype(f32)
    repr_ = (_seq_sum((op * op).astype(f32), 1) / f32(nens - 1)).astype(f32)
    repr_ = (repr_ + (obs_std * obs_std).astype(f32)).astype(f32)
    incr = np.zeros_like(state)
    for e in range(nens):
        rhs = (obs + obs_err[:, e] - obs_pred[:, e]).astype(f32)
        rhs = (rhs / repr_).astype(f32)                                        # ludcmp/lubksb of a 1x1 matrix
        w = (op * rhs[:, None]).astype(f32)
        tmp = np.zeros((ncell, nst), f32)
        for k in range(nens):
            tmp = (tmp + (sp[:, k, :] * w[:, k][:, None]).astype(f32)).astype(f32)
        incr[:, e, :] = (tmp / f32(nens - 1)).astype(f32)
    incr[~assim] = f32(0.0)
    return incr


class SoilMoistureDA:
    """EnKF assimilation of 'Synthetic SM' observations (history-file SoilMoist_tavg layer 1) into Noah-MP soil moisture."""

    def __init__(self, S, cfg, P_, nens, dtstep, parse_interval):
        self.S, self.nens = S, nens
        self.nx, self.ny = S.nc, S.nr
        self.cols, self.rows = S.cols, S.rows
        self.ncell = S.ntile
        assert cfg.s("Data assimilation algorithm") == "EnKF"
        assert cfg.s("Data assimilation set") == "Synthetic SM"
        assert cfg.s("Data assimilation scaling strategy").lower() == "none"
        self.incroption = cfg.i("Data assimilation exclude analysis increments")
        self.obsdir = P_(cfg.s("Synthetic soil moisture data directory"))
        names, spec = pertmod.read_pert_attrs(P_(cfg.s("State perturbation attributes file")), 4)
        self.state_interval = parse_interval(cfg.s("State perturbation frequency"))
        self.state_P = pertmod.ForcingPerturb(self.nx, self.ny, nens, spec, dtstep, names=names)
        assert all(s["typ"] == 0 for s in spec)
        self.state_bounds = read_state_attribs(P_(cfg.s("State attributes file")))
        onames, ospec = pertmod.read_pert_attrs(P_(cfg.s("Observation perturbation attributes file")), 1)
        self.obs_P = pertmod.ForcingPerturb(self.nx, self.ny, nens, ospec, dtstep, names=onames)
        assert ospec[0]["typ"] == 0
        self.obs_std = f32(ospec[0]["std"])
        tab = S.soil["SMCMAX"][S.soiltyp - 1].astype(f32)
        self.smcmax = np.repeat(tab, nens).reshape(self.ncell, nens)
        self.smcwlt = S.soil["SMCWLT"][S.soiltyp - 1].astype(f32)
        self.log = []

    def _r(self, a):
        a = np.asarray(a, f32)
        return a.reshape(self.ncell, self.nens, *a.shape[1:])

    def _set(self, st, smc, sh2o):
        st = dict(st)
        st["SMC"] = smc.reshape(-1, 4).astype(f32)
        st["SMH2O"] = sh2o.reshape(-1, 4).astype(f32)
        return st

    def _qc_state(self, soilm):
        lo, hi = self.state_bounds[0]
        soilm = soilm.copy()
        s1 = soilm[:, :, 0]
        s1 = np.where(s1 > hi, hi, s1); s1 = np.where(s1 < lo, lo, s1)
        soilm[:, :, 0] = s1
        return soilm

    def _soil(self, st):
        return self._r(st["SMC"]), self._r(st["SMH2O"]), self._r(np.asarray(st["STC"])[:, 3:])

    def perturb_states(self, now, st):
        """LIS_lsm_perturb_states: get_pert, apply, qc, setsoilm."""
        if (now.hour * 3600 + now.minute * 60 + now.second) % self.state_interval != 0:
            return st
        pr = self.state_P.get_pert()                                          # (4, nx, ny, nens)
        p = np.stack([pr[j][self.cols, self.rows, :] for j in range(4)], -1)  # (ncell, nens, 4)
        smc, sh2o, tslb = self._soil(st)
        soilm = self._qc_state(apply_state_perturbation(smc, p, self.nens))
        return self._set(st, *setsoilm(soilm, smc, sh2o, tslb, self.smcmax, self.nens))

    def read_obs(self, now):
        p = f"{self.obsdir}/{now:%Y%m}/SimObs_{now:%Y%m%d%H%M}.nc"
        if not os.path.exists(p):
            return None
        d = nc.Dataset(p)
        v = np.ma.filled(d["SoilMoist_tavg"][0], UDEF).astype(f32)          # layer 1, (nr, nc)
        d.close()
        return v[self.rows, self.cols]

    def qc_obs(self, obs, st, last, tg, swe):
        """NoahMP401_qc_soilmobs; each model quantity is the mean over the tiles (members) of a cell."""
        smc, sh2o, stc = self._soil(st)
        em = lambda a: (_seq_sum(self._r(a), 1) / f32(self.nens)).astype(f32)
        rainf = (em(last["TotalPrecip"]) / f32(self.S.dt)).astype(f32)
        t1 = em(tg); sneqv = em(swe); sca = (em(last["Snowcover"]) / f32(100.0)).astype(f32); shdfac = em(last["Greenness"])
        o = obs.copy()
        valid = o != UDEF
        bad = rainf > f32(3e-6)
        for j in range(4):
            bad |= np.abs(em(smc[:, :, j].reshape(-1)) - em(sh2o[:, :, j].reshape(-1))) > f32(0.0001)
        for j in range(4):
            bad |= em(stc[:, :, j].reshape(-1)) <= TKFRZ
        bad |= t1 <= TKFRZ
        bad |= self.S.vegt.astype(f32) <= 4
        bad |= sneqv > f32(0.001)
        bad |= sca > f32(0.0001)
        bad |= shdfac > f32(0.9)
        bad |= (self.smcmax[:, 0] - o) < f32(0.02)
        bad |= (o - self.smcwlt) < f32(0.02)
        o[valid & bad] = UDEF
        return o

    def analysis(self, now, st, last, tg, swe):
        """readDAobservations, perturb_DAobservations, EnKF increments and enkf_update. Returns the new state."""
        obs = self.read_obs(now)
        if obs is None:
            return st
        obs = self.qc_obs(obs, st, last, tg, swe)
        valid = obs != UDEF
        if not valid.any():
            return st
        pr = self.obs_P.get_pert()[0][self.cols, self.rows, :].astype(f32)     # (ncell, nens)
        pr[:, self.nens - 1] = f32(0.0)
        delta = (_seq_sum(pr[:, : self.nens - 1], 1) / f32(self.nens - 1)).astype(f32)
        for m in range(self.nens - 1):
            pr[:, m] = (pr[:, m] - delta).astype(f32)
        smc, sh2o, tslb = self._soil(st)
        incr = enkf_increments(smc, smc[:, :, 0].copy(), np.where(valid, obs, f32(0)), pr,
                               np.full(self.ncell, self.obs_std, f32), valid)
        self.log.append((now, int(valid.sum()), float(np.abs(incr).max())))
        if self.incroption != 0:
            return st
        soilm = self._qc_state((smc + incr).astype(f32))
        return self._set(st, *setsoilm(soilm, smc, sh2o, tslb, self.smcmax, self.nens))
