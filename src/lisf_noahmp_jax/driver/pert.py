"""GMAO perturbation scheme of LIS (dataassim/perturb/gmaopert) for fields that are white in space (xcorr = ycorr = 0):
L'Ecuyer ran2 + Box-Muller (LIS_ran2_gasdev), AR(1) in time, cross-correlation through the Jacobi square root of the
correlation matrix (nr_jacobi), zero-mean adjustment over the ensemble, truncation, additive/lognormal application.
Everything in float32 and in the same operation order as the Fortran. Spatially correlated (FFT) perturbations are not ported."""
import numpy as np

f32 = np.float32
NRANDSEED = 35
NTAB = NRANDSEED - 3
IM1, IM2 = 2147483563, 2147483399
AM = f32(1.) / f32(IM1)
IMM1 = IM1 - 1
IA1, IA2, IQ1, IQ2, IR1, IR2 = 40014, 40692, 53668, 52774, 12211, 3791
NDIV = 1 + IMM1 // NTAB
EPS = f32(1.2e-7)
RNMX = f32(1.) - EPS


def init_randseed(seed1):
    """seed1: int array (...) of negative seeds -> state (..., 35) int64"""
    seed1 = np.asarray(seed1, np.int64)
    idum = np.maximum(-seed1, 1)
    idum2 = idum.copy()
    iv = np.zeros(seed1.shape + (NTAB,), np.int64)
    for j in range(NTAB + 8, 0, -1):
        k = idum // IQ1
        idum = IA1 * (idum - k * IQ1) - k * IR1
        idum = np.where(idum < 0, idum + IM1, idum)
        if j <= NTAB:
            iv[..., j - 1] = idum
    iy = iv[..., 0]
    st = np.zeros(seed1.shape + (NRANDSEED,), np.int64)
    st[..., 0], st[..., 1], st[..., 2] = idum, idum2, iy
    st[..., 3:] = iv
    return st


def ran2(st, mask=None):
    """one uniform deviate per state (vectorized). Returns (new_state, value float32). Lanes where mask is False keep state."""
    idum, idum2, iy = st[..., 0].copy(), st[..., 1].copy(), st[..., 2].copy()
    iv = st[..., 3:].copy()
    k = idum // IQ1
    idum = IA1 * (idum - k * IQ1) - k * IR1
    idum = np.where(idum < 0, idum + IM1, idum)
    k = idum2 // IQ2
    idum2 = IA2 * (idum2 - k * IQ2) - k * IR2
    idum2 = np.where(idum2 < 0, idum2 + IM2, idum2)
    j = iy // NDIV
    ar = np.arange(iv.shape[0])
    ivj = iv[ar, j]
    iy = ivj - idum2
    iv[ar, j] = idum
    iy = np.where(iy < 1, iy + IMM1, iy)
    val = np.minimum(AM * iy.astype(f32), RNMX)
    new = np.empty_like(st)
    new[..., 0], new[..., 1], new[..., 2] = idum, idum2, iy
    new[..., 3:] = iv
    if mask is not None:
        new = np.where(mask[:, None], new, st)
    return new, val.astype(f32)


def gasdev_pair(st):
    """st: (n,35). Returns new_state, g1, g2 (Box-Muller with rejection, per lane)."""
    n = st.shape[0]
    g1 = np.zeros(n, f32); g2 = np.zeros(n, f32)
    todo = np.ones(n, bool)
    st = st.copy()
    while todo.any():
        s1, v1 = ran2(st, todo)
        s2, v2 = ran2(s1, todo)
        st = np.where(todo[:, None], s2, st)
        v1 = f32(2.) * v1 - f32(1.)
        v2 = f32(2.) * v2 - f32(1.)
        rsq = v1 ** 2 + v2 ** 2
        ok = todo & ~((rsq >= f32(1.)) | (rsq == 0.))
        with np.errstate(all="ignore"):
            fac = np.sqrt(-f32(2.) * np.log(rsq) / rsq)
        g1 = np.where(ok, v1 * fac, g1)
        g2 = np.where(ok, v2 * fac, g2)
        todo = todo & ~ok
    return st, g1.astype(f32), g2.astype(f32)


def white_field(rseed):
    """generate_white_field. rseed: (nx,ny,35) (modified copy returned). Loop order i (outer) j (inner): pairs of cells."""
    nx, ny, _ = rseed.shape
    flat = rseed.reshape(nx * ny, NRANDSEED).copy()
    nc = nx * ny
    even = np.arange(0, nc, 2)
    st, g1, g2 = gasdev_pair(flat[even])
    flat[even] = st
    out = np.zeros(nc, f32)
    out[even] = g1
    odd = np.arange(1, nc, 2)
    out[odd] = g2[: len(odd)]
    return flat.reshape(nx, ny, NRANDSEED), out.reshape(nx, ny)


def jacobi(amat):
    n = amat.shape[0]
    a = amat.astype(f32).copy()
    v = np.zeros((n, n), f32)
    for ip in range(n):
        v[ip, ip] = 1.
    b = np.array([a[i, i] for i in range(n)], f32)
    d = b.copy()
    z = np.zeros(n, f32)
    for i in range(1, 51):
        sm = f32(0.)
        for ip in range(n - 1):
            for iq in range(ip + 1, n):
                sm = f32(sm + abs(a[ip, iq]))
        if sm == 0.:
            return d, v
        tresh = f32(0.2) * sm / f32(n ** 2) if i < 4 else f32(0.)
        for ip in range(n - 1):
            for iq in range(ip + 1, n):
                g = f32(100.) * abs(a[ip, iq])
                if i > 4 and f32(abs(d[ip]) + g) == abs(d[ip]) and f32(abs(d[iq]) + g) == abs(d[iq]):
                    a[ip, iq] = 0.
                elif abs(a[ip, iq]) > tresh:
                    h = f32(d[iq] - d[ip])
                    if f32(abs(h) + g) == abs(h):
                        t = f32(a[ip, iq] / h)
                    else:
                        theta = f32(f32(0.5) * h / a[ip, iq])
                        t = f32(f32(1.) / f32(abs(theta) + np.sqrt(f32(1.) + theta ** 2)))
                        if theta < 0.:
                            t = -t
                    c = f32(f32(1.) / np.sqrt(f32(1. + t ** 2)))
                    s = f32(t * c)
                    tau = f32(s / f32(1. + c))
                    h = f32(t * a[ip, iq])
                    z[ip] = f32(z[ip] - h); z[iq] = f32(z[iq] + h)
                    d[ip] = f32(d[ip] - h); d[iq] = f32(d[iq] + h)
                    a[ip, iq] = 0.
                    for j in range(0, ip):
                        g = a[j, ip]; h = a[j, iq]
                        a[j, ip] = f32(g - s * f32(h + g * tau)); a[j, iq] = f32(h + s * f32(g - h * tau))
                    for j in range(ip + 1, iq):
                        g = a[ip, j]; h = a[j, iq]
                        a[ip, j] = f32(g - s * f32(h + g * tau)); a[j, iq] = f32(h + s * f32(g - h * tau))
                    for j in range(iq + 1, n):
                        g = a[ip, j]; h = a[iq, j]
                        a[ip, j] = f32(g - s * f32(h + g * tau)); a[iq, j] = f32(h + s * f32(g - h * tau))
                    for j in range(n):
                        g = v[j, ip]; h = v[j, iq]
                        v[j, ip] = f32(g - s * f32(h + g * tau)); v[j, iq] = f32(h + s * f32(g - h * tau))
        for ip in range(n):
            b[ip] = f32(b[ip] + z[ip]); d[ip] = b[ip]; z[ip] = 0.
    raise RuntimeError("too many iterations in jacobi")


def sqrt_corr_matrix(a):
    d, s = jacobi(a)
    s = s.copy()
    for j in range(len(d)):
        assert d[j] >= 0.
        s[:, j] = s[:, j] * np.sqrt(d[j])
    return s


class ForcingPerturb:
    """GMAO perturbations for N_pert fields on an nx x ny grid with n_ens members.
    spec: list (file order) of dict(typ, std, std_max, zeromean, tcorr, ccorr(list over file order)).
    order: indices mapping the Pert_State item order (the ESMF item-name order) to the file order."""

    def __init__(self, nx, ny, nens, spec, dtstep, names=None):
        """names: field names in file order; the perturbation items are processed in the ESMF item (alphabetical) order and
        the cross-correlation matrix is permuted consistently (verified against the Fortran for the 557WW forcing set)."""
        self.nx, self.ny, self.nens, self.dt = nx, ny, nens, f32(dtstep)
        n = len(spec)
        self.order = list(range(n)) if names is None else sorted(range(n), key=lambda i: names[i])
        sp = []
        for k in self.order:
            d = dict(spec[k]); d["ccorr"] = [spec[k]["ccorr"][l] for l in self.order]; sp.append(d)
        self.spec = sp
        self.n = n
        tmp1 = np.array([[sp[k]["ccorr"][l] for l in range(n)] for k in range(n)], f32)
        self.S = sqrt_corr_matrix(tmp1)
        gid = (np.arange(ny)[None, :] + 1) * 0 + 0
        col = np.arange(nx)[:, None] + 1
        row = np.arange(ny)[None, :] + 1
        domain_id = col + (row - 1) * nx
        self.rseed = np.zeros((nx, ny, nens, NRANDSEED), np.int64)
        for m in range(nens):
            seed1 = -(domain_id + m * nx * ny)
            self.rseed[:, :, m, :] = init_randseed(seed1)
        self.ntrmdt = np.zeros((n, nx, ny, nens), f32)
        self.pert = np.zeros((n, nx, ny, nens), f32)
        self.get_pert(initialize=True)

    def _propagate(self, initialize):
        n, nx, ny = self.n, self.nx, self.ny
        for m in range(n):
            tcorr = self.spec[m]["tcorr"]
            if (not initialize) and tcorr > 0.:
                cc = f32(np.exp(f32(-self.dt / f32(tcorr)))); dd = f32(np.sqrt(f32(1.) - cc ** 2))
                white_t = False
            else:
                cc, dd, white_t = f32(0.), f32(1.), True
            for e in range(self.nens):
                self.rseed[:, :, e, :], rfield = white_field(self.rseed[:, :, e, :])
                if white_t:
                    self.ntrmdt[m, :, :, e] = rfield
                else:
                    self.ntrmdt[m, :, :, e] = cc * self.ntrmdt[m, :, :, e] + dd * rfield

    def get_pert(self, initialize=False):
        n, nx, ny, ne = self.n, self.nx, self.ny, self.nens
        self._propagate(initialize)
        for m in range(n):
            if self.spec[m]["zeromean"] and ne > 2:
                A = self.ntrmdt[m]                              # (nx, ny, ne)
                s = A[..., 0].copy()
                for e in range(1, ne):
                    s = s + A[..., e]
                corr = -s / f32(ne)
                self.ntrmdt[m] = A + corr[..., None]
        out = np.zeros_like(self.pert)
        for m in range(n):
            sp = self.spec[m]
            std = f32(sp["std"])
            if sp["typ"] == 1:
                tr = np.log(f32(1.) + std ** 2)
                mean = f32(-0.5) * tr
                stdt = np.sqrt(tr)
            else:
                mean, stdt = f32(0.), std
            for e in range(ne):
                tmp = np.zeros((nx, ny), f32)
                for mm in range(n):
                    tmp = tmp + self.S[m, mm] * self.ntrmdt[mm, :, :, e]
                tmp = np.sign(tmp) * np.minimum(np.abs(tmp), f32(sp["std_max"]))
                val = mean + stdt * tmp
                out[m, :, :, e] = val if sp["typ"] == 0 else np.exp(val)
        self.pert = out
        return out


def read_pert_attrs(path, nvars):
    """LIS_readPertAttributes: file order records of (typ, std, std_max, zeromean, tcorr, xcorr, ycorr, ccorr...)."""
    lines = [l.rstrip("\n") for l in open(path) if not l.startswith("#")]
    spec, i = [], 0
    names = []
    for _ in range(nvars):
        names.append(lines[i].strip()); i += 1
        v = lines[i].split(); i += 1
        spec.append(dict(typ=int(v[0]), std=float(v[1]), std_max=float(v[2]), zeromean=int(v[3]) == 1, tcorr=float(v[4]),
                         xcorr=float(v[5]), ycorr=float(v[6]), ccorr=[float(x) for x in v[7:7 + nvars]]))
    return names, spec


def apply_forcing_perturbation(base, pert, typs, mins, maxs, bias_corr=True):
    """LIS_perturb_forcing for one field. base: (ncell,) unperturbed forcing; pert: (ncell, nens) perturbation values.
    Returns (ncell, nens). With bias correction (hard-wired on in LIS_readConfig) the last member stays unperturbed and the
    remaining members are shifted so that their mean departure from the last member is zero."""
    base = np.asarray(base, f32)
    ncell, nens = pert.shape
    f = np.repeat(base[:, None], nens, 1).astype(f32)
    for m in range(nens):
        if bias_corr and m == nens - 1:
            continue
        f[:, m] = f[:, m] + pert[:, m] if typs == 0 else f[:, m] * pert[:, m]
    if bias_corr:
        nxx = nens
        delta = np.zeros(ncell, f32)
        for m in range(nxx - 1):
            delta = delta + (f[:, m] - f[:, nxx - 1])
        delta = delta / f32(nxx - 1)
        for m in range(nxx - 1):
            f[:, m] = f[:, m] - delta
    return np.minimum(np.maximum(f, f32(mins)), f32(maxs)).astype(f32)
