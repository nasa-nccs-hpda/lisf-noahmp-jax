"""Land Data Toolkit (LDT) parameter processing for the Noah-MP 4.0.1 configuration of the LISF walkthrough, in Python.
Reads an LDT config (the keys of lisf_run/ldt_mp/ldt.config) and the raw parameter maps and writes the netCDF parameter file that
LIS reads (variables LANDCOVER, SURFACETYPE, TEXTURE, ELEVATION, SLOPE, ASPECT, GREENNESS, ALBEDO, SHDMIN/MAX, MXSNALBEDO, TBOT, ...).
Supported (everything else raises NotImplementedError rather than guessing):
  latlon run domain; landcover MODIS_Native IGBPNCEP 'tile'; soil texture STATSGOFAO_Native 'tile'; elevation/slope/aspect SRTM_Native
  'average'; NCEP_Native monthly albedo, greenness, greenness max/min with 'budget-bilinear'; Barlage_Native max snow albedo 'average'
  (HDF4 read by a minimal parser); ISLSCP1 bottom temperature 'budget-bilinear' with lapse-rate correction; constant slope type; Noah-MP PBLH.
Mirrors ldt/params/*/read_*.F90, ldt/interp/{upscaleByAveraging_input,upscaleByCnt,conserv_interp_input,conserv_interp}.F90 and the
fgrd/mask logic of ldt/core. Arithmetic is float32 in the same order as the Fortran. Usage: python -m lisf_noahmp_jax.driver.ldt ldt.config [outfile]
No 'fill' options are implemented: if a land cell is undefined in a field, the run stops (the walkthrough domains do not need it)."""
import os, sys, struct, datetime as dt
import numpy as np
import netCDF4 as nc
from .config import Config

f32 = np.float32
UDEF = f32(-9999.0)
NT_LC = 20                     # IGBPNCEP classes
NSOIL = 16                     # STATSGO/FAO texture classes
SOIL_WATER = 14
LC_WATER = 17


def nint(v):
    """Fortran NINT (half away from zero)."""
    return (np.floor(np.abs(v) + f32(0.5)) * np.sign(v)).astype(np.int64)


class Domain:
    def __init__(self, cfg):
        assert cfg.s("Map projection of the LIS domain") == "latlon"
        self.ll_lat = f32(cfg.f("Run domain lower left lat")); self.ll_lon = f32(cfg.f("Run domain lower left lon"))
        self.ur_lat = f32(cfg.f("Run domain upper right lat")); self.ur_lon = f32(cfg.f("Run domain upper right lon"))
        self.dx = f32(cfg.f("Run domain resolution (dx)")); self.dy = f32(cfg.f("Run domain resolution (dy)"))
        self.nx = int(round((float(self.ur_lon) - float(self.ll_lon)) / float(self.dx))) + 1
        self.ny = int(round((float(self.ur_lat) - float(self.ll_lat)) / float(self.dy))) + 1
        j = np.arange(-2, self.ny + 2, dtype=np.float64); i = np.arange(-2, self.nx + 2, dtype=np.float64)
        self.lat_b = np.repeat((float(self.ll_lat) + j * float(self.dy)).astype(f32)[:, None], self.nx + 4, 1)
        self.lon_b = np.repeat((float(self.ll_lon) + i * float(self.dx)).astype(f32)[None, :], self.ny + 4, 0)
        self.lat = np.ascontiguousarray(self.lat_b[2:-2, 2:-2]); self.lon = np.ascontiguousarray(self.lon_b[2:-2, 2:-2])


class PixGrid:
    """Input latlon grid as LDT's param_gridDesc: lower-left pixel centre, resolution, size (rows south-first)."""
    def __init__(self, lat1, lon1, d, nc_, nr_):
        self.lat1, self.lon1, self.dy, self.dx, self.nc, self.nr = f32(lat1), f32(lon1), f32(d), f32(d), nc_, nr_


def cell_index(dom, pg, y0, y1, x0, x1):
    """upscaleByAveraging_input for the pixel block rows y0:y1, cols x0:x1 (0-based, south-first): output cell (j,i), 0-based, or -1."""
    y = np.arange(y0, y1); x = np.arange(x0, x1)
    rlon1 = pg.lon1 + f32(360) if pg.lon1 < 0 else pg.lon1
    rlon = (rlon1 + pg.dx * (x.astype(f32) + f32(1) - f32(1))).astype(f32)
    rlon = np.where(rlon > 360, rlon - f32(360), rlon).astype(f32)
    rlat = (pg.lat1 + pg.dy * (y.astype(f32) + f32(1) - f32(1))).astype(f32)
    xo = np.where(rlon > 180, f32(1) + (rlon - f32(360) - dom.ll_lon) / dom.dx, f32(1) + (rlon - dom.ll_lon) / dom.dx).astype(f32)
    yo = (f32(1) + (rlat - dom.ll_lat) / dom.dy).astype(f32)
    io = nint(xo); jo = nint(yo)
    jv = (jo >= 1) & (jo <= dom.ny); iv = (io >= 1) & (io <= dom.nx)
    J = np.where(jv, jo - 1, -1)[:, None]; I = np.where(iv, io - 1, -1)[None, :]
    return np.where((J >= 0) & (I >= 0), J * dom.nx + I, -1)


def window(dom, pg):
    """Pixel index ranges (0-based) of LDT_RunDomainPts for an input grid finer than the domain: pixel centres from the cell edges inward."""
    lo_lat = float(dom.ll_lat) - float(dom.dy) / 2 + float(pg.dy) / 2; hi_lat = float(dom.ur_lat) + float(dom.dy) / 2 - float(pg.dy) / 2
    lo_lon = float(dom.ll_lon) - float(dom.dx) / 2 + float(pg.dx) / 2; hi_lon = float(dom.ur_lon) + float(dom.dx) / 2 - float(pg.dx) / 2
    y0 = int(round((lo_lat - float(pg.lat1)) / float(pg.dy))); y1 = int(round((hi_lat - float(pg.lat1)) / float(pg.dy))) + 1
    x0 = int(round((lo_lon - float(pg.lon1)) / float(pg.dx))); x1 = int(round((hi_lon - float(pg.lon1)) / float(pg.dx))) + 1
    return y0, y1, x0, x1


def tile_counts(dom, cell, vals, ncls):
    """upscaleByCnt: pixels per class per output cell. vals: integer classes 1..ncls (others ignored)."""
    ok = (cell >= 0) & (vals >= 1) & (vals <= ncls)
    flat = cell[ok] * ncls + (vals[ok] - 1)
    return np.bincount(flat, minlength=dom.ny * dom.nx * ncls).reshape(dom.ny, dom.nx, ncls).transpose(2, 0, 1).astype(f32)


def average(dom, cell, vals, valid):
    """upscaleByAveraging: sequential float32 sum in input order divided by the count; UDEF where no valid pixel."""
    ok = (cell >= 0) & valid
    c = cell[ok]; v = vals[ok].astype(f32)
    s = np.zeros(dom.ny * dom.nx, f32); n = np.zeros(dom.ny * dom.nx, np.int64)
    np.add.at(s, c, v); np.add.at(n, c, 1)
    return np.where(n > 0, s / np.maximum(n, 1), UDEF).astype(f32).reshape(dom.ny, dom.nx)


def read_ascii_i2(path, nc_, nr_):
    out = np.empty((nr_, nc_), np.int32)
    with open(path, "rb") as fh:
        for r in range(nr_):
            s = fh.readline()
            out[r] = [int(s[2 * c:2 * c + 2]) for c in range(nc_)]
    return out


def budget_bilinear(dom, gi, li, pg):
    """conserv_interp_input + conserv_interp (25-point inverse-distance weighted bilinear); gi, li: (nr, nc) south-first float32/bool."""
    im, jm = pg.nc, pg.nr
    go = np.zeros((dom.ny, dom.nx), f32); wo = np.zeros((dom.ny, dom.nx), f32)
    for jb in range(-2, 3):
        for ib in range(-2, 3):
            rlab = dom.lat_b[2 + jb:2 + jb + dom.ny, 2 + ib:2 + ib + dom.nx]; rlob = dom.lon_b[2 + jb:2 + jb + dom.ny, 2 + ib:2 + ib + dom.nx]
            xi = (f32(1) + (rlob - pg.lon1) / pg.dx).astype(f32); yi = (f32(1) + (rlab - pg.lat1) / pg.dy).astype(f32)
            i1 = xi.astype(np.int64); j1 = yi.astype(np.int64); xf = (xi - i1).astype(f32); yf = (yi - j1).astype(f32)
            rsq = ((dom.lat - rlab) ** 2 + (dom.lon - rlob) ** 2).astype(f32)
            idw = np.where(rsq == 0, f32(1), f32(1) / np.where(rsq == 0, f32(1), rsq)).astype(f32)
            w11 = (f32(1) - xf) * (f32(1) - yf) * idw; w21 = xf * (f32(1) - yf) * idw
            w12 = (f32(1) - xf) * yf * idw; w22 = xf * yf * idw
            inside = (i1 >= 1) & (i1 + 1 <= im) & (j1 >= 1) & (j1 + 1 <= jm)          # all four corners must exist
            for ii, jj, w in ((i1, j1, w11), (i1 + 1, j1, w21), (i1, j1 + 1, w12), (i1 + 1, j1 + 1, w22)):
                iic = np.clip(ii - 1, 0, im - 1); jjc = np.clip(jj - 1, 0, jm - 1)
                use = inside & li[jjc, iic]
                go += np.where(use, w * gi[jjc, iic], f32(0)); wo += np.where(use, w, f32(0))
    good = wo >= f32(0.5 * 25)
    return np.where(good, go / np.where(good, wo, f32(1)), UDEF).astype(f32)


def read_hdf4_float32_sds(path, nx, ny):
    """Minimal HDF4 reader: the one uncompressed big-endian float32 SDS data element of nx*ny values (tag 702)."""
    with open(path, "rb") as fh:
        assert fh.read(4) == b"\x0e\x03\x13\x01", "not an HDF4 file"
        dds = []; nxt = 4
        while nxt:
            fh.seek(nxt); n, nxt = struct.unpack(">HI", fh.read(6))
            for _ in range(n):
                tag, ref, off, ln = struct.unpack(">HHII", fh.read(12))
                dds.append((tag, ref, off, ln))
        cand = [(o, l) for t, r, o, l in dds if t == 702 and l == nx * ny * 4]
        if len(cand) != 1:
            raise NotImplementedError(f"{path}: expected one float32 SDS of {nx}x{ny}, found {len(cand)}")
        return np.fromfile(path, dtype=">f4", count=nx * ny, offset=cand[0][0]).reshape(ny, nx).astype(f32)


def fill_cont(A, mask, option, value, radius):
    """LDT_contIndivParam_Fill for each layer of A (L, ny, nx): in-place sequential sweep (row-major) over mask==1 cells with UDEF; 'neighbor' takes
    the first defined value of the LAST row of the (2*irad+1)-square that has one (the Fortran 'exit' leaves only the inner loop), 'average' the
    mean of the square; cells with nothing within the radius get the fill value; masked-out cells become UDEF."""
    L, ny, nx = A.shape
    for l in range(L):
        a = A[l]
        found = np.zeros((ny, nx), np.int32)
        todo = np.argwhere((mask == 1) & (a == UDEF))
        for j, i in todo:
            for irad in range(1, int(round(radius)) + 1):
                cols = (np.arange(i - irad, i + irad + 1)) % nx
                if option == "average":
                    s = f32(0); n = 0
                    for j1 in range(j - irad, j + irad + 1):
                        v = a[min(max(j1, 0), ny - 1), cols]; d = v != UDEF
                        for x in v[d]:
                            s = f32(s + x)
                        n += int(d.sum())
                    if n >= 1:
                        found[j, i] = n; a[j, i] = f32(s / f32(n))
                        break
                else:
                    for j1 in range(j - irad, j + irad + 1):
                        v = a[min(max(j1, 0), ny - 1), cols]; d = np.nonzero(v != UDEF)[0]
                        if d.size:
                            found[j, i] += 1; a[j, i] = v[d[0]]
                    if found[j, i] >= 1:
                        break
        bad = (mask == 1) & (a == UDEF) & (found < 1)
        a[bad] = f32(value)
        a[(mask == 0) & (a != UDEF)] = UDEF
    return A


def fill_discrete(dom, T, mask, fill_value, radius, fill_antarctica, wcut):
    """LDT_discreteParam_Fill for a fraction-per-class array T (L, ny, nx), water class SOIL_WATER excluded from the sums (force-exclude option):
    land cells whose non-water fractions sum to less than 1-wcut copy the whole class vector of the first neighbour (row-major scan of the
    growing square) that has at least 1-wcut; non-land cells become one-hot water. In-place sequential sweep."""
    L, ny, nx = T.shape
    keep = np.array([l != SOIL_WATER - 1 for l in range(L)])
    isum = T[keep].sum(0, dtype=f32)
    thr = f32(1.0) - wcut
    for j in range(ny):
        for i in range(nx):
            if mask[j, i] == 1 and isum[j, i] < thr:
                done = False
                for irad in range(1, radius + 1):
                    for j1 in range(j - irad, j + irad + 1):
                        jx = min(max(j1, 0), ny - 1)
                        for i1 in range(i - irad, i + irad + 1):
                            ix = i1 % nx
                            if isum[jx, ix] >= thr:
                                T[:, j, i] = T[:, jx, ix]; isum[j, i] = isum[jx, ix]; done = True
                                break
                        if done: break
                    if done: break
                if not done:
                    T[:, j, i] = 0
                    T[(fill_antarctica if dom.lat[j, i] < -60 else fill_value) - 1, j, i] = 1
                    isum[j, i] = f32(1) if (fill_antarctica if dom.lat[j, i] < -60 else fill_value) != SOIL_WATER else f32(0)
            elif mask[j, i] == 0:
                T[:, j, i] = 0; T[SOIL_WATER - 1, j, i] = 1; isum[j, i] = 0
    return T


def _need_defined(name, a, mask=None):
    bad = (a == UDEF) if mask is None else ((a == UDEF) & mask)
    if bad.any():
        raise NotImplementedError(f"{name}: {int(bad.sum())} undefined land cells; the 'fill' options are not ported")


def process(cfg_path):
    cfg = Config(cfg_path); base = os.path.dirname(os.path.abspath(cfg_path))
    P = lambda p: p if p.startswith("/") else os.path.join(base, p)
    assert cfg.s("Land surface model") == "Noah-MP.4.0.1"
    dom = Domain(cfg)
    ny, nx = dom.ny, dom.nx
    # --- landcover (MODIS_Native, IGBPNCEP, tile) and land mask
    assert cfg.s("Landcover data source") == "MODIS_Native" and cfg.s("Landcover classification") == "IGBPNCEP"
    assert cfg.s("Landcover spatial transform") == "tile" and cfg.s("Create or readin landmask") == "create"
    res = f32(1.0) / f32(120.0)
    pg = PixGrid(-90.0 + float(res) / 2, -180.0 + float(res) / 2, res, 43200, 21600)
    y0, y1, x0, x1 = window(dom, pg)
    igbp = np.memmap(P(cfg.s("Landcover file")), dtype=np.uint8, mode="r").reshape(21600, 43200)[::-1]     # file rows are north-first
    cell = cell_index(dom, pg, y0, y1, x0, x1)
    cnt = tile_counts(dom, cell, igbp[y0:y1, x0:x1].astype(np.int64), NT_LC + 1)                           # 21 classes incl. a spare slot
    cnt = cnt[:NT_LC]
    # param_index_fgrdcalc (inc_water_pts true)
    tot = cnt.sum(0, dtype=f32)
    fgrd = np.where(tot > 0, cnt / np.where(tot > 0, tot, 1), f32(0)).astype(f32)
    vegsum = cnt.sum(0, dtype=f32) - cnt[LC_WATER - 1]
    wfrac = f32(cfg.f("Water fraction cutoff value"))
    mask = np.where(tot == 0, f32(0), np.where(vegsum / np.where(tot > 0, tot, 1) >= wfrac, f32(1), f32(0))).astype(f32)
    # LDT_assign_openwatersfctype: water fraction moves to the extra (21st) slot, flagged with the open-water index (5)
    landcover = np.zeros((NT_LC + 1, ny, nx), f32); landcover[:NT_LC] = fgrd
    landcover[NT_LC] = fgrd[LC_WATER - 1]; landcover[LC_WATER - 1] = 0
    sfctype = np.where(landcover > 0, f32(1), f32(0)).astype(f32)
    sfctype[NT_LC] = np.where(landcover[NT_LC] > 0, f32(5), f32(0))
    landmask = mask.copy()                                                                   # LANDMASK: land cells only (used by the fills)
    mask = np.where((landcover[NT_LC] > 0) & (mask == 0), f32(1), mask).astype(f32)         # DOMAINMASK: water cells included
    # --- soil texture (STATSGOFAO_Native, tile)
    assert cfg.s("Soil texture data source") == "STATSGOFAO_Native" and cfg.s("Soil texture spatial transform") == "tile"
    tex = np.memmap(P(cfg.s("Soil texture map")), dtype=np.int8, mode="r").reshape(21600, 43200)[::-1]
    tcnt = tile_counts(dom, cell, tex[y0:y1, x0:x1].astype(np.int64), NSOIL)
    ttot = tcnt.sum(0, dtype=f32)
    texture = np.where(ttot > 0, tcnt / np.where(ttot > 0, ttot, 1), f32(0)).astype(f32)
    topt = cfg.s("Soil texture fill option")
    if topt == "neighbor":
        texture = fill_discrete(dom, texture, landmask, cfg.i("Soil texture fill value"), cfg.i("Soil texture fill radius"),
                                cfg.i("Soil texture fill value for Antarctica"), wfrac)
    elif topt != "none":
        raise NotImplementedError(f"Soil texture fill option {topt}")
    # --- elevation, slope, aspect (SRTM_Native GTOPO30 tiles, average)
    assert cfg.s("Elevation data source") == "SRTM_Native" and cfg.s("Topography spatial transform") == "average"
    D = P(cfg.s("Elevation map")).rstrip("/") + "/"
    pgt = pg                                                                       # same 30 arcsec grid
    mos = _gtopo30_window(D, y0, y1, x0, x1)                                       # south-first, (y1-y0, x1-x0)
    elev = average(dom, cell, mos.ravel().reshape(mos.shape), mos != UDEF)
    slope_px = _slope(mos, f32(pgt.dy), f32(pgt.lat1) + f32(y0) * f32(pgt.dy), "slope")
    aspect_px = _slope(mos, f32(pgt.dy), f32(pgt.lat1) + f32(y0) * f32(pgt.dy), "aspect")
    slope = average(dom, cell, slope_px, slope_px != UDEF); aspect = average(dom, cell, aspect_px, aspect_px != UDEF)
    # --- monthly albedo / greenness, greenness max/min (NCEP native 0.144 deg ascii, budget-bilinear)
    pg14 = PixGrid(-89.928, -180.0, 0.144, 2500, 1250)
    for k in ("Albedo", "Greenness"):
        assert cfg.s(f"{k} data source") == "NCEP_Native" and cfg.s(f"{k} spatial transform").strip('"') == "budget-bilinear"
    mons = "jan feb mar apr may jun jul aug sep oct nov dec".split()
    def ascii_field(path):
        g = read_ascii_i2(path, 2500, 1250).astype(f32)
        o = budget_bilinear(dom, g, g > 0, pg14)
        return np.where(o <= 0, UDEF, o / f32(100)).astype(f32)
    alb_stub = P(cfg.s("Albedo map")); gfr_stub = P(cfg.s("Greenness fraction map"))
    albedo = np.stack([ascii_field(f"{alb_stub}_{m}.asc") for m in mons]); green = np.stack([ascii_field(f"{gfr_stub}_{m}.asc") for m in mons])
    shdmax = ascii_field(P(cfg.s("Greenness maximum map")))[None]; shdmin = ascii_field(P(cfg.s("Greenness minimum map")))[None]
    def F(a, key, valkey=None):
        opt = cfg.s(f"{key} fill option")
        if opt == "none":
            return a
        if opt not in ("neighbor", "average"):
            raise NotImplementedError(f"{key} fill option {opt}")
        return fill_cont(a, landmask, opt, cfg.f(valkey or f"{key} fill value"), cfg.f(f"{key} fill radius"))
    albedo = F(albedo, "Albedo"); green = F(green, "Greenness")
    shdmax = F(shdmax, "Greenness", "Greenness maximum fill value")[0]; shdmin = F(shdmin, "Greenness", "Greenness minimum fill value")[0]
    # --- maximum snow albedo (Barlage, HDF4, 0.05 deg, average)
    assert cfg.s("Max snow albedo data source") == "Barlage_Native" and cfg.s("Max snow albedo spatial transform") == "average"
    sa = read_hdf4_float32_sds(P(cfg.s("Max snow albedo map")), 7200, 3600)[::-1]
    pgs = PixGrid(-89.95, -180.0, 0.05, 7200, 3600)
    ys0, ys1, xs0, xs1 = _loose_window(dom, pgs)
    cs = cell_index(dom, pgs, ys0, ys1, xs0, xs1)
    sub = sa[ys0:ys1, xs0:xs1]
    mxsnalb = F(average(dom, cs, sub, sub > 0)[None], "Max snow albedo")[0]
    # --- bottom temperature (ISLSCP1 1 deg, budget-bilinear, lapse-rate correction)
    assert cfg.s("Bottom temperature data source") == "ISLSCP1" and cfg.s("Bottom temperature spatial transform").strip('"') == "budget-bilinear"
    t = (np.fromfile(P(cfg.s("Bottom temperature map")), dtype=">i2").reshape(180, 360).astype(f32) / f32(100))[::-1]
    tb = budget_bilinear(dom, t, t > 220, PixGrid(-89.5, -179.5, 1.0, 360, 180))
    tb = F(np.where(tb <= 220, UDEF, tb).astype(f32)[None], "Bottom temperature")[0]
    tb = np.where(tb != UDEF, tb + f32(-0.0065) * (elev - f32(0)), tb).astype(f32)
    for name, a in (("GREENNESS", green), ("ALBEDO", albedo), ("SHDMAX", shdmax), ("SHDMIN", shdmin), ("MXSNALBEDO", mxsnalb), ("TBOT", tb),
                    ("ELEVATION", elev), ("SLOPE", slope), ("ASPECT", aspect)):
        _need_defined(name, a, landmask > 0)
    return dict(dom=dom, landcover=landcover, sfctype=sfctype, mask=mask, texture=texture, elev=elev, slope=slope, aspect=aspect,
                green=green, albedo=albedo, shdmax=shdmax, shdmin=shdmin, mxsnalb=mxsnalb, tbot=tb,
                slopetype=np.where(landmask == 1, f32(cfg.f("Slope type fill value")), f32(0)).astype(f32), landmask=landmask,
                pblh=np.full((ny, nx), f32(cfg.f("Noah-MP PBL Height Value")), f32))


def _loose_window(dom, pg):
    """Pixel block covering the domain with a one-cell margin (pixels outside the cells map to -1)."""
    lo_lat = float(dom.ll_lat) - float(dom.dy); hi_lat = float(dom.ur_lat) + float(dom.dy)
    lo_lon = float(dom.ll_lon) - float(dom.dx); hi_lon = float(dom.ur_lon) + float(dom.dx)
    y0 = max(0, int((lo_lat - float(pg.lat1)) / float(pg.dy))); y1 = min(pg.nr, int((hi_lat - float(pg.lat1)) / float(pg.dy)) + 2)
    x0 = max(0, int((lo_lon - float(pg.lon1)) / float(pg.dx))); x1 = min(pg.nc, int((hi_lon - float(pg.lon1)) / float(pg.dx)) + 2)
    return y0, y1, x0, x1


def _gtopo30_window(D, y0, y1, x0, x1):
    """Mosaic of the GTOPO30-derived 'raw_wgtopo30antarc' tiles (4800x6000 int16 big-endian, 40x50 deg) for a pixel window of the global
    43200x21600 30-arcsec grid (south-first indices). Returns float32 with UDEF where the tile value is -9999."""
    ns = ["N90", "N40", "S10"]; we = ["W180", "W140", "W100", "W060", "W020", "E020", "E060", "E100", "E140"]
    out = np.full((y1 - y0, x1 - x0), UDEF, f32)
    for ys in range(y0, y1):
        pass
    # north-first mosaic row of south-first index y is 21599 - y (rows beyond the 3 tile rows are Antarctica: not supported)
    rn0 = 21599 - (y1 - 1); rn1 = 21599 - y0 + 1
    if rn1 > 18000:
        raise NotImplementedError("Antarctic GTOPO30 tiles are not supported")
    for r_t in range(3):
        a0, a1 = max(rn0, r_t * 6000), min(rn1, (r_t + 1) * 6000)
        if a0 >= a1:
            continue
        for c_t in range(9):
            b0, b1 = max(x0, c_t * 4800), min(x1, (c_t + 1) * 4800)
            if b0 >= b1:
                continue
            tile = np.memmap(f"{D}{we[c_t]}{ns[r_t]}.DEM", dtype=">i2", mode="r").reshape(6000, 4800)
            blk = np.asarray(tile[a0 - r_t * 6000:a1 - r_t * 6000, b0 - c_t * 4800:b1 - c_t * 4800]).astype(f32)
            # place into the south-first output
            rows_out = (21599 - np.arange(a0, a1)) - y0
            out[rows_out[:, None], (np.arange(b0, b1) - x0)[None, :]] = np.where(blk == -9999, UDEF, blk)
    return out


def _slope(elev, res, lat0, kind):
    """calc_Slope_fromElev / calc_Aspect_fromElev on the subset window (south-first rows, east-west wrap-around, LDT's row indexing)."""
    NY, NX = elev.shape
    pi = f32(np.arccos(f32(-1.0))); Re = f32(6378.1e3)
    out = np.full((NY, NX), UDEF, f32)
    for j in range(NY):
        if j == 0:
            m1, e0, p1 = elev[0], elev[0], elev[1]
        elif j == NY - 1:
            m1, e0, p1 = elev[NY - 2], elev[NY - 1], elev[NY - 1]
        else:
            m1, e0, p1 = elev[j - 1], elev[j], elev[j]               # LDT uses row j (not j+1) for the upper row of interior rows
        lat = f32(lat0 + f32(j) * res)
        dx = f32(Re * f32(np.cos(f32(lat * pi / f32(180.0)))) * f32(res * pi / f32(180.0))); dy = f32(Re * res * pi / f32(180.0))
        rows = []
        for r in (p1, e0, m1):
            rows.append((np.roll(r, 1), r, np.roll(r, -1)))
        (Az, Bz, Cz), (Dz, Ez, Fz), (Gz, Hz, Iz) = rows
        # missing neighbours take the centre value
        def fillc(a):
            return np.where(a == UDEF, Ez, a)
        Az, Bz, Cz, Dz, Fz, Gz, Hz, Iz = (fillc(a) for a in (Az, Bz, Cz, Dz, Fz, Gz, Hz, Iz))
        dzdx = -(((Az - Cz) + f32(2.0) * (Dz - Fz)) + (Gz - Iz)) / (f32(8.0) * dx)
        dzdy = (((Az - Gz) + f32(2.0) * (Bz - Hz)) + (Cz - Iz)) / (f32(8.0) * dy)
        if kind == "slope":
            v = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                a = pi - np.arctan(dzdy / dzdx) + pi * f32(0.5) * dzdx / np.abs(dzdx)
            v = np.where(dzdx != 0, a, np.where(dzdy < 0, f32(0.0), pi))
        out[j] = np.where(Ez == UDEF, UDEF, v).astype(f32)
    return out


def write(res, path, ref=None):
    dom = res["dom"]; ny, nx = dom.ny, dom.nx
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("east_west", nx); ds.createDimension("north_south", ny); ds.createDimension("east_west_b", nx + 4)
    ds.createDimension("north_south_b", ny + 4); ds.createDimension("time", 1)
    ds.createDimension("sfctypes", 21); ds.createDimension("soiltypes", 16); ds.createDimension("month", 12)
    ds.MAP_PROJECTION = "EQUIDISTANT CYLINDRICAL"; ds.SOUTH_WEST_CORNER_LAT = dom.ll_lat; ds.SOUTH_WEST_CORNER_LON = dom.ll_lon
    ds.DX = dom.dx; ds.DY = dom.dy
    ds.title = "LDT parameter-processed output (Python port, lisf_noahmp_jax.driver.ldt)"; ds.history = f"created on date: {dt.datetime.now().isoformat()}"
    def var(name, a, dims):
        v = ds.createVariable(name, "f4", dims)
        v.scale_factor = f32(1.0); v.add_offset = f32(0.0); v.missing_value = UDEF          # LDT's attributes: undefined cells read back as masked
        v[:] = a
    var("time", np.zeros(1, f32), ("time",))
    m = res["mask"]
    var("DOMAINMASK", m, ("north_south", "east_west")); var("LANDMASK", res["landmask"], ("north_south", "east_west"))
    var("SURFACETYPE", res["sfctype"], ("sfctypes", "north_south", "east_west")); var("LANDCOVER", res["landcover"], ("sfctypes", "north_south", "east_west"))
    var("TEXTURE", res["texture"], ("soiltypes", "north_south", "east_west"))
    one = np.ones((ny, nx), f32); g2 = ("north_south", "east_west")
    var("ELEVFGRD", one, g2); var("ELEVATION", res["elev"], g2); var("SLOPEFGRD", one, g2); var("SLOPE", res["slope"], g2)
    var("ASPECTFGRD", one, g2); var("ASPECT", res["aspect"], g2)
    var("GREENNESS", res["green"], ("month",) + g2); var("SHDMIN", res["shdmin"], g2); var("SHDMAX", res["shdmax"], g2)
    var("ALBEDO", res["albedo"], ("month",) + g2); var("MXSNALBEDO", res["mxsnalb"], g2); var("TBOT", res["tbot"], g2)
    var("SLOPETYPE", res["slopetype"], g2); var("NOAHMP36_PBLH", res["pblh"], g2)
    var("lat", dom.lat, g2); var("lon", dom.lon, g2)
    var("lat_b", dom.lat_b, ("north_south_b", "east_west_b")); var("lon_b", dom.lon_b, ("north_south_b", "east_west_b"))
    ds.close()


if __name__ == "__main__":
    out = sys.argv[2] if len(sys.argv) > 2 else "lis_input.py.nc"
    write(process(sys.argv[1]), out)
    print("wrote", out)
