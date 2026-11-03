"""Run the JAX Noah 3.9 port on a case file written by baseline/gen_case.py and compare to the
Fortran baseline output. Modes: free (whole day free-running), open (one step from each recorded
Fortran start state)."""
import sys, time, json
import os
import numpy as np
import jax, jax.numpy as jnp
X64 = bool(os.environ.get('JAX_X64'))   # diagnostic only: run the same code in float64
if X64:
    jax.config.update('jax_enable_x64', True)
DT = np.float64 if X64 else np.float32
from jax import lax
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
import spec
from lisf_noahmp_jax.physics import noah39_jax as nj

f32 = np.float32


def load_case(path):
    with open(path, "rb") as f:
        N, NT, NS = np.fromfile(f, np.int32, 3)
        ints = {k: np.fromfile(f, np.int32, N) for k in spec.STATIC_INT}
        par = {k: np.fromfile(f, f32, N) for k in spec.STATIC}
        init = {k: np.fromfile(f, f32, N) for k in spec.INIT}
        lay = {k: np.fromfile(f, f32, NS * N).reshape(NS, N) for k in spec.INIT_LAYER}
        forc = {k: np.fromfile(f, f32, NT * N).reshape(NT, N) for k in spec.FORC}
    par.update(ints)
    st = dict(init)
    st.update(lay)
    return int(N), int(NT), par, st, forc


def to_st(st):
    return {k: jnp.asarray(v, dtype=(DT if v.dtype == np.float32 else v.dtype)) for k, v in st.items()}


def cast(v):
    v = np.asarray(v)
    return jnp.asarray(v, dtype=(DT if v.dtype == np.float32 else v.dtype))


def build_step(opt_thcnd=1):
    def one(par, st, fc):
        return nj.column_step(par, st, fc, opt_thcnd=opt_thcnd)
    # layers carried as (NS, N): vmap over axis 1 for layer arrays, axis 0 for scalars
    st_axes = {k: (1 if k in spec.INIT_LAYER else 0) for k in spec.INIT + spec.INIT_LAYER}
    par_axes = {k: 0 for k in spec.STATIC + spec.STATIC_INT}
    fc_axes = {k: 0 for k in spec.FORC}
    v = jax.vmap(one, in_axes=(par_axes, st_axes, fc_axes), out_axes=(
        {k: (1 if k in spec.INIT_LAYER else 0) for k in spec.INIT + spec.INIT_LAYER}, 0))
    return v


def pack_out(st, out):
    cols = []
    for n in spec.OUT:
        if n.startswith("stc") and n[3:].isdigit():
            cols.append(st["stc"][int(n[3:]) - 1])
        elif n.startswith("smc") and n[3:].isdigit():
            cols.append(st["smc"][int(n[3:]) - 1])
        elif n.startswith("sh2o") and n[4:].isdigit():
            cols.append(st["sh2o"][int(n[4:]) - 1])
        elif n in st:
            cols.append(st[n])
        else:
            cols.append(out[n])
    return jnp.stack(cols)  # (NV, N)


def make_runner(par, N, NT):
    step = build_step()
    parj = {k: cast(v) for k, v in par.items()}

    def day(st0, forc):
        def body(st, fc):
            new_st, out = step(parj, st, fc)
            return new_st, pack_out(new_st, out)
        return lax.scan(body, st0, forc)

    return jax.jit(day), parj, step


if __name__ == "__main__":
    case, ref_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    N, NT, par, st0, forc = load_case(case)
    NV = len(spec.OUT)
    ref = np.fromfile(ref_path, f32).reshape(NT, NV, N)
    day, parj, step = make_runner(par, N, NT)
    forcj = {k: cast(v) for k, v in forc.items()}
    if mode == "free":
        t = time.time(); _, o = day(to_st(st0), forcj); o.block_until_ready(); t1 = time.time() - t
        t = time.time(); _, o = day(to_st(st0), forcj); o.block_until_ready(); t2 = time.time() - t
        np.asarray(o).astype(f32).tofile(case.replace(".bin", "_jax_free64.bin" if X64 else "_jax_free.bin"))
        print(f"free-running day: first call {t1:.3f}s (incl compile), second {t2:.4f}s, N={N}")
    elif mode == "open":
        # one step from each recorded Fortran start state
        outs = []
        jstep = jax.jit(step)
        for s in range(NT):
            if s == 0:
                stj = to_st(st0)
            else:
                r = ref[s - 1]
                g = lambda n: cast(r[spec.OUT.index(n)])
                stj = {k: g(k) for k in spec.INIT}
                for lk in spec.INIT_LAYER:
                    stj[lk] = jnp.stack([g(f"{lk}{k}") for k in range(1, 5)])
            fc = {k: cast(v[s]) for k, v in forc.items()}
            new_st, out = jstep(parj, stj, fc)
            outs.append(np.asarray(pack_out(new_st, out)))
        np.stack(outs).astype(f32).tofile(case.replace(".bin", "_jax_open64.bin" if X64 else "_jax_open.bin"))
        print("open-loop done")
