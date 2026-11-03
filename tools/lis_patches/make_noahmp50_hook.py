"""Apply the Noah-MP 5.0 dump hook to the BUILD COPY of NoahmpDriverMainMod.F90 (never the repo).
Records, for every land tile and time step, the inputs/states before the physics (pre) and the states/outputs after it (post) as float32:
all NoahmpIO scalars of interest, all NoahmpIO (i,j) and (i,k,j) arrays of the LIS driver (names as in NoahmpIO: TVXY, SMOIS, ...), and the whole
LisNoahmpParam_type structure (names 'p_<NAME>', the same names as the 4.0.1 hook). Output: jaxdump_mp.bin + jaxdump_mp_{pre,post}.layout in the run directory,
read by lis/dumpio_mp.py. Environment switches (read at run time): JAXDUMP_MAXSTEP=n records only the first n time steps; JAXDUMP_SNOWONLY=1 keeps snow tiles.
usage: python3 make_noahmp50_hook.py [build_dir]   (default: the author's build copy)"""
import re, sys, os, shutil

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
BUILD = _args[0] if _args else "/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_build/lis/surfacemodels/land/noahmp.5.0/phys/noahmp/drivers/lis/"
path = BUILD + "NoahmpDriverMainMod.F90"
if not os.path.exists(path + ".orig"):
    shutil.copy(path, path + ".orig")
src = open(path + ".orig").read()
iotype = open(BUILD + "NoahmpIOVarType.F90").read()
ptype = open(BUILD + "LisNoahmpParamType.F90").read()

# ---- fields of NoahmpIO_type: name, type, rank
txt = re.sub(r"&\s*\n\s*&?", "", iotype)
fields = []
for ln in txt.split("\n"):
    ln = ln.split("!")[0]
    if "::" not in ln:
        continue
    decl, rest = ln.split("::", 1)
    d = re.search(r"dimension\(([^)]*)\)", decl)
    rank = len(d.group(1).split(",")) if d else 0
    d0 = decl.strip().lower()
    kind = "i" if d0.startswith("integer") else ("l" if d0.startswith("logical") else ("c" if d0.startswith("character") else "r"))
    for nm in rest.split(","):
        nm = nm.split("=")[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", nm):
            fields.append((nm, kind, rank))
names = [f[0] for f in fields]
i0, i1 = names.index("ivgtyp"), names.index("NONRIVERXY")
arr = [f for f in fields[i0:i1 + 1] if f[2] in (2, 3)]
arr += [f for f in fields if f[0] in ("SWDDIR", "SWDDIF", "TERRAIN", "GVFMIN", "GVFMAX", "VEGFRA") and f[2] == 2]
# per-point parameter arrays handed to the physics through NoahmpIO (soil properties per layer, 2-D irrigation/runoff/tile-drain parameters)
arr += [f for f in fields if f[2] in (2, 3) and f[1] != "c" and re.search(r"(_3d|_2d)$", f[0], re.I)]
arr += [f for f in fields if f[0] in ("KLAT_FAC", "TDSMC_FAC", "TD_DC", "TD_DCOEF", "TD_DDRAIN", "TD_RADI", "TD_SPAC") and f[2] == 2 and f[0] not in [a[0] for a in arr]]
seen = set(); arr = [a for a in arr if not (a[0] in seen or seen.add(a[0]))]
vec = [f for f in fields if f[0] in ("DZS", "ZSOIL")]
scal_names = ("IOPT_DVEG IOPT_CRS IOPT_BTR IOPT_RUNSRF IOPT_RUNSUB IOPT_SFC IOPT_FRZ IOPT_INF IOPT_RAD IOPT_ALB IOPT_SNF IOPT_TKSNO IOPT_TBOT IOPT_STC "
              "IOPT_GLA IOPT_RSF IZ0TLND IOPT_SOIL IOPT_PEDO IOPT_CROP IOPT_IRR IOPT_IRRM IOPT_INFDV IOPT_TDRN sf_urban_physics year month day hour minute "
              "itimestep nsoil nsnow YR dtbl ICE XICE_THRESHOLD JULIAN DX soiltstep calculate_soil soil_update_steps YEARLEN FNDSNOWH").split()
scal = [f for f in fields if f[0] in scal_names and f[2] == 0]
scal += [f for f in fields if f[2] == 0 and f[1] in ("r", "i") and f[0].endswith("_TABLE") and f[0] not in scal_names]       # table scalars (global parameters)
scal += [f for f in fields if f[0] in ("SLOPETYP", "ISURBAN", "URBTYPE_beg") and f[2] == 0]
miss = [n for n in scal_names if n not in [f[0] for f in scal]]
if miss:
    print("scalars not found:", miss)

# ---- LisNoahmpParam_type fields
consts = {"MBAND": 2, "NSOIL": 4, "NSTAGE": 8}
pf = []
for ln in ptype.split("\n"):
    ln = ln.split("!")[0].strip()
    m = re.match(r"(real\(kind=kind_noahmp\)|integer|logical)\s*::\s*(\w+)(?:\((\w+)\))?\s*$", ln)
    if m:
        d = m.group(3)
        pf.append((m.group(2), m.group(1)[0], (consts[d] if d in consts else int(d)) if d else 0))


def W(expr, kind):
    return f"real({expr})" if kind == "r" else expr


def call(name, expr, kind, rank, pref=""):
    return f"             call jd_add('{pref}{name}', {W(expr, kind)})"


pre, post = [], []
for nm, k, r in scal:
    pre.append(call(nm, f"NoahmpIO%{nm}", k, 0))
for nm, k, r in vec:
    pre.append(f"             if (allocated(NoahmpIO%{nm})) call jd_add('{nm}', {W('NoahmpIO%' + nm, k)})")
for nm, k, r in arr:
    e = f"NoahmpIO%{nm}(I,J)" if r == 2 else f"NoahmpIO%{nm}(I,lbound(NoahmpIO%{nm},2):ubound(NoahmpIO%{nm},2),J)"
    line = f"             if (allocated(NoahmpIO%{nm})) call jd_add('{nm}', {W(e, k)})"
    pre.append(line); post.append(line)
for nm, k, r in pf:
    pre.append(f"             call jd_add('p_{nm}', {W('LISparam%' + nm, k)})")
post_sc = ["ITIMESTEP"]  # placeholder (post record carries only the arrays)
# ---- optional: physics derived-type members (intermediate quantities) in the post record, switch --types
DEBUG_TYPES = "--types" in sys.argv
dbg = []
if DEBUG_TYPES:
    SRC = BUILD.replace("drivers/lis/", "src/")
    for modfile, cont in (("EnergyVarType", "energy"), ("WaterVarType", "water"), ("BiochemVarType", "biochem")):
        t = re.sub(r"&\s*\n\s*&?", "", open(SRC + modfile + ".F90").read())
        subs = {}
        for m in re.finditer(r"type\s*::\s*(\w+)\s*\n(.*?)end\s+type", t, re.S | re.I):
            subs[m.group(1).lower()] = m.group(2)
        for sub, pref in (("flux_type", "f"), ("state_type", "s"), ("parameter_type", "p")):
            body = subs.get(sub)
            if body is None:
                continue
            for ln in body.split("\n"):
                ln = ln.split("!")[0]
                if "::" not in ln:
                    continue
                decl, rest = ln.split("::", 1)
                d0 = decl.strip().lower()
                if d0.startswith("character"):
                    continue
                kind = "i" if d0.startswith("integer") else ("l" if d0.startswith("logical") else "r")
                alloc = "allocatable" in d0
                comp = {"f": "flux", "s": "state", "p": "param"}[pref]
                for nm in rest.split(","):
                    nm = nm.split("=")[0].strip()
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", nm):
                        continue
                    vpath = f"noahmp%{cont}%{comp}%{nm}"
                    name = f"{cont[0]}{pref}_{nm}"
                    if alloc:
                        rank = len(re.search(r"dimension\(([^)]*)\)", d0).group(1).split(",")) if "dimension" in d0 else 1
                        if rank == 1 and kind != "l":
                            dbg.append(f"             if (allocated({vpath})) call jd_add('{name}', {W(vpath, kind)})")
                    else:
                        dbg.append(f"             call jd_add('{name}', {W(vpath, kind)})")
    post += dbg
    print("derived-type debug members:", len(dbg))


hook_pre = ["             jd_keep = .true.",
            "             if (jd_snowonly()) jd_keep = (NoahmpIO%SNOW(I,J) > 0.) .or. (NoahmpIO%SNOWH(I,J) > 0.) .or. (NoahmpIO%ISNOWXY(I,J) < 0)",
            "             if (jd_maxstep() > 0) then",
            "               if (NoahmpIO%ITIMESTEP > jd_maxstep()) jd_keep = .false.",
            "             end if",
            "             call jd_begin()"] + pre + ["             call jd_flush(1)"]
hook_post = ["             call jd_begin()"] + post + ["             call jd_flush(2)"]

module = '''module jaxdump50_mod
  ! Debug/validation hook (not part of LISF): records Noah-MP 5.0 driver inputs/states (pre) and states/outputs (post) as float32 per land tile and step.
  implicit none
  integer, parameter :: JD_MAX = 6000
  real    :: jd_buf(JD_MAX)
  character(len=40) :: jd_nm(JD_MAX)
  integer :: jd_sz(JD_MAX)
  integer :: jd_n = 0, jd_pos = 0
  integer :: jd_unit = 0
  logical :: jd_open = .false., jd_layout_done(2) = .false., jd_keep = .true.
  interface jd_add
    module procedure jd_add_r, jd_add_i, jd_add_l, jd_add_rv, jd_add_iv
  end interface
contains
  logical function jd_snowonly()
    logical, save :: checked = .false., val = .false.
    character(len=8) :: ev
    integer :: ln
    if (.not. checked) then
      call get_environment_variable('JAXDUMP_SNOWONLY', ev, ln)
      val = (ln > 0); checked = .true.
    end if
    jd_snowonly = val
  end function jd_snowonly
  integer function jd_maxstep()
    integer, save :: val = -1
    character(len=16) :: ev
    integer :: ln, ios
    if (val < 0) then
      val = 0
      call get_environment_variable('JAXDUMP_MAXSTEP', ev, ln)
      if (ln > 0) read(ev, *, iostat=ios) val
    end if
    jd_maxstep = val
  end function jd_maxstep
  subroutine jd_begin()
    jd_n = 0; jd_pos = 0
  end subroutine jd_begin
  subroutine jd_push(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v(:)
    jd_n = jd_n + 1
    jd_nm(jd_n) = name
    jd_sz(jd_n) = size(v)
    jd_buf(jd_pos+1:jd_pos+size(v)) = v
    jd_pos = jd_pos + size(v)
  end subroutine jd_push
  subroutine jd_add_r(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x
    call jd_push(name, (/ x /))
  end subroutine jd_add_r
  subroutine jd_add_i(name, x)
    character(len=*), intent(in) :: name
    integer, intent(in) :: x
    call jd_push(name, (/ real(x) /))
  end subroutine jd_add_i
  subroutine jd_add_l(name, x)
    character(len=*), intent(in) :: name
    logical, intent(in) :: x
    call jd_push(name, (/ merge(1.0, 0.0, x) /))
  end subroutine jd_add_l
  subroutine jd_add_rv(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x(:)
    call jd_push(name, x)
  end subroutine jd_add_rv
  subroutine jd_add_iv(name, x)
    character(len=*), intent(in) :: name
    integer, intent(in) :: x(:)
    call jd_push(name, real(x))
  end subroutine jd_add_iv
  subroutine jd_flush(which)
    integer, intent(in) :: which
    integer :: i, lu
    if (.not. jd_keep) return
    if (.not. jd_open) then
      open(newunit=jd_unit, file='jaxdump_mp.bin', access='stream', form='unformatted', status='replace')
      jd_open = .true.
    end if
    if (.not. jd_layout_done(which)) then
      open(newunit=lu, file=merge('jaxdump_mp_pre.layout ', 'jaxdump_mp_post.layout', which == 1), status='replace')
      do i = 1, jd_n
        write(lu, '(a,1x,i0)') trim(jd_nm(i)), jd_sz(i)
      end do
      close(lu)
      jd_layout_done(which) = .true.
    end if
    write(jd_unit) which, jd_pos
    write(jd_unit) jd_buf(1:jd_pos)
  end subroutine jd_flush
end module jaxdump50_mod

'''

# insert module before the driver module, the 'use' line, and the hooks
k = src.index("module NoahmpDriverMainMod")
src = src[:k] + module + src[k:]
src = src.replace("  use NoahmpMainGlacierMod\n", "  use NoahmpMainGlacierMod\n  use jaxdump50_mod\n", 1)
m1 = "             call ConfigVarInitDefault  (noahmp)\n"
assert m1 in src
src = src.replace(m1, "\n".join(hook_pre) + "\n" + m1, 1)
m2 = "             call BiochemVarOutTransfer(noahmp, NoahmpIO)"
j = src.index(m2) + len(m2)
j = src.index("\n", j) + 1
src = src[:j] + "\n".join(hook_post) + "\n" + src[j:]
open(path, "w").write(src)
print("scalars", len(scal), "vectors", len(vec), "arrays", len(arr), "LISparam", len(pf))
