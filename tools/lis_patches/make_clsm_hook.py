"""Insert the dump hook around the clsmf25 core call in clsmf25_main.F90 of the BUILD COPY.
usage: python3 make_clsm_hook.py <build>/lis/surfacemodels/land/clsm.f2.5   (restore the original with cp -f from the repo to undo)"""
import re, sys, os, shutil
d = sys.argv[1]
src = open(f"{d}/clsmf25_main.F90").read()
assert "jaxdump_clsm_mod" not in src
shutil.copy(os.path.join(os.path.dirname(__file__), "jaxdump_clsm_mod.F90"), d)
i = src.index("     call clsmf25( N_cat, dtstep, sfrac,")
j = src.index("clsmf25_struc(nid)%cat_route%fldfrc)", i) + len("clsmf25_struc(nid)%cat_route%fldfrc)")
call = src[i:j]
body = call[call.index("(") + 1: call.rindex(")")]
body = "\n".join(l.split("!")[0] for l in body.split("\n"))
body = re.sub(r"&\s*\n", "", body)                      # join continuation lines
args, depth, cur = [], 0, ""
for ch in body:
    if ch == "(":
        depth += 1
    if ch == ")":
        depth -= 1
    if ch == "," and depth == 0:
        args.append(cur.strip()); cur = ""
    else:
        cur += ch
args.append(cur.strip())
rank2k = {"GHTCNT", "WESN", "HTSN", "SNDZ"}
rank2t = {"fices"}
lines, layout = [], []
for a in args:
    if a.startswith("."):
        continue
    if a.lower() in ("n_cat",):
        continue
    if a.lower() in ("dtstep", "sfrac"):
        lines.append(f"call jdc_s({a})"); layout.append((a, "s")); continue
    if a in rank2k:
        lines.append(f"call jdc_k({a})"); layout.append((a, "k")); continue
    if a in rank2t:
        lines.append(f"call jdc_t({a})"); layout.append((a, "t")); continue
    if a.endswith("%cat_id") or a.endswith("%vegcls"):
        lines.append(f"call jdc_i1({a})"); layout.append((a, "v")); continue
    lines.append(f"call jdc_v1({a})"); layout.append((a, "v"))
def block(tag):
    s = f"     call jdc_begin(N_cat, {tag})\n"
    for l in lines:
        s += "     " + l + "\n"
    return s
# generated lines can exceed 132 columns: split at the commas is not possible for call args, so rely on -ffree-line-length-0
new = block(0) + call + "\n" + block(1)
src = src[:i] + new + src[j:]
src = src.replace("subroutine clsmf25_main(nid)\n", "subroutine clsmf25_main(nid)\n  use jaxdump_clsm_mod\n", 1)
open(f"{d}/clsmf25_main.F90", "w").write(src)
open(f"{d}/jaxdump_clsm.layout", "w").write("\n".join(f"{n}|{k}" for n, k in layout))
print("hooked", len(layout), "arguments")
