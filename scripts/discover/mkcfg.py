import sys, re
src, dst, ey, em, ed, out = sys.argv[1:7]
keys = {"Ending year:": ey, "Ending month:": em, "Ending day:": ed, "Output directory:": '"%s"' % out, "Diagnostic output file:": '"%s/lislog"' % out}
lines = []
for l in open(src):
    for k, v in keys.items():
        if l.startswith(k):
            l = f"{k:40s}{v}\n"
    lines.append(l)
open(dst, "w").writelines(lines)
print("wrote", dst)
