import re, glob, os, sys

os.chdir(r"C:\Users\moham\thesis_run\thesis_report")

CITE = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")

used = {}
for f in sorted(glob.glob("*.tex")):
    s = open(f, encoding="utf-8", errors="replace").read()
    for m in CITE.finditer(s):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                used.setdefault(k, set()).add(f)

bib = open("references.bib", encoding="utf-8", errors="replace").read()
defined = {d.strip() for d in re.findall(r"^@\w+\{([^,]+),", bib, re.M)}

print(f"distinct keys cited : {len(used)}")
print(f"entries in .bib     : {len(defined)}")

missing = sorted(set(used) - defined)
unused = sorted(defined - set(used))
print(f"\nCITED BUT MISSING FROM BIB ({len(missing)}): {missing if missing else 'none'}")
print(f"\nIN BIB BUT NEVER CITED ({len(unused)}):")
for k in unused:
    print("   ", k)
