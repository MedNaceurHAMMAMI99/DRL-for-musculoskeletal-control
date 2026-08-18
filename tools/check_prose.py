"""Mechanical prose checks across the thesis sources."""
import io, re, glob, os, collections

D = r"C:\Users\moham\thesis_run\thesis_report"
files = [f for f in sorted(glob.glob(os.path.join(D, "*.tex")))
         if not os.path.basename(f).startswith("guide")]

# US/UK spelling consistency — the thesis should pick one and hold it
PAIRS = [("optimize", "optimise"), ("optimization", "optimisation"),
         ("normalize", "normalise"), ("normalization", "normalisation"),
         ("minimize", "minimise"), ("minimization", "minimisation"),
         ("maximize", "maximise"), ("analyze", "analyse"),
         ("modeling", "modelling"), ("modeled", "modelled"),
         ("behavior", "behaviour"), ("labeled", "labelled"),
         ("center", "centre"), ("fiber", "fibre"),
         ("generalization", "generalisation"), ("parameterized", "parameterised")]

counts = collections.Counter()
where = collections.defaultdict(set)
text = {}
for f in files:
    s = io.open(f, encoding="utf-8", errors="replace").read()
    text[f] = s
    low = s.lower()
    for us, uk in PAIRS:
        for w in (us, uk):
            n = len(re.findall(r"\b" + w + r"\b", low))
            if n:
                counts[w] += n
                where[w].add(os.path.basename(f))

print("=" * 68)
print("SPELLING VARIANT SPLITS  (both forms present in the thesis)")
print("=" * 68)
split = 0
for us, uk in PAIRS:
    if counts[us] and counts[uk]:
        split += 1
        print(f"  {us:16s} {counts[us]:3d}   vs   {uk:16s} {counts[uk]:3d}")
        print(f"      {us}: {', '.join(sorted(where[us]))}")
        print(f"      {uk}: {', '.join(sorted(where[uk]))}")
if not split:
    print("  none")

print()
print("=" * 68)
print("OTHER MECHANICAL CHECKS")
print("=" * 68)
for f in files:
    s = text[f]
    base = os.path.basename(f)
    body = re.sub(r"%.*", "", s)          # strip comments
    issues = []
    # double words
    for m in re.finditer(r"\b(\w{3,})\s+\1\b", body, re.I):
        issues.append(f"repeated word '{m.group(1)}'")
    # straight quotes where TeX wants ``...''
    n = len(re.findall(r'(?<![=\\])"', body))
    if n:
        issues.append(f"{n} straight double-quote(s)")
    # spaces before punctuation
    n = len(re.findall(r"\s+[,;.](?=\s)", body))
    if n:
        issues.append(f"{n} space-before-punctuation")
    # TODO markers
    for kw in ("TODO", "FIXME", "XXX", "\\todo"):
        if kw in body:
            issues.append(f"marker {kw}")
    if issues:
        print(f"  {base}: " + "; ".join(issues))
