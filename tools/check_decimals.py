"""Find French decimal commas (0,81) in an English thesis.

Thousands separators in this document are written \\, (thin space), e.g.
233\\,000, so a bare comma between two digits is a decimal separator.
"""
import io, re, glob, os
os.chdir(r"C:\Users\moham\thesis_run\thesis_report")
PAT = re.compile(r"(?<=\d),(?=\d)")
total = 0
for f in sorted(glob.glob("*.tex")):
    s = io.open(f, encoding="utf-8", errors="replace").read()
    s = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", s, flags=re.S)
    hits = []
    for i, line in enumerate(s.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        for m in PAT.finditer(line):
            a = max(0, m.start() - 8)
            hits.append((i, line[a:m.end() + 8].strip()))
    if hits:
        print(f"\n{f}  ({len(hits)})")
        for i, ctx in hits[:14]:
            print(f"   L{i:<5d} ...{ctx}...")
        if len(hits) > 14:
            print(f"   ... and {len(hits)-14} more")
        total += len(hits)
print(f"\nTOTAL: {total}")
