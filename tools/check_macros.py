"""Find LaTeX macros whose leading backslash was lost (usually replaced by a tab).

These compile silently and print the macro name as literal text in the PDF.
"""
import io, glob, os, re
os.chdir(r"C:\Users\moham\thesis_run\thesis_report")

# macro bodies that, seen without a preceding backslash or letter, indicate a
# lost backslash: \textbf -> extbf, \emph -> mph, \begin -> egin, etc.
STUBS = ["extbf", "extit", "exttt", "mph", "egin", "nd\\{", "ref\\{", "cref\\{",
         "Cref\\{", "label\\{", "SI\\{", "item\\b", "section\\{"]
PAT = re.compile(r"(?<![\\A-Za-z])(" + "|".join(STUBS) + r")")

found = 0
for f in sorted(glob.glob("*.tex")):
    s = io.open(f, encoding="utf-8", errors="replace").read()
    for i, line in enumerate(s.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        for m in PAT.finditer(line):
            stub = m.group(1)
            # 'item' and 'section' appear as ordinary English words; require a brace
            if stub.startswith("item") or stub.startswith("section"):
                continue
            found += 1
            ctx = line.strip()[:100]
            tab = "TAB" if "\t" in line else "   "
            print(f"[{tab}] {f} L{i}: '{stub}'  ->  {ctx}")
print(f"\n{found} suspect macro(s)")
