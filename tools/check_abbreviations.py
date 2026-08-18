"""Check that every listed abbreviation is actually used in the thesis body,
and that abbreviations used in the body are listed."""
import io, re, glob, os
os.chdir(r"C:\Users\moham\thesis_run\thesis_report")

front = io.open("frontmatter.tex", encoding="utf-8").read()
# the abbreviation table rows
block = front[front.index("List of Abbreviations"):]
block = block[:block.index("\\end{tabularx}")]
listed = []
for line in block.split("\n"):
    m = re.match(r"^\s*([A-Za-z0-9\\{}]+)\s*&", line)
    if m:
        k = m.group(1).replace("\\MuJoCo", "MuJoCo").replace("\\RL{}", "RL").strip()
        if k and k not in ("\\textbf{Abbrev.}", "textbf{Abbrev.}"):
            listed.append(k)

body = ""
for f in sorted(glob.glob("*.tex")):
    if f in ("frontmatter.tex",) or f.startswith("guide"):
        continue
    s = io.open(f, encoding="utf-8", errors="replace").read()
    s = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", s, flags=re.S)
    s = re.sub(r"%[^\n]*", "", s)
    body += "\n" + s

print("LISTED BUT NEVER USED IN THE BODY:")
unused = []
for k in listed:
    pat = r"\b" + re.escape(k) + r"\b"
    if not re.search(pat, body):
        unused.append(k)
        print(f"   {k}")
if not unused:
    print("   none")

print("\nUSED IN THE BODY BUT NOT LISTED (common ones):")
for cand in ["CNS", "CCI", "VAF", "SLSQP", "NNLS", "TPE", "IQM", "API", "CPU",
             "XML", "QP", "MSE", "RMSE", "SD", "L4DC"]:
    if re.search(r"\b" + cand + r"\b", body) and cand not in listed:
        print(f"   {cand}")
