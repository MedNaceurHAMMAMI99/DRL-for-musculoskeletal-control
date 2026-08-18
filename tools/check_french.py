"""Find untranslated French anywhere in the thesis sources."""
import io, re, glob, os

os.chdir(r"C:\Users\moham\thesis_run\thesis_report")
MARKERS = [
    r"\b(?:le|la|les|des|une|dans|pour|avec|sur|est|sont|par|qui|que|cette|ce|ces|aux|du|au|et|ou|mais|donc|nous|vous|leur|son|sa|ses)\b",
]
# words that also exist in English or are proper nouns / LaTeX
ALLOW = {"la", "le", "des", "sur", "est", "son", "ce", "ces", "au", "du", "et", "ou",
         "par", "on", "a", "de", "en", "un", "une", "pour", "que", "qui"}

STRONG = re.compile(
    r"\b(?:l'\w+|d'\w+|qu'\w+|c'est|n'est|dans le|dans la|pour la|pour le|"
    r"avec le|avec la|sur le|sur la|est un|est une|sont des|par le|par la|"
    r"une fois|de la|du fichier|des \w+ial|taille d'|correction de|"
    r"\w+ation optimis\w+|recompense|echantillons|apparies|biais|accelere)\b",
    re.I)

ACCENTS = re.compile(r"[àâçéèêëîïôûùüÿœÀÂÇÉÈÊËÎÏÔÛÙÜŸŒ]")

for f in sorted(glob.glob("*.tex")):
    s = io.open(f, encoding="utf-8", errors="replace").read()
    hits = []
    for i, line in enumerate(s.split("\n"), 1):
        if line.strip().startswith("%"):
            continue
        m = STRONG.search(line)
        if m:
            hits.append((i, m.group(0), line.strip()[:88]))
        elif ACCENTS.search(line) and not re.search(r"\\'|\\`|\\^|\\c\{|\\\"", line):
            a = ACCENTS.search(line)
            hits.append((i, "accent:" + a.group(0), line.strip()[:88]))
    if hits:
        print(f"\n{f}")
        for i, what, line in hits[:12]:
            print(f"   L{i:<5d} [{what}]  {line}")
        if len(hits) > 12:
            print(f"   ... and {len(hits)-12} more")
