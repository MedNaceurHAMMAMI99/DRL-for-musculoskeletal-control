"""Resolve every DOI in references.bib against Crossref and compare metadata.

Prints one line per entry: OK / MISMATCH / NOT FOUND, with the discrepancy.
A MISMATCH means the DOI resolves to a different paper than the .bib claims.
"""
import re, json, os, sys, time, urllib.request, urllib.error

BIB = r"C:\Users\moham\thesis_run\thesis_report\references.bib"
s = open(BIB, encoding="utf-8").read()

entries = []
for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", s, re.S):
    body = m.group(3)
    def f(name):
        # the last field of an entry has no trailing newline inside `body`,
        # so the terminator must be optional
        mm = re.search(name + r"\s*=\s*\{(.*?)\}\s*,?\s*(?:\n|$)", body, re.S)
        return " ".join(mm.group(1).split()) if mm else ""
    entries.append({
        "key": m.group(2).strip(), "doi": f("doi"),
        "title": f("title"), "author": f("author"), "year": f("year"),
    })

def norm(t):
    t = re.sub(r"[{}\\]", "", t or "").lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(t.split())

def firstauthor(a):
    a = re.sub(r"[{}\\]", "", a or "")
    return norm(a.split(" and ")[0].split(",")[0])

with_doi = [e for e in entries if e["doi"]]
print(f"{len(entries)} entries, {len(with_doi)} carry a DOI\n")

bad = []
for e in with_doi:
    url = "https://api.crossref.org/works/" + e["doi"]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "thesis-bib-check/1.0 (mailto:none@example.org)"})
        with urllib.request.urlopen(req, timeout=25) as r:
            msg = json.load(r)["message"]
    except urllib.error.HTTPError as ex:
        print(f"[NOT FOUND] {e['key']:20s} doi={e['doi']}  (HTTP {ex.code})")
        bad.append((e["key"], "doi does not resolve"))
        continue
    except Exception as ex:
        print(f"[ERROR    ] {e['key']:20s} {ex}")
        continue

    ct = (msg.get("title") or [""])[0]
    auths = msg.get("author") or []
    fa = norm(auths[0].get("family", "")) if auths else ""
    tb, tc = norm(e["title"]), norm(ct)
    tmatch = tb[:45] in tc or tc[:45] in tb
    amatch = (not fa) or (fa in norm(e["author"]))
    if tmatch and amatch:
        print(f"[OK       ] {e['key']:20s}")
    else:
        print(f"[MISMATCH ] {e['key']:20s} doi={e['doi']}")
        print(f"              bib says : {e['title'][:78]}")
        print(f"              doi is   : {ct[:78]}")
        print(f"              bib 1st author: {firstauthor(e['author'])}   doi 1st author: {fa}")
        bad.append((e["key"], f"resolves to '{ct[:60]}'"))
    time.sleep(0.12)

print("\n" + "=" * 62)
print("SUSPECT:", len(bad))
for k, why in bad:
    print(f"  {k:20s} {why}")
print("\nNo DOI (cannot be machine-checked):")
for e in entries:
    if not e["doi"]:
        print(f"  {e['key']:20s} {e['title'][:60]}")
