# Verification tools

Three checks that answer questions a examiner can reasonably ask, by
measurement rather than assertion. Run them after any edit to the model, the
bibliography, or the method chapters.

```bash
python tools/check_references.py   # every DOI -> Crossref, compared to the .bib
python tools/check_citations.py    # every \cite key -> resolves against the .bib
python tools/describe_model.py     # the model as compiled: bodies, joints, muscles
```

## `check_references.py`

Resolves every DOI in `thesis_report/references.bib` against the Crossref API and
compares the returned title and first author with what the entry claims. Reports
`OK`, `MISMATCH` (the DOI belongs to a different paper) or `NOT FOUND`.

**This check earned its place.** It caught a fabricated entry: `Mrabet2015`,
cited in the introduction for a national stroke-incidence statistic, carried a
DOI that resolves to Teichmann, *"Neurologie et neurosciences dans les aphasies
primaires progressives"* — a different author, title and page. The entry and the
statistic it supported were removed.

Two known false positives, both harmless:

- Non-ASCII author names (`Kidziński`) fail the ASCII substring comparison. Check
  the printed title; if it matches, the entry is fine.
- 21 entries carry no DOI (books, arXiv preprints, older conference papers) and
  cannot be machine-checked. They are listed at the end of the run so the set
  needing manual attention is explicit rather than invisible.

## `check_citations.py`

Cross-checks every `\cite`-family key used in the `.tex` sources against the keys
defined in `references.bib`, in both directions. Undefined keys print as `???` in
the compiled PDF; unused entries are harmless but worth knowing about.

## `describe_model.py`

Prints the arm as MuJoCo actually compiles it: bodies and masses, joints with
their ranges, actuators with peak isometric force and operating length range, and
the muscle ordering and CCI groups from `config.py`.

**Also earned its place.** Chapter 3 had described a six-degree-of-freedom model
with a two-DOF wrist, segment masses totalling 4.04 kg, and a muscle table
listing a brachioradialis and a pectoralis major. The compiled model reports
`nq = nv = 4`, no wrist joint, 3.61 kg, and neither of those muscles. The chapter
now matches this output, and this script is how to keep it that way.
