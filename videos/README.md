# Presentation clips

Twenty-four short animations covering the thesis end to end, at
**1920×1080, 60 fps**. Each clip is self-contained: it states one idea, shows
the measurement behind it, and ends on the sentence you want the room to keep.

```
python render_all.py              # all clips, 1080p60  ->  clips/
python render_all.py --preview    # 480p draft, ~20x faster, for layout checks
python render_all.py S12 S18      # just the clips whose name matches
```

Output lands in `clips/` numbered in presentation order, so the folder plays
straight through in sequence.

---

## The sequence

### Part 1 — the problem (why the thesis exists)

| # | Clip | What it shows |
|---|------|---------------|
| 01 | `S01_TheQuestion` | Title, and the research question stated exactly |
| 02 | `S02_Redundancy` | Nine ropes on one door: three completely different pulls, one identical door angle |
| 03 | `S03_Redundancy2` | 9 unknowns, 4 equations — and the minimum-effort rule that picks one answer |
| 04 | `S04_StaticOptimisation` | The classical formula assembled term by term, in words |
| 05 | `S05_MomentArms` | `τ = Rᵀf` — the equation the whole thesis turns on |

### Part 2 — the model and the method

| # | Clip | What it shows |
|---|------|---------------|
| 06 | `S06_TheArm` | Four degrees of freedom, nine muscles, and why shoulder rotation had to be restrained |
| 07 | `S07_HillModel` | Hill's three elements; why "how much force?" has no single answer |
| 08 | `S08_HowRLWorks` | The agent–environment loop, and that a reward is a score, not an instruction |
| 09 | `S09_TheReward` | All seven reward terms, each with the failure that put it there |
| 10 | `S10_TheExploit` | The agent destroying the arm on 100 % of episodes, because quitting scored 20 higher |
| 11 | `S11_ReplayRatio` | The replay-ratio bug: 0.233 updates per step instead of 0.933 |

### Part 3 — the key finding

| # | Clip | What it shows |
|---|------|---------------|
| 12 | `S12_DiscountHorizon` | γ sets a planning horizon; the default was 4× longer than the movement |
| 13 | `S13_EntropyRefuted` | A well-argued hypothesis, and the two-sided ablation that killed it |
| 14 | `S14_Ablation` | Five parameters tested one at a time. Four do nothing. One explains everything |
| 15 | `S15_AlgoBenchmark` | The benchmark that was unfair, re-run — and the ranking changes |
| 16 | `S16_GammaBeatsAlgorithm` | One setting mattered ~3× more than the choice of algorithm |

### Part 4 — the answer

| # | Clip | What it shows |
|---|------|---------------|
| 17 | `S17_ForceMethod` | How both methods are made to face an identical problem |
| 18 | `S18_PatternAgrees` | Per-muscle forces, learned vs classical, all nine muscles |
| 19 | `S19_PatternVsEconomy` | The pattern agrees; the economy does not — and the null the cosine needed |
| 20 | `S20_CoContraction` | Antagonists cancelling: where the wasted effort actually goes |
| 21 | `S21_SeedVariance` | Five identical runs: 4 % spread in outcome, 77 % spread in muscle energy |
| 22 | `S22_EffortSweep` | Raising the effort weight closes the gap — and the seed spread collapses |
| 23 | `S23_VsConventional` | The honest trade-off, and the claim that was withdrawn |
| 24 | `S25_SpeedWithdrawn` | The speed argument, measured properly and abandoned |
| 25 | `S24_Conclusion` | The answer, with a confidence rating on every claim |

---

## The colour system

Colours here are not a style choice; they were checked with a validator
(OKLab ΔE, Machado–Oliveira–Fernandes colour-vision simulation at full
severity) against the dark surface. The point is that nobody in the room —
including the roughly 1 in 12 men with a colour vision deficiency — should
have to work out which bar is which.

| Role | Colour | |
|---|---|---|
| learned / DRL | `#009CE0` blue | the only two colours that ever encode |
| classical / reference | `#CC7C00` amber | competing quantities |
| third series, rare | `#DE4E9E` pink | validated against both of the above |
| ordered data (a sweep) | `#0084C4 → #00A8F8 → #84CCFC` | one hue, ordering carried by lightness |
| verdict: good / bad | `#00B048` / `#FC403C` | reserved — never a data series |
| surface / text / grid | `#121316` / `#EEF1F5` / `#3A4048` | |

Measured separations for the pair that carries almost every chart:

```
blue <-> amber     ΔE 25.1 simulated CVD     ΔE 29.2 normal vision
worst all-pairs    ΔE 11.6 simulated CVD     ΔE 21.6 normal vision
```

Rules that follow from this, and are kept everywhere:

- **Two series at a time.** Violet was tested as a third categorical colour and
  rejected — it collapses against blue under deuteranopia (ΔE 4.9). Green was
  tested and rejected against amber for the same reason (ΔE 4.9).
- **Ordered data gets a single-hue ramp**, never three different hues. The
  `w₂ = 1 / 5 / 20` sweep reads dark-to-light, so the ordering survives even in
  greyscale or on a bad projector.
- **Nothing depends on hue alone.** Every series is directly labelled and every
  legend swatch sits beside its written name.
- **Red and amber never sit next to each other**, and green is never used as a
  series colour beside amber. Status colours always travel with a word.
- **Text stays in text colours** (`INK`/`MUTED`), never in a series colour, so a
  coloured mark always means data.

## Layout rules

- Everything sits inside a safe box (`±6.55 × ±3.55` of a `14.22 × 8` frame), so
  nothing is lost to projector overscan or a cropped slide.
- Body text never drops below 21 pt at 1080p — readable from the back of a room.
- Every slide has the same three anchors: title, rule, and an optional
  takeaway strip at the bottom, so the eye does not have to re-find the layout
  on each cut.
- Bars have rounded ends anchored flat to their baseline, with a visible gap
  between neighbours so adjacent fills never touch.

## Files

| File | |
|---|---|
| `v_theme.py` | palette, `Slide` base class, bar/table/legend helpers |
| `v01_problem.py` | clips 01–05 |
| `v02_method.py` | clips 06–11 |
| `v03_finding.py` | clips 12–16 |
| `v04_answer.py` | clips 17–25 |
| `render_all.py` | renders everything, in order, into `clips/` |

Every number spoken in these clips comes from the thesis and the technical
guide in `../thesis_report/`. Nothing here is illustrative-only.
