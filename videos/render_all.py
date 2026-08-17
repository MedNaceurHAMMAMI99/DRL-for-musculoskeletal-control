"""Render every presentation clip at 1080p.

    python render_all.py            # all scenes, 1920x1080 @ 60 fps
    python render_all.py --preview  # 480p draft, for checking layout fast
    python render_all.py S12 S18    # only scenes whose name contains these

Finished files land in  media/videos/<module>/1080p60/<Scene>.mp4
and are copied into  clips/  with a numeric prefix so they sort in
presentation order.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# Presentation order. Each entry is (module, scene, title).
SCENES = [
    ("v01_problem", "S01_TheQuestion",         "The question"),
    ("v01_problem", "S02_Redundancy",          "The counting problem"),
    ("v01_problem", "S03_Redundancy2",         "Infinitely many answers"),
    ("v01_problem", "S04_StaticOptimisation",  "The classical answer"),
    ("v01_problem", "S05_MomentArms",          "Force into torque"),
    ("v02_method",  "S06_TheArm",              "The model"),
    ("v02_method",  "S07_HillModel",           "How a muscle is modelled"),
    ("v02_method",  "S08_HowRLWorks",          "How RL works"),
    ("v02_method",  "S09_TheReward",           "The reward function"),
    ("v02_method",  "S10_TheExploit",          "The reward exploit"),
    ("v02_method",  "S11_ReplayRatio",         "The silent bug"),
    ("v03_finding", "S12_DiscountHorizon",     "The discount factor"),
    ("v03_finding", "S13_EntropyRefuted",      "A hypothesis refuted"),
    ("v03_finding", "S14_Ablation",            "Which parameter mattered"),
    ("v03_finding", "S15_AlgoBenchmark",       "The unfair benchmark"),
    ("v03_finding", "S16_GammaBeatsAlgorithm", "Setting beats algorithm"),
    ("v04_answer",  "S17_ForceMethod",         "A fair comparison"),
    ("v04_answer",  "S18_PatternAgrees",       "Per-muscle forces"),
    ("v04_answer",  "S19_PatternVsEconomy",    "Pattern versus economy"),
    ("v04_answer",  "S20_CoContraction",       "Co-contraction"),
    ("v04_answer",  "S21_SeedVariance",        "Five identical runs"),
    ("v04_answer",  "S22_EffortSweep",         "Closing the gap"),
    ("v04_answer",  "S23_VsConventional",      "Against a controller"),
    ("v04_answer",  "S25_SpeedWithdrawn",      "The withdrawn speed claim"),
    ("v04_answer",  "S24_Conclusion",          "The answer"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*", help="substring match on scene name")
    ap.add_argument("--preview", action="store_true", help="480p draft instead of 1080p")
    args = ap.parse_args()

    quality = "-ql" if args.preview else "-qh"      # -qh is 1920x1080 @ 60 fps
    subdir = "480p15" if args.preview else "1080p60"

    # keep the presentation index attached, so a filtered run still names
    # its output with the position the clip holds in the full sequence
    todo = [(n, s) for n, s in enumerate(SCENES, 1)
            if not args.filters or any(f.lower() in s[1].lower() for f in args.filters)]
    out = HERE / "clips"
    out.mkdir(exist_ok=True)

    t_all = time.time()
    failures = []
    for i, (idx, (mod, scene, title)) in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {scene} -- {title}", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "manim", quality, "--disable_caching",
             f"{mod}.py", scene],
            cwd=HERE, capture_output=True, text=True)
        src = HERE / "media" / "videos" / mod / subdir / f"{scene}.mp4"
        if r.returncode != 0 or not src.exists():
            failures.append(scene)
            print(f"    FAILED\n{r.stderr[-1500:]}", flush=True)
            continue
        dst = out / f"{idx:02d}_{scene}.mp4"
        shutil.copy2(src, dst)
        print(f"    {dst.name}  ({src.stat().st_size/1e6:.1f} MB, "
              f"{time.time()-t0:.0f}s)", flush=True)

    print(f"\nDone in {(time.time()-t_all)/60:.1f} min. "
          f"{len(todo)-len(failures)}/{len(todo)} rendered.", flush=True)
    if failures:
        print("FAILED:", ", ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
