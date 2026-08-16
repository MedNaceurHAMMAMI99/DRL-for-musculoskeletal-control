"""
Mark orphaned Optuna trials as FAIL.

A trial is left in state RUNNING if its worker dies without unwinding — which
has now happened twice: once when four workers raced on schema creation
(2026-08-14 15:14) and once when the machine hard-locked under the load of
those same four workers (2026-08-14 ~15:20, Kernel-Power 41, no bugcheck).

Orphans are not harmless. `study.optimize` counts RUNNING trials toward
`n_trials`, and the TPE sampler treats their (absent) results as pending, so a
restarted search silently does less work than asked and samples as though those
points were still in flight.

Nothing here touches COMPLETE or PRUNED trials, so real results can never be
destroyed by running it.

Run:  python cleanup_optuna.py            (report only)
      python cleanup_optuna.py --apply    (actually fail the orphans)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import optuna
import config

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="SAC")
    p.add_argument("--apply", action="store_true",
                   help="without this, only report what would change")
    a = p.parse_args()

    db = os.path.join(config.RUNS_DIR, f"optuna_{a.algo.upper()}.db")
    if not os.path.exists(db):
        print(f"No study database at {db} — nothing to clean.")
        return

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db}", engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.load_study(study_name=f"{a.algo.upper()}_arm_reach",
                              storage=storage)

    states = {}
    for t in study.trials:
        states.setdefault(t.state.name, []).append(t.number)

    print(f"Study: {a.algo.upper()}_arm_reach   ({len(study.trials)} trials)")
    for name, nums in sorted(states.items()):
        print(f"  {name:<9} {len(nums):>3}   {nums}")

    running = [t for t in study.trials
               if t.state == optuna.trial.TrialState.RUNNING]
    if not running:
        print("\nNo orphaned RUNNING trials. Nothing to do.")
    elif not a.apply:
        print(f"\n{len(running)} orphaned RUNNING trial(s) would be marked FAIL: "
              f"{[t.number for t in running]}")
        print("Re-run with --apply to do it.")
    else:
        for t in running:
            storage.set_trial_state_values(t._trial_id,
                                           optuna.trial.TrialState.FAIL)
            print(f"  trial {t.number}: RUNNING -> FAIL")
        print(f"\nMarked {len(running)} orphan(s) FAIL.")

    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nUsable results in this study: {len(complete)} completed trial(s).")


if __name__ == "__main__":
    main()
