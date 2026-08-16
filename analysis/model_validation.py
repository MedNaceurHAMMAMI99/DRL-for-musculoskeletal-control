"""
Biomechanical model characterization (Phase-2 "next step", no training needed).

Produces the model's own force-length curves and muscle moment arms so they can be
overlaid on independent reference data (e.g. Holzbaur 2005 force-length, Murray 1995
elbow moment arms) for validation. This computes what is genuinely computable from the
model; it does NOT fabricate a reference comparison. Where reference values are
required, the output leaves an explicit slot for the user to fill from the literature.

Outputs runs/model_validation.json (+ a figure) with:
  - force_length : active (a=1.0, 0.5) and passive normalized force vs fibre length
  - moment_arms  : each muscle's moment arm (m) about the elbow across its range
                   of motion, from MuJoCo's actuator_moment (real, model-derived)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import mujoco
import config
from biomechanics.hill_model import f_CE, f_PEE


def force_length_curves(n=200):
    p = config.BICEPS_LONGUS
    l = np.linspace(0.5, 1.6, n)
    return {
        "l_norm":        l.tolist(),
        "active_a1.0":   [f_CE(x, 1.0, p["F_max"]) / p["F_max"] for x in l],
        "active_a0.5":   [f_CE(x, 0.5, p["F_max"]) / p["F_max"] for x in l],
        "passive":       [f_PEE(x, p["F_max"]) / p["F_max"] for x in l],
        "reference_source": "Compare against Holzbaur 2005 / Thelen 2003 (fill R^2 below)",
        "R2_vs_reference": None,   # <-- user fills after overlaying reference data
    }


def moment_arms(n=40):
    """Muscle moment arms (m) about the elbow across its range of motion."""
    model = mujoco.MjModel.from_xml_path(config.ARM_XML)
    data  = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flex")
    if jid < 0:
        return {"error": "elbow_flex joint not found"}
    qadr = model.jnt_qposadr[jid]
    lo, hi = model.jnt_range[jid] if model.jnt_limited[jid] else (0.0, 2.5)
    angles = np.linspace(lo, hi, n)

    arms = {m: [] for m in config.MUSCLE_NAMES}
    for a in angles:
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = a
        mujoco.mj_forward(model, data)
        R = data.actuator_moment.reshape(model.nu, model.nv)  # nu x nv
        for i, m in enumerate(config.MUSCLE_NAMES):
            arms[m].append(float(R[i, model.jnt_dofadr[jid]]))
    return {
        "elbow_angle_rad": angles.tolist(),
        "moment_arm_m":    arms,
        "reference_source": "Compare against Murray 1995 elbow moment arms (fill RMSE below)",
        "RMSE_vs_reference": None,   # <-- user fills after overlaying reference data
    }


def run():
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    out = {"force_length": force_length_curves(), "moment_arms": moment_arms()}
    path = os.path.join(config.RUNS_DIR, "model_validation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")

    # Optional figure (skipped silently if matplotlib backend unavailable).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fl = out["force_length"]; ma = out["moment_arms"]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(fl["l_norm"], fl["active_a1.0"], label="active a=1.0")
        ax[0].plot(fl["l_norm"], fl["active_a0.5"], label="active a=0.5")
        ax[0].plot(fl["l_norm"], fl["passive"], label="passive")
        ax[0].set(xlabel="normalized fibre length", ylabel="F / F_max",
                  title="Force-length (model)"); ax[0].legend()
        if "moment_arm_m" in ma:
            for m, v in ma["moment_arm_m"].items():
                ax[1].plot(ma["elbow_angle_rad"], v, label=m)
            ax[1].set(xlabel="elbow angle (rad)", ylabel="moment arm (m)",
                      title="Muscle moment arms (MuJoCo)")
            ax[1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        figpath = os.path.join(config.FIGURES_DIR, "model_validation.png")
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        fig.savefig(figpath, dpi=140); plt.close(fig)
        print(f"  wrote {figpath}")
    except Exception as e:
        print(f"  (figure skipped: {e})")
    return out


if __name__ == "__main__":
    run()
