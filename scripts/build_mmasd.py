"""Build MMASD processed tensors: ROMP 3D (SMPL-24) and OpenPose 2D (BODY_25) for the same tracked person.

Outputs (data/processed/mmasd/):
  X3d.npy (N, T, 24, 3), V3d.npy valid mask;  X2d.npy (N, T, 25, 2), V2d.npy (zeros where no OpenPose)
  F3d.npy kinematic features; index.csv (clip, activity, participant, frames, persons, tracking QC)
Usage: .venv\\Scripts\\python scripts\\build_mmasd.py [--workers 6]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import mmasd as D  # noqa: E402
from src.utils.config import load_config, save_json  # noqa: E402


def process_clip_cached(args):
    """Worker entry: compute one clip, write arrays to the cache, return metadata only (small IPC payload)."""
    *a, cache = args
    out = Path(cache) / f"{a[1]}.npz"
    if out.exists():
        with np.load(out, allow_pickle=True) as z:
            return z["rec"].item()
    rec, Z3, V3, Z2, V2, F3 = process_clip(tuple(a))
    arrs = {k: v for k, v in (("Z3", Z3), ("V3", V3), ("Z2", Z2), ("V2", V2), ("F3", F3)) if v is not None}
    np.savez(out, rec=np.array(rec, dtype=object), **arrs)
    return rec


def process_clip(args):
    act, clip, romp_dir, op_dir, T = args
    rec = {"activity": act, "clip": clip, **D.parse_clip(clip)}
    try:
        rf = D.load_romp_clip(Path(romp_dir))
        rec["n_frames"] = len(rf)
        rec["persons_mean"] = float(np.mean([len(p) for p in rf])) if rf else 0
        rec["persons_max"] = int(max((len(p) for p in rf), default=0))
        tr = D.track(rf)
        pt = D.primary_track(rf, tr)
        rec["n_tracks"] = len(tr)
        rec["track_coverage_3d"] = len(pt) / max(len(rf), 1)
        X3 = D.track_to_array(rf, pt, 24, 3)
        Z3, V3 = D.resample_and_normalise(X3, T, "romp3d")
        Z2 = V2 = None
        if op_dir and Path(op_dir).exists():
            of = D.load_openpose_clip(Path(op_dir))
            rec["n_frames_openpose"] = len(of)
            if len(of) == len(rf):
                otr = D.track(of, conf_idx=2)
                ref = D.track_centroids(rf, pt)
                mt, dist = D.matching_track(of, otr, ref, conf_idx=2)
                rec["openpose_match_dist_px"] = float(dist) if np.isfinite(dist) else None
                rec["track_coverage_2d"] = len(mt) / max(len(of), 1)
                if mt:
                    X2 = D.track_to_array(of, mt, 25, 2, conf_idx=2)
                    rec["openpose_missing_kp_frac"] = float(np.isnan(X2[..., 0]).mean())
                    Z2, V2 = D.resample_and_normalise(X2, T, "openpose")
        rec["ok3d"] = Z3 is not None
        rec["ok2d"] = Z2 is not None
        F3 = D.kinematic_features(Z3, V3) if Z3 is not None else None
        return rec, Z3, V3, Z2, V2, F3
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["ok3d"] = rec["ok2d"] = False
        return rec, None, None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["paths"]["mmasd"]
    T = cfg["experiment"]["max_sequence_length"]
    out = Path(cfg["paths"]["processed"]) / "mmasd"
    out.mkdir(parents=True, exist_ok=True)
    romp = root / "ROMP_3D_Coordinates_71-joints"
    op = root / "2D_openpose" / "output"
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    jobs = [(a.name, c.name, str(c), str(op / a.name / c.name) if (op / a.name / c.name).exists() else None, T,
             str(cache)) for a in sorted(romp.iterdir()) if a.is_dir() for c in sorted(a.iterdir()) if c.is_dir()]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        recs = list(ex.map(process_clip_cached, jobs, chunksize=1))
    print(f"processed {len(recs)} clips in {time.time() - t0:.0f}s")
    keep = [r for r in recs if r["ok3d"]]
    N = len(keep)
    X3 = np.zeros((N, T, 24, 3), np.float32)
    V3 = np.zeros((N, T, 24, 3), bool)
    X2 = np.zeros((N, T, 25, 2), np.float32)
    V2 = np.zeros((N, T, 25, 2), bool)
    F3 = []
    for i, r in enumerate(keep):
        with np.load(cache / f"{r['clip']}.npz", allow_pickle=True) as z:
            X3[i], V3[i] = z["Z3"], z["V3"]
            F3.append(z["F3"])
            if "Z2" in z:
                X2[i], V2[i] = z["Z2"], z["V2"]
    np.save(out / "X3d.npy", X3)
    np.save(out / "V3d.npy", V3)
    np.save(out / "X2d.npy", X2)
    np.save(out / "V2d.npy", V2)
    np.save(out / "F3d.npy", np.stack(F3))
    idx = pd.DataFrame(keep)
    idx.to_csv(out / "index.csv", index=False)
    pd.DataFrame(recs).to_csv(out / "clips_all.csv", index=False)
    save_json({"edges_3d": D.SMPL_EDGES, "edges_2d": D.BODY25_EDGES, "T": T,
               "note": "3D = ROMP SMPL-24 joints (pixel-scaled); 2D = OpenPose BODY_25 of the matched track"},
              out / "meta.json")
    print(f"kept {N} clips with valid 3D; with valid 2D: {int(idx.ok2d.sum())}")


if __name__ == "__main__":
    main()
