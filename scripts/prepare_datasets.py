"""Safely extract the raw dataset archives found in the dataset root.

* Original archives are opened read-only and never moved, renamed or deleted.
* Every archive is CRC-tested before extraction (``--skip-crc`` to skip).
* Member paths are checked against path traversal / absolute paths.
* Extraction is idempotent: files that already exist with the expected size are skipped.
* A JSON manifest (archive -> member list, sizes, CRC status) is written to outputs/audit.

Usage (from the project root):
    .venv\\Scripts\\python scripts\\prepare_datasets.py [--only move4as mmasd questionnaire] [--skip-crc]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

MOVE4AS_ZIP = "28296518.zip"
MMASD_PARTS = [
    "drive-download-20261023T064244Z-1-002.zip",
    "drive-download-20261023T064244Z-1-003.zip",
    "drive-download-20261023T064244Z-1-004.zip",
]
QUESTIONNAIRE_ZIP = "autism-screening-for-toddlers-data.zip"


def _safe_target(dest: Path, member: str) -> Path:
    target = (dest / member).resolve()
    if not str(target).startswith(str(dest.resolve())):
        raise ValueError(f"Unsafe archive member path: {member!r}")
    return target


def crc_test(zf: zipfile.ZipFile, label: str) -> str | None:
    t = time.time()
    bad = zf.testzip()
    print(f"  CRC test {label}: {'OK' if bad is None else 'FAILED at ' + bad} ({time.time() - t:.0f}s)", flush=True)
    return bad


def extract(zf: zipfile.ZipFile, dest: Path, members=None, strip_prefix: str = "") -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    infos = [zf.getinfo(m) for m in members] if members else zf.infolist()
    for info in infos:
        name = info.filename
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix):]
        if not name or name.endswith("/"):
            continue
        target = _safe_target(dest, name)
        rec = {"member": info.filename, "path": str(target), "size": info.file_size, "crc": f"{info.CRC:08x}"}
        if target.exists() and target.stat().st_size == info.file_size:
            rec["status"] = "exists"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            with zf.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
            tmp.replace(target)
            rec["status"] = "extracted"
        out.append(rec)
    return out


def prepare_move4as(skip_crc: bool) -> dict:
    src = DATASET_ROOT / MOVE4AS_ZIP
    dest = DATASET_ROOT / "Move4AS"
    man = {"archive": str(src), "archive_size": src.stat().st_size, "dest": str(dest)}
    with zipfile.ZipFile(src) as outer:
        man["outer_crc_failed_member"] = None if skip_crc else crc_test(outer, MOVE4AS_ZIP)
        top = [i.filename for i in outer.infolist() if i.filename != "Dataset.zip"]
        man["outer_files"] = extract(outer, dest, top)
        # Nested Dataset.zip is STORED (uncompressed) so it can be read in place without a temp copy.
        with outer.open("Dataset.zip") as inner_fh, zipfile.ZipFile(inner_fh) as inner:
            man["inner_crc_failed_member"] = None if skip_crc else crc_test(inner, "Dataset.zip")
            man["inner_files"] = extract(inner, dest / "Dataset")
        for nested in ("Preprocess_3DMotion_rawfiles.zip",):
            with zipfile.ZipFile(dest / nested) as z:
                man.setdefault("nested", {})[nested] = extract(z, dest / Path(nested).stem)
    return man


def prepare_mmasd(skip_crc: bool) -> dict:
    dest = DATASET_ROOT / "MMASD"
    man = {"dest": str(dest), "parts": {}}
    for part in MMASD_PARTS:
        src = DATASET_ROOT / part
        rec = {"archive_size": src.stat().st_size}
        with zipfile.ZipFile(src) as z:
            rec["crc_failed_member"] = None if skip_crc else crc_test(z, part)
            rec["files"] = extract(z, dest / "archives")
        man["parts"][part] = rec
    # Inner zips (2D_openpose, ROMP_2D, ROMP_3D, ROMP_3D_71) -> MMASD/<modality>/
    man["inner"] = {}
    for inner in sorted((dest / "archives").rglob("*.zip")):
        with zipfile.ZipFile(inner) as z:
            bad = None if skip_crc else crc_test(z, inner.name)
            files = extract(z, dest / inner.stem)
        man["inner"][inner.name] = {"crc_failed_member": bad, "n_files": len(files),
                                    "total_bytes": sum(f["size"] for f in files)}
    for x in (dest / "archives").rglob("*.xlsx"):
        shutil.copy2(x, dest / x.name)
    for x in (dest / "archives").rglob("README.txt"):
        shutil.copy2(x, dest / "README_drive.txt")
    return man


def prepare_questionnaire(skip_crc: bool) -> dict:
    src = DATASET_ROOT / QUESTIONNAIRE_ZIP
    dest = DATASET_ROOT / "Questionnaire"
    with zipfile.ZipFile(src) as z:
        bad = None if skip_crc else crc_test(z, QUESTIONNAIRE_ZIP)
        files = extract(z, dest)
    return {"archive": str(src), "dest": str(dest), "crc_failed_member": bad, "files": files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["questionnaire", "mmasd", "move4as"])
    ap.add_argument("--skip-crc", action="store_true")
    args = ap.parse_args()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    fns = {"questionnaire": prepare_questionnaire, "mmasd": prepare_mmasd, "move4as": prepare_move4as}
    rc = 0
    for name in args.only:
        print(f"[{name}] preparing ...", flush=True)
        try:
            man = fns[name](args.skip_crc)
            man["status"] = "ok"
        except Exception as e:  # report and continue with the other datasets
            man = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            rc = 1
        with open(AUDIT_DIR / f"extract_manifest_{name}.json", "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1)
        print(f"[{name}] {man['status']}", man.get("error", ""), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
