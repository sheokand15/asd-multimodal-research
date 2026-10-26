"""Download / verify Move4AS from the official figshare record (article 28296518, CC BY 4.0).

Behaviour
---------
1. Queries the figshare API for the official file list with MD5 checksums.
2. For every file, looks for a valid local copy, in this order:
     a. ``Dataset/Move4AS/<name>`` (already extracted)
     b. member ``<name>`` inside the figshare "download all" bundle ``Dataset/28296518.zip``
   A local copy is accepted only if its size AND MD5 match the API record.
3. Missing or corrupted files are downloaded individually with progress, bounded retries,
   timeouts and HTTP Range resume of ``*.part`` files.  Nothing existing is deleted; a
   corrupted extracted file is renamed to ``*.corrupt-<timestamp>`` only if --replace-corrupt.
4. A manifest is written to ``outputs/audit/move4as_download_manifest.json``.

Usage:
    .venv\\Scripts\\python scripts\\download_move4as.py            # verify, download only what is missing
    .venv\\Scripts\\python scripts\\download_move4as.py --verify-only

Manual recovery (if automatic download fails, e.g. proxy / quota):
    1. Open https://springernature.figshare.com/articles/dataset/Move4AS_A_Multimodal_Dataset_Addressing_Motor_Function_in_Autism/28296518
    2. Click "Download all" (≈5.3 GB) and save it as  E:\\ASD Project\\Dataset\\28296518.zip
       (or download single files into E:\\ASD Project\\Dataset\\Move4AS\\).
    3. Re-run this script with --verify-only, then scripts\\prepare_datasets.py --only move4as
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
DEST = DATASET_ROOT / "Move4AS"
BUNDLE = DATASET_ROOT / "28296518.zip"
MANIFEST = PROJECT_ROOT / "outputs" / "audit" / "move4as_download_manifest.json"
API = "https://api.figshare.com/v2/articles/28296518"
ALLOWED_HOSTS = ("api.figshare.com", "ndownloader.figshare.com", "figshare.com", "s3-eu-west-1.amazonaws.com",
                 "figshare-production", "amazonaws.com")
CHUNK = 1 << 22


def md5_stream(fh, size_hint: int | None = None, label: str = "") -> str:
    h = hashlib.md5()
    done = 0
    t = time.time()
    while True:
        b = fh.read(CHUNK)
        if not b:
            break
        h.update(b)
        done += len(b)
        if size_hint and size_hint > 200 * CHUNK and time.time() - t > 5:
            print(f"    md5 {label}: {done / size_hint:6.1%}", flush=True)
            t = time.time()
    return h.hexdigest()


def fetch_file_list(timeout: int, retries: int) -> list[dict]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(API, timeout=timeout)
            r.raise_for_status()
            return r.json()["files"]
        except requests.RequestException as e:
            print(f"  API attempt {attempt}/{retries} failed: {e}")
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("Could not reach the figshare API; see manual recovery instructions in this script's docstring.")


def check_local(f: dict) -> tuple[str, str | None]:
    """Return (status, source) where status in {valid, corrupt, missing}."""
    name, size, md5 = f["name"], f["size"], f["computed_md5"]
    p = DEST / name
    if p.exists():
        if p.stat().st_size == size:
            with open(p, "rb") as fh:
                if md5_stream(fh, size, name) == md5:
                    return "valid", str(p)
        return "corrupt", str(p)
    if BUNDLE.exists():
        try:
            with zipfile.ZipFile(BUNDLE) as z:
                info = z.getinfo(name)
                if info.file_size == size:
                    with z.open(info) as fh:
                        if md5_stream(fh, size, name) == md5:
                            return "valid", f"{BUNDLE}::{name}"
                return "corrupt", f"{BUNDLE}::{name}"
        except KeyError:
            pass
        except zipfile.BadZipFile as e:
            return "corrupt", f"{BUNDLE} ({e})"
    return "missing", None


def download(f: dict, timeout: int, retries: int) -> str:
    url, size, md5 = f["download_url"], f["size"], f["computed_md5"]
    DEST.mkdir(parents=True, exist_ok=True)
    final = DEST / f["name"]
    part = final.with_suffix(final.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
                host = requests.utils.urlparse(r.url).hostname or ""
                if not any(h in host for h in ALLOWED_HOSTS):
                    raise RuntimeError(f"Unexpected redirect host {host!r}")
                if r.status_code == 200 and have:
                    have = 0  # server ignored Range -> restart
                elif r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                mode = "ab" if have else "wb"
                done, t = have, time.time()
                with open(part, mode) as out:
                    for chunk in r.iter_content(CHUNK):
                        out.write(chunk)
                        done += len(chunk)
                        if time.time() - t > 5:
                            print(f"    {f['name']}: {done / size:6.1%} ({done / 1e6:.0f}/{size / 1e6:.0f} MB)", flush=True)
                            t = time.time()
            if part.stat().st_size != size:
                raise RuntimeError(f"size mismatch {part.stat().st_size} != {size}")
            with open(part, "rb") as fh:
                got = md5_stream(fh, size, f["name"])
            if got != md5:
                part.unlink()
                raise RuntimeError(f"MD5 mismatch {got} != {md5} (partial file discarded)")
            part.replace(final)
            return "downloaded"
        except Exception as e:  # bounded retry with backoff
            print(f"  download attempt {attempt}/{retries} for {f['name']} failed: {e}", flush=True)
            time.sleep(min(60, 2 ** attempt))
    return "failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--replace-corrupt", action="store_true")
    args = ap.parse_args()

    files = fetch_file_list(args.timeout, args.retries)
    total = sum(f["size"] for f in files)
    free = __import__("shutil").disk_usage(DATASET_ROOT).free
    print(f"Official record: {len(files)} files, {total / 1e9:.2f} GB; free disk {free / 1e9:.1f} GB")
    manifest = {"api": API, "license": "CC BY 4.0", "doi": "10.6084/m9.figshare.28296518.v1", "files": []}
    rc = 0
    for f in files:
        status, src = check_local(f)
        rec = {"name": f["name"], "size": f["size"], "md5": f["computed_md5"], "url": f["download_url"],
               "local_status": status, "local_source": src}
        print(f"  {f['name']:<40} {status:<8} {src or ''}")
        if status != "valid" and not args.verify_only:
            if status == "corrupt" and src and "::" not in src:
                if not args.replace_corrupt:
                    rec["action"] = "skipped (corrupt local file kept; pass --replace-corrupt)"
                    rc = 1
                    manifest["files"].append(rec)
                    continue
                Path(src).rename(f"{src}.corrupt-{int(time.time())}")
            if f["size"] > free * 0.9:
                rec["action"] = "skipped: insufficient disk space"
                rc = 1
            else:
                rec["action"] = download(f, args.timeout, args.retries)
                rc |= rec["action"] == "failed"
        elif status != "valid":
            rc = 1
        manifest["files"].append(rec)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Manifest: {MANIFEST}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
