"""Official resources only; no automatic access to hidden labels."""
import json
import zipfile
from pathlib import Path

import gdown

from .common import ROOT, save_json, sha256

FILES = {
    "train": ("1zMhI7afifY7HiU6kHRg22e2YzNSk_Q3R", "train_set.zip", 1428601179),
    "dev": ("10_yp9IlmglEvwXRcKV3Fv9bTATej4N7g", "dev_set.zip", 989881860),
    "evaluation": ("1QQynBksW_5Vpavn2XN6NMyxhqPNlegTv", "evaluation_phase.zip", 472428761),
    "metadata": ("1TLaDoCW5Eh18uhPSu-BhEiGxbL6kNwWJ", "meta_file_train_set.csv", 560),
}


def download(which):
    dest = ROOT / "data/downloads"
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key in which:
        file_id, name, expected = FILES[key]
        target = dest / name
        if not target.exists() or target.stat().st_size != expected:
            tmp = target.with_suffix(target.suffix + ".part")
            result = gdown.download(id=file_id, output=str(tmp), quiet=False, resume=True)
            if result is None or tmp.stat().st_size != expected:
                raise RuntimeError(f"Incomplete {name}; expected {expected} bytes. Check official folder for changes.")
            tmp.replace(target)
        manifest[key] = {"file": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
                         "sha256": sha256(target), "url": f"https://drive.google.com/file/d/{file_id}/view"}
        save_json(manifest_path, manifest)
        if target.suffix == ".zip":
            out = ROOT / "data" / key
            marker = out / ".extracted.json"
            if marker.exists() and json.loads(marker.read_text())["sha256"] == manifest[key]["sha256"]:
                print(f"Already extracted {key}", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target) as archive:
                for info in archive.infolist():
                    resolved = (out / info.filename).resolve()
                    if not resolved.is_relative_to(out.resolve()):
                        raise ValueError("Unsafe archive path")
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("Symlinks are not accepted in data archives")
                archive.extractall(out)
            save_json(marker, {"sha256": manifest[key]["sha256"]})
            print(f"Extracted {key}: {out}", flush=True)
