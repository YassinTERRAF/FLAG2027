"""Inspect the frozen EGMA release without training or scoring new trials."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from .backbone.common import sha256
from .backbone.infer import validate_archive
from .layout import ROOT, release_path


def read_json(path):
    return json.loads(release_path(path).read_text())


def check_hash(path, expected):
    file = release_path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"Release file missing: {file.relative_to(ROOT)}. See docs/REVIEW.md for the required assets."
        )
    if sha256(file) != expected:
        raise ValueError(f"Frozen release checksum mismatch: {file.relative_to(ROOT)}")


def verify():
    release = read_json("docs/EGMA_RELEASE.json")
    final = Path("research_v6/runs/final")
    output = Path("research_v6/runs/submission")
    for path, expected in release["artifacts"].items():
        check_hash(path, expected)
    head = read_json(final / "manifest.json")
    check_hash(final / "best.pt", head["checkpoint_sha256"])
    check_hash(final / "final_config.json", head["final_config_sha256"])
    config = read_json(final / "final_config.json")
    mapper = read_json("artifacts/final/system.json")
    for sources in (config["source_hashes"], mapper["source_hashes"], mapper["pretrained_registries"]):
        for path, expected in sources.items():
            check_hash(path, expected)
    for member in mapper["members"]:
        check_hash(member["checkpoint"], member["sha256"])
    for registry in mapper["pretrained_registries"]:
        for path, expected in read_json(registry)["files"].items():
            check_hash(path, expected)
    submission = read_json(output / "manifest.json")
    check_hash("submissions_egma.zip", submission["archive_sha256"])
    archive_path = release_path("submissions_egma.zip")
    validate_archive(archive_path, ROOT / "data/evaluation")
    with ZipFile(archive_path) as archive:
        count = 0
        for cell, entry in submission["cells"].items():
            track, language = cell.split("/")
            condition = "heard" if language == "English" else "unheard"
            name = f"{track}/sub_score_v4_{language}_{condition}.txt"
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != entry["sha256"] or len(content.splitlines()) != entry["trials"]:
                raise ValueError(f"Frozen score file mismatch: {name}")
            count += entry["trials"]
        if len(archive.namelist()) != len(submission["cells"]) or archive.testzip() is not None:
            raise ValueError("Invalid frozen archive")
    print(f"Verified EGMA weights, source, and {count:,} archived scores.")
    print("Historical user-reported result: 3.41% EER.")
