"""Frozen system inference and exact CodaBench packaging, without test-set statistics."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .common import ROOT, device_for, save_json, sha256, unit, resolve_project_path
from .data import exactly_one, resolve_media
from .metrics import submission_distance
from .train import encode, load_model

CELLS = [(track, lang, condition) for track in ("no_gender", "gender")
         for lang, condition in (("English", "heard"), ("Bangla", "unheard"))]


def parse_pairs(path):
    rows = []
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        # Official evaluation gender lists contain "faces 00000.jpg" (a missing
        # slash). This syntax-only repair preserves the exact key and row order.
        if len(fields) == 4 and fields[2].endswith("/faces") and "/" not in fields[3] and fields[3].endswith(".jpg"):
            fields = [fields[0], fields[1], fields[2] + "/" + fields[3]]
        if len(fields) != 3 or fields[0] in seen:
            raise ValueError(f"Expected unique <key> <audio> <face>: {line}")
        key, audio, face = fields
        if not audio.endswith(".wav") or not face.lower().endswith((".jpg", ".png", ".jpeg")):
            raise ValueError(f"Unexpected trial format: {line}")
        seen.add(key)
        rows.append((key, audio, face))
    if not rows:
        raise ValueError("Empty pair file")
    return rows


def cell_files(root, track, lang):
    root = Path(root)
    lists = [p for p in root.rglob(f"{lang}*.txt") if track in p.parts and "__MACOSX" not in p.parts
             and "score" not in p.name.lower()]
    if len(lists) != 1:
        raise ValueError(f"Need one explicit {track}/{lang} pair list, got {lists}")
    pair_path = lists[0]
    # Resolve features by the track path, permitting features/faces/<track> layouts.
    found = {}
    for modality in ("faces", "voices"):
        paths = [p for p in root.rglob(f"{lang}*_{modality}.csv") if track in p.parts and "__MACOSX" not in p.parts]
        if len(paths) != 1:
            raise ValueError(f"Need one {track}/{lang} {modality} feature file, got {paths}")
        found[modality] = paths[0]
    return pair_path, found


def predict_pair(models, faces, voices, track, device, ages=None):
    scores = []
    for model, checkpoint in models:
        f, v, gf, gv = encode(model, faces, voices, device)
        score = np.einsum("ij,ij->i", f, v)
        if track == "no_gender":
            score += checkpoint["gender_score_weight"] * np.einsum("ij,ij->i", gf, gv)
        if checkpoint.get("age_score_weight", 0.0):
            if ages is None:
                raise ValueError("Frozen age scorer is missing its auxiliary predictions")
            score -= checkpoint["age_score_weight"] * np.abs(ages[0] - ages[1])
        scores.append(score)
    return np.mean(scores, axis=0)


def infer(system_path, data_root, out, device_name="mps"):
    system_path, out = Path(system_path), Path(out)
    system = json.loads(system_path.read_text())
    if system.get("status") != "frozen" or system.get("score_direction") != "distance_lower_is_match":
        raise ValueError("Requires a frozen system with validated export direction")
    for name, expected in {**system.get("source_hashes", {}), **system.get("pretrained_registries", {})}.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen system resource changed after validation: {name}")
    if "requirements_lock_sha256" in system and sha256(ROOT / "requirements-lock.txt") != system["requirements_lock_sha256"]:
        raise ValueError("Dependency lock changed after the system was frozen")
    device = device_for(device_name)
    models = []
    for member in system["members"]:
        path = resolve_project_path(member["checkpoint"])
        if sha256(path) != member["sha256"]:
            raise ValueError("Checkpoint changed after freeze")
        models.append(load_model(path, device))
    extra = system["feature_set"].startswith("official_plus_facenet_wavlm")
    age = system["feature_set"] == "official_plus_facenet_wavlm_age"
    needs_age_term = any(ck.get("age_score_weight", 0.0) for _, ck in models)
    encoders = None
    demographics = None
    if extra:
        from .encoders import FrozenEncoders
        encoders = FrozenEncoders(device_name)
    if age:
        from .demographics import FrozenDemographics
        demographics = FrozenDemographics(device_name)
    out.mkdir(parents=True, exist_ok=True)
    record = {"system_sha256": sha256(system_path), "score_direction": "distance_lower_is_match", "cells": {}}
    for track, lang, condition in CELLS:
        pair_path, features = cell_files(data_root, track, lang)
        pairs = parse_pairs(pair_path)
        faces = pd.read_csv(features["faces"], header=None).to_numpy(dtype=np.float32)
        voices = pd.read_csv(features["voices"], header=None).to_numpy(dtype=np.float32)
        if faces.shape != (len(pairs), 4096) or voices.shape != (len(pairs), 192):
            raise ValueError("Pair/feature count or dimensions do not match exactly")
        if not np.isfinite(faces).all() or not np.isfinite(voices).all():
            raise ValueError("Non-finite test features")
        # Batched MLP evaluation is independent per row: no batch norm, cross attention,
        # clustering, neighbor lookup, score statistics, or comparisons between trials.
        if extra:
            scores = []
            for i, (_, audio, face) in enumerate(tqdm(pairs, desc=f"{track}/{lang}")):
                fp = resolve_media(pair_path.parent, face)
                vp = resolve_media(pair_path.parent, audio)
                f = np.concatenate([unit(faces[i:i+1]), unit(encoders.face_embedding(fp)[None])], axis=1)
                v = np.concatenate([unit(voices[i:i+1]), unit(encoders.voice_embedding(vp)[None])], axis=1)
                if age:
                    fh, vh = demographics.face_embeddings([fp]), demographics.voice_embeddings([vp])
                    f = np.concatenate([f, unit(fh)], axis=1)
                    v = np.concatenate([v, unit(vh)], axis=1)
                    ages = demographics.age_predictions(fh, vh) if needs_age_term else None
                else:
                    ages = None
                scores.append(float(predict_pair(models, f, v, track, device, ages)[0]))
            scores = np.asarray(scores)
        else:
            scores = predict_pair(models, faces, voices, track, device)
        distances = submission_distance(scores)
        dest = out / track / f"sub_score_v4_{lang}_{condition}.txt"
        dest.parent.mkdir(exist_ok=True)
        dest.write_text("".join(f"{key} {float(s):.10f}\n" for (key, _, _), s in zip(pairs, distances)))
        record["cells"][f"{track}/{lang}"] = {"pairs": len(pairs), "pair_list_sha256": sha256(pair_path),
            "face_features_sha256": sha256(features["faces"]), "voice_features_sha256": sha256(features["voices"]),
            "score_file_sha256": sha256(dest)}
    archive = out / "submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for track, lang, condition in CELLS:
            name = f"{track}/sub_score_v4_{lang}_{condition}.txt"
            z.write(out / name, name)
    validate_archive(archive, data_root)
    record["submission_sha256"] = sha256(archive)
    save_json(out / "inference_manifest.json", record)
    print(f"Prepared {archive}. No upload was performed.")


def validate_archive(archive, data_root):
    expected = {f"{t}/sub_score_v4_{l}_{c}.txt" for t, l, c in CELLS}
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != expected or len(z.namelist()) != 4:
            raise ValueError("Submission must contain exactly four files at the specified paths")
        for track, lang, condition in CELLS:
            name = f"{track}/sub_score_v4_{lang}_{condition}.txt"
            pair_path, _ = cell_files(data_root, track, lang)
            keys = [row[0] for row in parse_pairs(pair_path)]
            lines = [line.split() for line in z.read(name).decode().splitlines()]
            if any(len(row) != 2 for row in lines) or [row[0] for row in lines] != keys:
                raise ValueError(f"Invalid key coverage/order in {name}")
            if not np.isfinite([float(row[1]) for row in lines]).all():
                raise ValueError(f"Non-finite scores in {name}")
    return True
