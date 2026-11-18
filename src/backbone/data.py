"""Training-only ingestion and identity-disjoint validation protocols."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import ROOT, load_pack, save_json, sha256


def exactly_one(root, pattern):
    paths = sorted(p for p in Path(root).rglob(pattern) if "__MACOSX" not in p.parts)
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {root}, found {paths}")
    return paths[0]


def training_metadata(path):
    first = Path(path).read_text().splitlines()[0].split(",")
    headerless = len(first) == 2 and first[0].strip().startswith("id") and first[1].strip().lower() in ("m", "f", "male", "female")
    df = pd.read_csv(path, header=None, names=["id", "gender"]) if headerless else pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if c in ("id", "identity", "speaker_id", "speaker", "class", "class_id")), None)
    gender_col = next((c for c in df.columns if "gender" in c or c == "sex"), None)
    if not id_col or not gender_col:
        raise ValueError(f"Inspect metadata headers and add explicit mapping: {list(df.columns)}")
    values = {"male": 0, "m": 0, "female": 1, "f": 1, "0": 0, "1": 1}
    result = {}
    for _, row in df.iterrows():
        key = str(row[id_col]).strip()
        val = values.get(str(row[gender_col]).strip().lower())
        if val is None:
            raise ValueError(f"Unknown gender in training metadata: {row[gender_col]}")
        if key in result:
            raise ValueError(f"Duplicate training identity: {key}")
        result[key] = val
    return result


def prepare_official(out):
    root = ROOT / "data/train"
    face_csv = exactly_one(root, "train_English_faces.csv")
    voice_csv = exactly_one(root, "train_English_voices.csv")
    pair_list = exactly_one(root, "train_English.txt")
    face = pd.read_csv(face_csv, header=None).to_numpy(dtype=np.float32)
    voice = pd.read_csv(voice_csv, header=None).to_numpy(dtype=np.float32)
    if face.shape[1] != 4097 or voice.shape[1] != 193 or len(face) != len(voice):
        raise ValueError(f"Unexpected feature shapes {face.shape}, {voice.shape}")
    if not np.array_equal(face[:, -1], voice[:, -1]):
        raise ValueError("Face/voice class labels are not row-aligned")
    lines = [line.split() for line in pair_list.read_text().splitlines() if line.strip()]
    if len(lines) != len(face):
        raise ValueError("Training list and feature row counts differ")
    # Raw identities are taken from identity folders, never inferred from names/people.
    faces, voices, ids = [], [], []
    for fields in lines:
        audio = next((p for p in fields if p.lower().endswith(".wav")), None)
        image = next((p for p in fields if p.lower().endswith((".jpg", ".png", ".jpeg"))), None)
        if not audio or not image or "English" not in audio or "English" not in image:
            raise ValueError(f"Only explicitly English training records accepted: {fields}")
        identity = Path(audio).parent.name
        if identity != Path(image).parent.name:
            raise ValueError("Paired training paths have different identities")
        ids.append(identity)
        faces.append(str(resolve_media(pair_list.parent, image)))
        voices.append(str(resolve_media(pair_list.parent, audio)))
    ids = np.asarray(ids)
    for value in np.unique(face[:, -1]):
        if len(np.unique(ids[face[:, -1] == value])) != 1:
            raise ValueError("CSV class labels disagree with raw path identities")
    if len(np.unique(ids)) != len(np.unique(face[:, -1])):
        raise ValueError("CSV identity mapping is not one-to-one")
    meta = training_metadata(ROOT / "data/downloads/meta_file_train_set.csv")
    genders = np.asarray([meta[x] for x in ids], dtype=np.int64)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, faces=face[:, :-1], voices=voice[:, :-1], labels=ids,
                        genders=genders, face_paths=np.asarray(faces), voice_paths=np.asarray(voices))
    provenance = {"role": "official_training", "language": "English", "feature_set": "official_vggface_ecapa",
                  "rows": len(ids), "identities": len(np.unique(ids)), "pack_sha256": sha256(out),
                  "sources": {str(p.relative_to(ROOT)): sha256(p) for p in (face_csv, voice_csv, pair_list)}}
    save_json(out.with_suffix(".json"), provenance)
    print(json.dumps(provenance, indent=2))


def resolve_media(base, name):
    p = Path(name)
    if p.is_absolute():
        raise ValueError(f"Absolute paths in official list require explicit review: {name}")
    candidates = [Path(base) / p, Path(base) / p.name]
    # Some distributed lists include English_test while the base is already that folder.
    if len(p.parts) > 1:
        candidates.append(Path(base) / Path(*p.parts[1:]))
    for target in candidates:
        if target.is_file():
            return target.resolve()
    raise FileNotFoundError(f"Cannot resolve {name} relative to {base}")


def assert_training(pack_path):
    meta_path = Path(pack_path).with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    if meta.get("role") != "official_training" or meta.get("language") != "English":
        raise ValueError("Model development accepts official English training data only")
    if meta["pack_sha256"] != sha256(pack_path):
        raise ValueError("Training pack changed after provenance was recorded")
    return meta


def make_splits(pack, path, seed=2027):
    """60% fit, 20% tuning, 20% untouched audit; all disjoint by identity."""
    ids = np.unique(pack["labels"])
    sex = np.asarray([pack["genders"][np.flatnonzero(pack["labels"] == i)[0]] for i in ids])
    trainval, audit = train_test_split(ids, test_size=0.2, random_state=seed, stratify=sex)
    tv_sex = np.asarray([pack["genders"][np.flatnonzero(pack["labels"] == i)[0]] for i in trainval])
    train, val = train_test_split(trainval, test_size=0.25, random_state=seed + 1, stratify=tv_sex)
    split = {"seed": seed, "train": sorted(train.tolist()), "validation": sorted(val.tolist()),
             "audit": sorted(audit.tolist()), "labels_sha256": __import__("hashlib").sha256(pack["labels"].tobytes()).hexdigest()}
    save_json(path, split)
    return split


def trial_indices(pack, identities, seed=41, repeats=8):
    """Independent random within-identity positives and identity-balanced negatives.

    These trials are generated ONLY from labeled official English TRAINING identities.
    They must never be constructed for progress/evaluation identities.
    """
    rng = np.random.default_rng(seed)
    groups = {i: np.flatnonzero(pack["labels"] == i) for i in identities}
    genders = {i: int(pack["genders"][groups[i][0]]) for i in identities}
    results = {}
    for track in ("no_gender", "gender"):
        f, v, y = [], [], []
        for identity, ix in groups.items():
            neg_ids = [i for i in identities if i != identity and
                       (track == "no_gender" or genders[i] == genders[identity])]
            if not neg_ids:
                raise ValueError("Need at least two held-out identities per gender")
            for _ in range(repeats):
                for vi in ix:
                    positive = ix[ix != vi]
                    if not len(positive):
                        raise ValueError("Need >=2 samples per identity for unpaired validation")
                    f.extend([int(rng.choice(positive)), int(rng.choice(groups[rng.choice(neg_ids)]))])
                    v.extend([int(vi), int(vi)])
                    y.extend([1, 0])
        results[track] = (np.asarray(f), np.asarray(v), np.asarray(y))
    return results
