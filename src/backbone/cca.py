"""Small-data baseline: identity-balanced regularized canonical correlation.

All matrices and centers are learned from English training identities. Inference
contains only fixed projections of the individual trial, with no reference cohort.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

from .common import load_pack, save_json, sha256, unit
from .data import assert_training, make_splits, trial_indices
from .train import evaluate_scores, scores_for_trials


class RCCA(nn.Module):
    def __init__(self, face_dim, voice_dim, embedding_dim):
        super().__init__()
        for name, shape in {"wf": (face_dim, embedding_dim), "wv": (voice_dim, embedding_dim),
                            "mf": (face_dim,), "mv": (voice_dim,),
                            "gf": (face_dim,), "gv": (voice_dim,), "bf": (1,), "bv": (1,)}.items():
            self.register_buffer(name, torch.zeros(shape))

    def forward(self, faces, voices):
        f, v = F.normalize(faces, dim=-1), F.normalize(voices, dim=-1)
        ef = F.normalize((f - self.mf) @ self.wf, dim=-1)
        ev = F.normalize((v - self.mv) @ self.wv, dim=-1)
        lf, lv = f @ self.gf + self.bf, v @ self.gv + self.bv
        return ef, ev, torch.stack([torch.zeros_like(lf), lf], -1), torch.stack([torch.zeros_like(lv), lv], -1)


def fit_rcca(pack, train_ids, cfg):
    f, v = unit(pack["faces"]), unit(pack["voices"])
    ids = pack["labels"]
    means = [np.stack([x[ids == identity].mean(axis=0) for identity in train_ids]) for x in (f, v)]
    mf, mv = [x.mean(axis=0) if cfg["center"] else np.zeros(x.shape[1], dtype=np.float32) for x in means]
    ux, sx, vx = np.linalg.svd(means[0] - mf, full_matrices=False)
    uy, sy, vy = np.linalg.svd(means[1] - mv, full_matrices=False)
    ridge = cfg["ridge"]
    ax, ay = sx / np.sqrt(sx**2 + ridge), sy / np.sqrt(sy**2 + ridge)
    a, s, b = np.linalg.svd(ax[:, None] * (ux.T @ uy) * ay[None, :], full_matrices=False)
    dim = min(cfg["embedding_dim"], len(s))
    wf = (vx.T / np.sqrt(sx**2 + ridge)) @ a[:, :dim] * s[:dim]**cfg["power"]
    wv = (vy.T / np.sqrt(sy**2 + ridge)) @ b.T[:, :dim] * s[:dim]**cfg["power"]
    ix = np.isin(ids, train_ids)
    gender = []
    for x in (f, v):
        classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=2027).fit(x[ix], pack["genders"][ix])
        gender.append((classifier.coef_[0].astype(np.float32), classifier.intercept_.astype(np.float32)))
    args = {"face_dim": f.shape[1], "voice_dim": v.shape[1], "embedding_dim": dim}
    model = RCCA(**args)
    state = {"wf": wf, "wv": wv, "mf": mf, "mv": mv,
             "gf": gender[0][0], "bf": gender[0][1], "gv": gender[1][0], "bv": gender[1][1]}
    model.load_state_dict({k: torch.tensor(value, dtype=torch.float32) for k, value in state.items()})
    return model.eval(), args


def save_rcca(pack, train_ids, cfg, path, gender_score_weight, validation_ids=(), result=None):
    model, args = fit_rcca(pack, train_ids, cfg)
    checkpoint = {"kind": "rcca", "model_args": args, "state_dict": model.state_dict(), "config": cfg,
                  "gender_score_weight": gender_score_weight, "train_ids": sorted(train_ids),
                  "validation_ids": sorted(validation_ids), "epoch": 1, "seed": 2027,
                  "validation_mean_english_eer_percent": result}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return {"checkpoint": str(path.resolve()), "epoch": 1, "gender_score_weight": gender_score_weight,
            "validation_mean_english_eer_percent": result}


def cca_search(pack_path, output):
    from .train import encode
    meta = assert_training(pack_path)
    pack = load_pack(pack_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise ValueError("Existing selected run; use another directory")
    split = make_splits(pack, output / "split.json")
    trials = trial_indices(pack, split["validation"])
    results, best = [], None
    for center, ridge, dim, power in itertools.product([False, True], [.001, .01, .1, 1.0, 10.0], [4, 8, 16, 32], [0, 1]):
        cfg = {"method": "rcca", "center": center, "ridge": ridge, "embedding_dim": dim, "power": power}
        model, _ = fit_rcca(pack, split["train"], cfg)
        encoded = encode(model, pack["faces"], pack["voices"], "cpu")
        for weight in [0.0, 0.15, 0.3]:
            result = evaluate_scores(trials, scores_for_trials(encoded, trials, weight))
            row = {"config": cfg, "gender_score_weight": weight, "validation": result}
            results.append(row)
            if best is None or result["mean_english_eer_percent"] < best["validation"]["mean_english_eer_percent"]:
                best = row
                print(f"RCCA best {result['mean_english_eer_percent']:.2f}%: {cfg}, gender weight={weight}", flush=True)
    save_json(output / "candidates.json", results)
    member = save_rcca(pack, split["train"], best["config"], output / "best.pt", best["gender_score_weight"],
                       split["validation"], best["validation"]["mean_english_eer_percent"])
    save_json(output / "selection.json", {"pack": str(Path(pack_path).resolve()), "pack_sha256": sha256(pack_path),
        "feature_set": meta["feature_set"], "config": best["config"], "members": [member],
        "validation": best["validation"], "split": split,
        "warning": "Local English training holdout only; four-cell challenge EER cannot be inferred."})
    print(json.dumps(best, indent=2))
