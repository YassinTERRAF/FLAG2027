import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from .common import ROOT, device_for, load_pack, save_json, seed_all, sha256, project_path, source_hashes
from .data import assert_training, make_splits, trial_indices
from .metrics import verification_metrics
from .model import Association, joint_loss


def architecture(pack, cfg, n_classes):
    return {"face_dim": pack["faces"].shape[1], "voice_dim": pack["voices"].shape[1],
            "n_classes": n_classes, **{k: cfg[k] for k in ("embedding_dim", "hidden_dim", "dropout")}}


@torch.inference_mode()
def encode(model, faces, voices, device, batch_size=256):
    model.eval()
    chunks = [[], [], [], []]
    for i in range(0, len(faces), batch_size):
        f = torch.as_tensor(faces[i:i+batch_size], dtype=torch.float32, device=device)
        v = torch.as_tensor(voices[i:i+batch_size], dtype=torch.float32, device=device)
        out = model(f, v)
        for j, x in enumerate(out):
            if j >= 2:
                x = x.softmax(dim=-1)
            chunks[j].append(x.cpu().numpy())
    return [np.concatenate(x) for x in chunks]


def scores_for_trials(encoded, trials, gender_score_weight, ages=None, age_score_weight=0.0):
    f, v, gf, gv = encoded
    result = {}
    for track, (fi, vi, _) in trials.items():
        s = np.einsum("ij,ij->i", f[fi], v[vi])
        if track == "no_gender":
            # Fixed per-trial auxiliary prediction; never cohort scaling/calibration.
            s = s + gender_score_weight * np.einsum("ij,ij->i", gf[fi], gv[vi])
        if age_score_weight:
            if ages is None:
                raise ValueError("This fixed scorer requires native frozen age-head predictions")
            s = s - age_score_weight * np.abs(ages[0][fi] - ages[1][vi])
        result[track] = s
    return result


def evaluate_scores(trials, scores):
    results = {track: verification_metrics(trials[track][2], s) for track, s in scores.items()}
    results["mean_english_eer_percent"] = float(np.mean([v["eer_percent"] for v in results.values()]))
    return results


def train_one(pack, train_ids, val_ids, cfg, directory, seed, device, fixed_epochs=None):
    """No progress/evaluation data argument exists in the training API."""
    seed_all(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    classes = sorted(train_ids)
    groups = {i: np.flatnonzero(pack["labels"] == i) for i in classes}
    model_args = architecture(pack, cfg, len(classes))
    model = Association(**model_args).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    epochs = int(fixed_epochs or cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    trials = trial_indices(pack, val_ids) if val_ids else None
    val_indices = np.flatnonzero(np.isin(pack["labels"], val_ids)) if val_ids else None
    # Remap validation trial row references so only validation features are encoded.
    if trials:
        mapping = np.full(len(pack["labels"]), -1)
        mapping[val_indices] = np.arange(len(val_indices))
        local_trials = {k: (mapping[f], mapping[v], y) for k, (f, v, y) in trials.items()}
    best, best_epoch, best_state, best_weight, history = float("inf"), 0, None, 0.0, []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(cfg["steps_per_epoch"]):
            selected = rng.choice(len(classes), size=min(cfg["identities_per_batch"], len(classes)), replace=False)
            fi, vi, labels = [], [], []
            for c in selected:
                ix = groups[classes[c]]
                fi.extend(rng.choice(ix, cfg["samples_per_identity"], replace=True))
                vi.extend(rng.choice(ix, cfg["samples_per_identity"], replace=True))
                labels.extend([int(c)] * cfg["samples_per_identity"])
            f = torch.as_tensor(pack["faces"][fi], dtype=torch.float32, device=device)
            v = torch.as_tensor(pack["voices"][vi], dtype=torch.float32, device=device)
            lab = torch.as_tensor(labels, dtype=torch.long, device=device)
            sex = torch.as_tensor(pack["genders"][fi], dtype=torch.long, device=device)
            optim.zero_grad(set_to_none=True)
            loss = joint_loss(model, model(f, v), lab, sex, cfg)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            losses.append(loss.item())
        scheduler.step()
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "elapsed_seconds": time.time() - started}
        if trials:
            embeddings = encode(model, pack["faces"][val_indices], pack["voices"][val_indices], device)
            options = []
            for w in cfg["gender_score_weights"]:
                score = scores_for_trials(embeddings, local_trials, w)
                result = evaluate_scores(local_trials, score)
                options.append((result["mean_english_eer_percent"], float(w), result))
            metric, weight, result = min(options, key=lambda x: (x[0], x[1]))
            row.update({"validation": result, "gender_score_weight": weight})
            if metric < best - 1e-5:
                best, best_epoch, best_weight = metric, epoch, weight
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            message = f"val_mean_EER={metric:.2f}% best={best:.2f}%"
        else:
            best_epoch, best_weight = epoch, cfg["selected_gender_score_weight"]
            message = "fixed-epoch full-training refit"
        history.append(row)
        save_json(directory / "history.json", history)
        print(f"{directory.name} epoch {epoch}/{epochs} loss={row['loss']:.4f} {message}", flush=True)
        if trials and epoch - best_epoch >= cfg["patience"]:
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    checkpoint = {"state_dict": best_state, "model_args": model_args, "config": cfg, "seed": seed,
                  "epoch": best_epoch, "gender_score_weight": best_weight, "train_ids": classes,
                  "validation_ids": sorted(val_ids), "validation_mean_english_eer_percent": best if trials else None}
    torch.save(checkpoint, directory / "best.pt")
    return {"checkpoint": str((directory / "best.pt").resolve()), "epoch": best_epoch,
            "gender_score_weight": best_weight, "validation_mean_english_eer_percent": best if trials else None}


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") == "rcca":
        from .cca import RCCA
        model = RCCA(**checkpoint["model_args"])
    else:
        model = Association(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def validate_ensemble(pack, checkpoints, identities, device):
    trials = trial_indices(pack, identities)
    scores = {k: [] for k in trials}
    for path in checkpoints:
        model, ck = load_model(path, device)
        if set(identities) & set(ck["train_ids"]):
            raise ValueError("Validation identities overlap checkpoint training identities")
        embeddings = encode(model, pack["faces"], pack["voices"], device)
        ages = (pack["face_age"], pack["voice_age"]) if "face_age" in pack else None
        for k, s in scores_for_trials(embeddings, trials, ck["gender_score_weight"], ages, ck.get("age_score_weight",0.0)).items():
            scores[k].append(s)
    return evaluate_scores(trials, {k: np.mean(s, axis=0) for k, s in scores.items()})


def search(pack_path, config_path, output, device_name=None):
    provenance = assert_training(pack_path)
    pack = load_pack(pack_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise ValueError("This search is already selected; use a new output directory for a new experiment")
    cfg = yaml.safe_load(Path(config_path).read_text())
    device = device_for(device_name or cfg["device"])
    split = make_splits(pack, output / "split.json", cfg["seed"])
    results = []
    for candidate in cfg["candidates"]:
        cc = {**cfg, **candidate}
        r = train_one(pack, split["train"], split["validation"], cc, output / candidate["name"], cfg["seed"], device)
        results.append({"candidate": candidate, **r})
        save_json(output / "candidates.json", results)
    best = min(results, key=lambda r: r["validation_mean_english_eer_percent"])
    selected_cfg = {**cfg, **best["candidate"]}
    members = [best]
    for seed in cfg["ensemble_seeds"]:
        r = train_one(pack, split["train"], split["validation"], selected_cfg,
                      output / f"ensemble_seed{seed}", seed, device)
        members.append(r)
    paths = [m["checkpoint"] for m in members]
    val_report = validate_ensemble(pack, paths, split["validation"], device)
    selection = {"pack": str(Path(pack_path).resolve()), "pack_sha256": sha256(pack_path),
                 "feature_set": provenance["feature_set"], "config": selected_cfg, "members": members,
                 "validation": val_report, "split": split,
                 "warning": "English training-identity holdout only; no Bengali/general leaderboard EER estimate."}
    # Freeze system selection BEFORE looking at the audit split.
    save_json(output / "selection.json", selection)
    print(json.dumps(selection["validation"], indent=2), flush=True)


def audit(selection_path, device_name="mps"):
    selection_path = Path(selection_path)
    report_path = selection_path.parent / "audit.json"
    if report_path.exists():
        raise ValueError("Audit already run. Read the existing report; do not tune against the audit split.")
    selection = json.loads(selection_path.read_text())
    if sha256(selection["pack"]) != selection["pack_sha256"]:
        raise ValueError("Pack hash changed")
    pack = load_pack(selection["pack"])
    report = validate_ensemble(pack, [m["checkpoint"] for m in selection["members"]],
                               selection["split"]["audit"], device_for(device_name))
    report.update({"selection_sha256": sha256(selection_path), "role": "one_time_english_identity_holdout",
                   "limitations": "No Bengali labels. Trials reuse identities and recordings; trial count is not independent sample count."})
    save_json(report_path, report)
    print(json.dumps(report, indent=2))


def finalize(selection_path, output, device_name="mps"):
    selection_path, output = Path(selection_path), Path(output)
    if (output / "system.json").exists():
        raise ValueError("A frozen system already exists here. Keep it immutable; use a new directory for a deliberate new version.")
    selected = json.loads(selection_path.read_text())
    audit_path = selection_path.parent / "audit.json"
    if not audit_path.exists() or json.loads(audit_path.read_text())["selection_sha256"] != sha256(selection_path):
        raise ValueError("Run the frozen system audit before full-training refit")
    assert_training(selected["pack"])
    pack = load_pack(selected["pack"])
    cfg = selected["config"]
    epochs = int(np.median([m["epoch"] for m in selected["members"]]))
    # Fixed rule defined before evaluation: retain members' chosen auxiliary weights.
    members = []
    for member in selected["members"]:
        _, ck = load_model(member["checkpoint"], torch.device("cpu"))
        cc = {**cfg, "selected_gender_score_weight": member["gender_score_weight"]}
        if cfg.get("method") == "rcca":
            from .cca import save_rcca
            r = save_rcca(pack, np.unique(pack["labels"]).tolist(), cfg, output / "rcca/best.pt",
                          member["gender_score_weight"])
        else:
            r = train_one(pack, np.unique(pack["labels"]).tolist(), [], cc,
                          output / f"seed{ck['seed']}", ck["seed"], device_for(device_name), fixed_epochs=epochs)
        if member.get("age_score_weight", 0.0):
            final_checkpoint = torch.load(r["checkpoint"], map_location="cpu", weights_only=True)
            final_checkpoint["age_score_weight"] = member["age_score_weight"]
            torch.save(final_checkpoint, r["checkpoint"])
            r["age_score_weight"] = member["age_score_weight"]
        r["sha256"] = sha256(r["checkpoint"])
        r["checkpoint"] = project_path(r["checkpoint"])
        members.append(r)
    manifest = {"status": "frozen", "feature_set": selected["feature_set"], "pack_sha256": selected["pack_sha256"],
                "selection_sha256": sha256(selection_path), "audit_sha256": sha256(audit_path), "members": members,
                "score_direction": "distance_lower_is_match", "ensemble": "arithmetic_mean_of_per_trial_scores",
                "training_language": "English", "refit_epochs": epochs, "source_hashes": source_hashes(),
                "pretrained_registries": {}}
    if selected["feature_set"].startswith("official_plus_facenet_wavlm"):
        manifest["pretrained_registries"]["models/registry.json"] = sha256(ROOT / "models/registry.json")
    if selected["feature_set"] == "official_plus_facenet_wavlm_age":
        manifest["pretrained_registries"]["models/demographics_registry.json"] = sha256(ROOT / "models/demographics_registry.json")
    if (ROOT / "requirements-lock.txt").exists():
        manifest["requirements_lock_sha256"] = sha256(ROOT / "requirements-lock.txt")
    save_json(output / "system.json", manifest)
    print(f"Frozen system: {output / 'system.json'}")
