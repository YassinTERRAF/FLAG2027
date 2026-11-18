"""Cross-fit training episodes: their identities were not used by their mapper."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from flag2027.common import ROOT, save_json, sha256
from flag2027.train import train_one, load_model
from research_v5.validate import context
from .evidence import prepare


def sex_for(pack, ids):
    return np.asarray([pack['genders'][np.flatnonzero(pack['labels'] == i)[0]] for i in ids])


def fitting_bank(pack, ids, fingerprints, prior, cfg, directory, device, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'bank_spec.json'
    spec = {'identities': sorted(ids), 'seed': seed, 'prior': prior, 'episodes': cfg['episode_bank'],
            'episode_anchors': cfg['episode_anchors'], 'crossfit_fixed_epochs': 10,
            'source_hashes': {p: sha256(ROOT/p) for p in ['research_v6/episodes.py', 'research_v6/evidence.py']}}
    if marker.exists() and json.loads(marker.read_text()) != spec:
        raise ValueError('Episode bank specification changed')
    save_json(marker, spec)
    if (directory/'bank.pt').exists():
        return torch.load(directory/'bank.pt', map_location='cpu', weights_only=True)
    ids = np.asarray(sorted(ids))
    sexes = sex_for(pack, ids)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    bank = []
    rng = np.random.default_rng(seed)
    mapper_cfg = {'embedding_dim': 128, 'hidden_dim': 256, 'dropout': .6,
        'learning_rate': .001, 'weight_decay': .01, 'epochs': 10, 'steps_per_epoch': 30,
        'identities_per_batch': 12, 'samples_per_identity': 4, 'temperature': .07,
        'proxy_weight': .5, 'gender_weight': .15, 'selected_gender_score_weight': .25}
    for fold, (fit, unseen) in enumerate(cv.split(ids, sexes)):
        checkpoint = directory/f'crossfit{fold}/best.pt'
        if not checkpoint.exists():
            train_one(pack, ids[fit].tolist(), [], mapper_cfg, checkpoint.parent, seed+fold,
                      device, fixed_epochs=10)
        mapper, ck = load_model(checkpoint, device)
        if set(ck['train_ids']) != set(ids[fit]) or set(ck['train_ids']) & set(ids[unseen]):
            raise ValueError('Cross-fit mapper identity contamination')
        groups = {g: ids[unseen][sexes[unseen] == g] for g in [0, 1]}
        if min(map(len, groups.values())) < 2:
            raise ValueError('Each unseen cross-fit pool needs two identities per gender')
        count = cfg['episode_bank']//3+(fold < cfg['episode_bank'] % 3)
        for episode in range(count):
            chosen = np.concatenate([rng.choice(groups[g], min(len(groups[g]), cfg['episode_identities']//2), replace=False)
                                     for g in [0, 1]]).tolist()
            anchors = cfg['episode_anchors'][len(bank) % len(cfg['episode_anchors'])]
            f, v, fi, vi, tracks, labels, edge_features = context(pack, chosen, seed+1000+len(bank), anchors, fingerprints)
            inputs = prepare(f, v, fi, vi, tracks, edge_features, [(mapper, .25)], device, prior)
            bank.append({'data': {k: torch.as_tensor(v) for k, v in inputs.items()},
                         'labels': torch.as_tensor(labels, dtype=torch.float32), 'unseen_episode_identities': chosen,
                         'mapper_training_identities': ck['train_ids'], 'mapper_sha256': sha256(checkpoint)})
            print(f'Cross-fit episode {len(bank)}/{cfg["episode_bank"]}; unseen IDs={len(chosen)} anchors={anchors}', flush=True)
    torch.save(bank, directory/'bank.pt')
    return bank
