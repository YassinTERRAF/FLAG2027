"""Nested evaluation of learned evidence layers, retaining exact V5 contexts."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from flag2027.common import ROOT, device_for, load_pack, save_json, seed_all, sha256
from research_v2.train import development_pack
from research_v5.mapping import load_research_model
from research_v5.validate import context, media_hashes, metrics
from .evidence import prepare, to_device
from .episodes import fitting_bank
from .model import EvidenceAssociation, objective


def create(variant, cfg, device):
    return EvidenceAssociation(variant, cfg['hidden_dim'], cfg['dropout'], cfg['max_correction']).to(device)


@torch.inference_mode()
def predict(model, data):
    model.eval()
    return model(data)['score'].cpu().numpy()


def restore(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = create(ck['variant'], ck['config'], device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, ck['selection']


def fit(variant, bank, cfg, directory, device, seed, validation=None, epochs=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'best.pt').exists():
        return restore(directory/'best.pt', device)
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = create(variant, cfg, device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg['epochs'])
    max_epochs = cfg['epochs'] if epochs is None else int(epochs)
    best_epoch, best, best_state = 0, float('inf'), None
    if validation is not None:
        val_data, val_labels, val_tracks = validation
        initial = metrics(val_labels, predict(model, val_data), val_tracks)
        best = initial['mean_english_eer_percent']
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    started = time.time()
    for epoch in range(1, max_epochs+1):
        model.train()
        losses = []
        for _ in range(cfg['steps_per_epoch']):
            episode = bank[int(rng.integers(len(bank)))]
            data = {k: v.to(device) for k, v in episode['data'].items()}
            labels = episode['labels'].to(device)
            optim.zero_grad(set_to_none=True)
            output = model(data)
            support_mask = (torch.rand(data['keep'].shape[1], device=device) > cfg['support_dropout']).float()
            dropped = model(data, support_mask=support_mask)
            loss = objective(model, output, labels, data, cfg, dropped)
            if not loss.isfinite().item():
                raise RuntimeError('Non-finite evidence-network loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
            losses.append(loss.item())
        scheduler.step()
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'seconds': time.time()-started}
        if validation is not None:
            report = metrics(val_labels, predict(model, val_data), val_tracks)
            value = report['mean_english_eer_percent']
            row['inner_validation'] = report
            if value < best-1e-5:
                best, best_epoch = value, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            message = f'inner EER={value:.2f}% best={best:.2f}% epoch={best_epoch}'
        else:
            best_epoch = epoch
            message = 'fixed-epoch refit; outer labels unopened'
        history.append(row)
        save_json(directory/'history.json', history)
        print(f'{directory.name} {variant} epoch {epoch}/{max_epochs} loss={row["loss"]:.3f} {message}', flush=True)
        if validation is not None and epoch-best_epoch >= cfg['patience']:
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    selection = {'variant': variant, 'epoch': best_epoch, 'inner_eer_percent': best if validation is not None else None,
                 'parameters': sum(p.numel() for p in model.parameters()), 'checkpoint': str(directory/'best.pt')}
    torch.save({'variant': variant, 'config': cfg, 'state_dict': best_state, 'selection': selection,
                'role': 'learned_transductive_architecture_research'}, directory/'best.pt')
    return model, selection


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='research_v6/config.yaml')
    p.add_argument('--out', default='research_v6/runs/architecture')
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = device_for(cfg['device'])
    seed_all(cfg['seed'])
    pack, original = development_pack(cfg['pack'], cfg['original_split'])
    full = load_pack(cfg['pack'])
    hashes = media_hashes(full)
    keep = np.isin(full['labels'], sorted(set(original['train']) | set(original['validation'])))
    hashes = tuple(np.asarray(h)[keep] for h in hashes)
    prior_dir = Path(cfg['prior_validation'])
    previous = json.loads((prior_dir/'summary.json').read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    core = ['research_v6/train.py', 'research_v6/model.py', 'research_v6/evidence.py', 'research_v6/episodes.py',
            'research_v5/validate.py', 'research_v5/transductive.py', 'research_v5/mapping.py', 'src/flag2027/train.py', 'src/flag2027/model.py']
    spec = {'config': cfg, 'source_hashes': {f: sha256(ROOT/f) for f in core}, 'torch': str(torch.__version__),
            'pack_sha256': sha256(cfg['pack']), 'original_split_sha256': sha256(cfg['original_split']),
            'previous_summary_sha256': sha256(prior_dir/'summary.json'), 'preserved_zip_sha256': sha256(ROOT/'submissions.zip'),
            'original_manifest_sha256': sha256(ROOT/'artifacts/final/system.json')}
    spec['frozen_folds'] = {str(path): sha256(path) for fold in range(3)
        for path in [Path(cfg['fold_directory'])/f'fold{fold}'/name for name in ['split.json','inner_control_v1/best.pt','outer_control/best.pt']]}
    marker = out/'run_config.json'
    if marker.exists() and json.loads(marker.read_text()) != spec:
        raise ValueError('Experiment source/config/data changed; use a new run directory')
    save_json(marker, spec)
    if (out/'summary.json').exists():
        print('Verified completed architecture experiment.')
        return
    rows = []
    for fold in range(3):
        base = Path(cfg['fold_directory'])/f'fold{fold}'
        split = json.loads((base/'split.json').read_text())
        prior = previous['folds'][fold]['selected_config']
        folder = out/f'fold{fold}'
        if set(split['outer_validation']) & set(original['audit']):
            raise ValueError('Original audit must stay excluded')
        mapper, weight, fitted = load_research_model(base/'inner_control_v1/best.pt', device)
        if set(fitted) != set(split['inner_fit']):
            raise ValueError('Wrong inner mapper fitting identities')
        f, v, fi, vi, tracks, y, edge = context(pack, split['inner_validation'], 5027+fold, 16, hashes)
        val_data = to_device(prepare(f, v, fi, vi, tracks, edge, [(mapper, weight)], device, prior), device)
        bank = fitting_bank(pack, split['inner_fit'], hashes, prior, cfg, folder/'inner_bank', device, cfg['seed']+100*fold)
        selected = {}
        for variant in cfg['variants']:
            _, selected[variant] = fit(variant, bank, cfg, folder/f'inner_{variant}', device, cfg['seed']+fold,
                                      validation=(val_data, y, tracks))
        del bank
        outer_bank = fitting_bank(pack, split['outer_fit'], hashes, prior, cfg, folder/'outer_bank', device, cfg['seed']+100*fold+50)
        models = {}
        for variant in cfg['variants']:
            models[variant], _ = fit(variant, outer_bank, cfg, folder/f'outer_{variant}', device, cfg['seed']+fold,
                                    epochs=selected[variant]['epoch'])
        del outer_bank
        mapper, weight, fitted = load_research_model(base/'outer_control/best.pt', device)
        if set(fitted) != set(split['outer_fit']) or set(fitted) & set(split['outer_validation']):
            raise ValueError('Wrong outer mapper/identity overlap')
        f, v, fi, vi, tracks, y, edge = context(pack, split['outer_validation'], 5027+10_000+fold, 16, hashes)
        data = to_device(prepare(f, v, fi, vi, tracks, edge, [(mapper, weight)], device, prior), device)
        reference = create('hard_residual', cfg, device).eval()
        prior_scores = predict(reference, data)
        baseline = metrics(y, prior_scores, tracks)
        if abs(baseline['mean_english_eer_percent']-previous['folds'][fold]['transductive']['mean_english_eer_percent']) > 1e-5:
            raise ValueError('V6 zero-residual prior does not reproduce V5 comparison exactly')
        row = {'fold': fold, 'prior': baseline, 'inner_selection': selected, 'results': {}}
        arrays = {'labels': y, 'tracks': tracks, 'prior': prior_scores}
        for variant, model in models.items():
            scores = predict(model, data)
            row['results'][variant] = metrics(y, scores, tracks)
            arrays[variant] = scores
        rows.append(row)
        np.savez_compressed(folder/'outer_local_scores.npz', **arrays)
        save_json(out/'folds.json', rows)
        print('OUTER', json.dumps(row), flush=True)
    summary = {'role': 'learned_transductive_architecture_research', 'folds': rows, 'config': cfg,
               'run_config_sha256': sha256(marker), 'reserved_audit_excluded': original['audit'],
               'rules': 'User reports updated organizer rules allow transduction; archived earlier rules differ; updated text not provided',
               'novelty': 'Proposed task-specific combination; no claim of established priority or guaranteed improvement',
               'limitations': ['Reused English development identities', 'No Bengali reference labels or hidden evaluation EER',
                 'Cross-trial sampling and density dependence', 'Local mean is not a platform score']}
    for name in ['prior']+cfg['variants']:
        values = [(row['prior'] if name == 'prior' else row['results'][name])['mean_english_eer_percent'] for row in rows]
        summary[name] = {'mean_english_eer_percent': float(np.mean(values)),
                         'fold_sample_std_percent': float(np.std(values, ddof=1)), 'fold_means': values}
    assert sha256(ROOT/'submissions.zip') == spec['preserved_zip_sha256']
    save_json(out/'summary.json', summary)
    print('ARCHITECTURE SUMMARY', json.dumps({n: summary[n] for n in ['prior']+cfg['variants']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
