"""Fixed-epoch candidate refit; no evaluation data is accepted here."""
import json
from pathlib import Path
import numpy as np
from flag2027.common import ROOT, device_for, load_pack, save_json, sha256
from research_v2.train import development_pack
from research_v5.validate import media_hashes
from .episodes import fitting_bank
from .train import fit


def main():
    # Architecture/loss choice uses this explicitly exploratory development CV.
    # Its winning score must not be described as a fresh independent test result.
    candidates = []
    for name in ['architecture_long','architecture_no_rank']:
        folder = ROOT/'research_v6/runs'/name
        summary = json.loads((folder/'summary.json').read_text())
        candidates.append((summary['multiscale_attention']['mean_english_eer_percent'], folder))
    _, experiment = min(candidates, key=lambda pair: pair[0])
    data = json.loads((experiment/'summary.json').read_text())
    cfg = data['config']
    spec = json.loads((experiment/'run_config.json').read_text())
    for name, expected in spec['source_hashes'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Source changed after local validation')
    # Refit the selected soft-attention architecture/loss variant.
    # Epoch is the median of inner-selected epochs; only variant choice uses CV.
    epochs = int(np.median([r['inner_selection']['multiscale_attention']['epoch'] for r in data['folds']]))
    pack, original = development_pack(cfg['pack'], cfg['original_split'])
    full = load_pack(cfg['pack'])
    hashes = media_hashes(full)
    keep = np.isin(full['labels'], sorted(set(original['train']) | set(original['validation'])))
    hashes = tuple(np.asarray(h)[keep] for h in hashes)
    prior = json.loads((ROOT/cfg['prior_validation']/'selected.json').read_text())['selected_config']
    ids = sorted(np.unique(pack['labels']).tolist())
    if set(ids) & set(original['audit']):
        raise ValueError('Reserved audit identities must not enter the new head refit')
    out = ROOT/'research_v6/runs/final'
    out.mkdir(parents=True, exist_ok=True)
    final_spec = {'config': cfg, 'epochs': epochs, 'epoch_rule': 'median inner-selected epochs',
                  'selected_experiment': str(experiment.relative_to(ROOT)),
                  'architecture_selection_rule': 'lowest observed development-fold mean among prespecified full-attention loss variants',
                  'score_caveat': 'best-observed development comparison; not a fresh independent test estimate',
                  'training_identities': ids, 'excluded_audit': original['audit'], 'prior': prior,
                  'experiment_sha256': sha256(experiment/'summary.json'),
                  'source_hashes': {**spec['source_hashes'], **{p: sha256(ROOT/p) for p in ['research_v6/final.py','research_v6/infer.py']}},
                  'pack_sha256': sha256(cfg['pack']), 'preserved_zip_sha256': sha256(ROOT/'submissions.zip')}
    marker = out/'final_config.json'
    if marker.exists() and json.loads(marker.read_text()) != final_spec:
        raise ValueError('Final refit specification changed; use a new version')
    save_json(marker, final_spec)
    device = device_for(cfg['device'])
    bank = fitting_bank(pack, ids, hashes, prior, cfg, out/'bank', device, cfg['seed']+1000)
    _, selection = fit('multiscale_attention', bank, cfg, out, device, cfg['seed']+1000, epochs=epochs)
    save_json(out/'manifest.json', {'role': 'fixed_epoch_architecture_candidate', 'checkpoint': str(out/'best.pt'),
        'checkpoint_sha256': sha256(out/'best.pt'), 'final_config_sha256': sha256(marker),
        'training_identities': len(ids), 'epochs': epochs, 'selection': selection,
        'local_development_mean_english_eer_percent': data['multiscale_attention']['mean_english_eer_percent'],
        'score_role': 'best_observed_development_ablation',
        'hidden_evaluation_eer': None, 'rules_status': 'Updated transduction permission reported by user; updated text not archived'})
    if sha256(ROOT/'submissions.zip') != final_spec['preserved_zip_sha256']:
        raise ValueError('Original user ZIP changed')
    print('Saved fixed-epoch EGMA candidate:', out/'best.pt', flush=True)


if __name__ == '__main__':
    main()
