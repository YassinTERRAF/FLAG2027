"""Select transductive settings on English inner holdouts; assess outer holdouts."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import yaml
from flag2027.common import ROOT, device_for, save_json, sha256
from flag2027.metrics import verification_metrics
from research_v2.train import development_pack, balanced_trials
from .mapping import load_research_model, map_features, per_trial_scores
from .transductive import cluster, score

CORE = ['research_v5/validate.py', 'research_v5/transductive.py', 'research_v5/mapping.py',
        'research_v5/features.py', 'research_v5/infer.py']


def metrics(labels, scores, tracks):
    result = {}
    for track in ['no_gender', 'gender']:
        mask = tracks == track
        result[track] = verification_metrics(labels[mask], scores[mask])
    result['mean_english_eer_percent'] = float(np.mean([v['eer_percent'] for v in result.values()]))
    return result


def media_hashes(pack):
    cache = ROOT/'research_v5/runs/training_content_hashes.json'
    spec = {'pack_sha256': sha256(ROOT/'artifacts/demographic.npz')}
    if cache.exists():
        data = json.loads(cache.read_text())
        if data['spec'] != spec:
            raise ValueError('Training fingerprint source changed')
        return data['face'], data['voice']
    face = [sha256(p) for p in pack['face_paths']]
    voice = [sha256(p) for p in pack['voice_paths']]
    save_json(cache, {'spec': spec, 'face': face, 'voice': voice})
    return face, voice


def context(pack, ids, seed, anchors, fingerprints):
    trials = balanced_trials(pack, ids, seed, anchors=anchors)
    frows = np.concatenate([t[0] for t in trials.values()])
    vrows = np.concatenate([t[1] for t in trials.values()])
    labels = np.concatenate([t[2] for t in trials.values()])
    tracks = np.concatenate([np.repeat(track, len(t[0])) for track, t in trials.items()])
    _, first_f, fi = np.unique(np.asarray(fingerprints[0])[frows], return_index=True, return_inverse=True)
    _, first_v, vi = np.unique(np.asarray(fingerprints[1])[vrows], return_index=True, return_inverse=True)
    faces = pack['faces'][frows[first_f]]
    voices = pack['voices'][vrows[first_v]]
    return faces, voices, fi, vi, tracks, labels, (pack['faces'][frows], pack['voices'][vrows])


def candidates(cfg):
    return [{'face_threshold': f, 'voice_threshold': v, 'pool_weight': p, 'graph_weight': g,
             'linkage': cfg['linkage'], 'graph_pseudocount': cfg['graph_pseudocount']}
            for f, v, p, g in itertools.product(cfg['face_thresholds'], cfg['voice_thresholds'],
                                               cfg['pool_weights'], cfg['graph_weights'])]


def assess(mapped, faces, voices, fi, vi, tracks, labels, candidate, groups=None, independent_scores=None):
    if groups is None:
        groups = (cluster(faces[:, 4096:4608], candidate['face_threshold'], candidate['linkage']),
                  cluster(voices[:, 192:704], candidate['voice_threshold'], candidate['linkage']))
    fg, vg = groups
    scores = score(mapped, fg, vg, fi, vi, tracks == 'no_gender', candidate['pool_weight'],
                   candidate['graph_weight'], candidate['graph_pseudocount'], independent_scores=independent_scores)
    result = metrics(labels, scores, tracks)
    result['face_clusters'], result['voice_clusters'] = int(fg.max()+1), int(vg.max()+1)
    return result, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='research_v5/config.yaml')
    parser.add_argument('--out', default='research_v5/runs/validation_preserved_baseline')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = device_for(cfg['device'])
    pack, original = development_pack(cfg['pack'], cfg['original_split'])
    # Fingerprints must retain the original feature-pack row indexing.
    from flag2027.common import load_pack
    full = load_pack(cfg['pack'])
    fingerprints = media_hashes(full)
    keep = np.isin(full['labels'], sorted(set(original['train']) | set(original['validation'])))
    fingerprints = tuple(np.asarray(h)[keep] for h in fingerprints)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(cfg['fold_directory'])/f'fold{i}'/name for i in range(3)
             for name in ['split.json', 'inner_control_v1/best.pt', 'outer_control/best.pt']]
    spec = {'config': cfg, 'source_hashes': {p: sha256(ROOT/p) for p in CORE},
            'pack_sha256': sha256(cfg['pack']), 'split_sha256': sha256(cfg['original_split']),
            'checkpoint_and_fold_hashes': {str(p): sha256(p) for p in paths},
            'baseline_manifest_sha256': sha256(ROOT/cfg['system']),
            'baseline_zip_sha256': sha256(ROOT/'artifacts/submission/submission.zip')}
    marker = out/'run_config.json'
    if marker.exists() and json.loads(marker.read_text()) != spec:
        raise ValueError('Experiment configuration/code/data changed; use a new run directory')
    save_json(marker, spec)
    if (out/'selected.json').exists():
        print('Verified completed validation. No test data read.')
        return
    options = candidates(cfg)
    baseline = {**options[0], 'pool_weight': 0., 'graph_weight': 0.}
    all_inner, folds = [], []
    for fold in range(3):
        folder = Path(cfg['fold_directory'])/f'fold{fold}'
        split = json.loads((folder/'split.json').read_text())
        model, weight, fitted = load_research_model(folder/'inner_control_v1/best.pt', device)
        if set(fitted) != set(split['inner_fit']) or set(fitted) & set(split['inner_validation']):
            raise ValueError('Inner checkpoint/heldout identity overlap')
        faces, voices, fi, vi, tracks, labels, edge_features = context(pack, split['inner_validation'], cfg['seed']+fold,
                                                     cfg['validation_anchors_per_identity'], fingerprints)
        mapped = map_features([(model, weight)], faces, voices, device)
        independent = per_trial_scores([(model, weight)], *edge_features, tracks == 'no_gender', device)
        groups = {}
        for f, v in itertools.product(cfg['face_thresholds'], cfg['voice_thresholds']):
            groups[f, v] = (cluster(faces[:, 4096:4608], f, cfg['linkage']),
                            cluster(voices[:, 192:704], v, cfg['linkage']))
        inner = []
        for option in options:
            result, _ = assess(mapped, faces, voices, fi, vi, tracks, labels, option,
                               groups[option['face_threshold'], option['voice_threshold']], independent_scores=independent)
            inner.append(result)
        # Deterministic tie-breaking favors less cross-trial score modification.
        chosen = min(range(len(options)), key=lambda i: (inner[i]['mean_english_eer_percent'],
                     options[i]['graph_weight'], options[i]['pool_weight'], options[i]['face_threshold'], options[i]['voice_threshold']))
        all_inner.append([r['mean_english_eer_percent'] for r in inner])
        save_json(out/f'fold{fold}_inner.json', {'candidates': options, 'results': inner, 'selected': chosen})
        model, weight, fitted = load_research_model(folder/'outer_control/best.pt', device)
        if set(fitted) != set(split['outer_fit']) or set(fitted) & set(split['outer_validation']):
            raise ValueError('Outer checkpoint/heldout identity overlap')
        if set(split['outer_validation']) & set(original['audit']):
            raise ValueError('Reserved original audit would be reused')
        faces, voices, fi, vi, tracks, labels, edge_features = context(pack, split['outer_validation'], cfg['seed']+10_000+fold,
                                                     cfg['validation_anchors_per_identity'], fingerprints)
        mapped = map_features([(model, weight)], faces, voices, device)
        independent = per_trial_scores([(model, weight)], *edge_features, tracks == 'no_gender', device)
        control, baseline_scores = assess(mapped, faces, voices, fi, vi, tracks, labels, baseline, independent_scores=independent)
        proposal, scores = assess(mapped, faces, voices, fi, vi, tracks, labels, options[chosen], independent_scores=independent)
        row = {'fold': fold, 'outer_identities': len(split['outer_validation']), 'selected_config': options[chosen],
               'baseline': control, 'transductive': proposal, 'outer_split': split['outer_validation'],
               'trials': len(labels), 'unique_faces': len(faces), 'unique_voices': len(voices)}
        folds.append(row)
        np.savez_compressed(out/f'fold{fold}_local_scores.npz', labels=labels, tracks=tracks,
                            baseline=baseline_scores, transductive=scores)
        save_json(out/'folds.json', folds)
        print('FOLD', json.dumps(row), flush=True)
    mean_inner = np.mean(all_inner, axis=0)
    chosen = min(range(len(options)), key=lambda i: (mean_inner[i], options[i]['graph_weight'], options[i]['pool_weight'],
                                                   options[i]['face_threshold'], options[i]['voice_threshold']))
    summary = {'role': 'NONCOMPLIANT_TRANSDUCTIVE_RESEARCH', 'flag_eligible': False,
               'config': cfg, 'folds': folds, 'selected_config': options[chosen],
               'selection': 'Mean inner holdout EER only, never evaluation inputs, labels or platform feedback',
               'selected_inner_mean_eer_percent': float(mean_inner[chosen]),
               'run_config_sha256': sha256(marker), 'original_audit_excluded': original['audit'],
               'limitations': ['English only; no Bengali reference labels', 'Previously researched development identities',
                 'Pooling and trial graph exploit test context and sampling; banned by FLAG',
                 'Local models are single fold controls; evaluation uses preserved four-seed ensemble',
                 'No source video separation can be guaranteed', 'Local performance is not hidden evaluation EER']}
    for name in ['baseline', 'transductive']:
        values = [row[name]['mean_english_eer_percent'] for row in folds]
        summary[name] = {'mean_english_eer_percent': float(np.mean(values)),
                         'fold_sample_std_percent': float(np.std(values, ddof=1)), 'fold_means': values}
    summary['improved_outer_mean'] = summary['transductive']['mean_english_eer_percent'] < summary['baseline']['mean_english_eer_percent']
    save_json(out/'summary.json', summary)
    save_json(out/'selected.json', {'flag_eligible': False, 'role': 'NONCOMPLIANT_TRANSDUCTIVE_RESEARCH',
        'selected_before_score_archive_export': True, 'selected_config': options[chosen],
        'prior_unlabeled_feature_encoding_occurred': True,
        'revision_reason': 'Preserve each trial official feature row after baseline consistency check; no hidden evaluation labels used',
        'validation_summary_sha256': sha256(out/'summary.json'), 'run_config_sha256': sha256(marker),
        'improved_outer_mean': summary['improved_outer_mean']})
    print('SUMMARY', json.dumps({k: summary[k] for k in ['baseline', 'transductive', 'selected_config']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
