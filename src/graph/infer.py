"""Save an explicitly NONCOMPLIANT four-file research score archive. No upload."""
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
from flag2027.common import ROOT, device_for, load_pack, save_json, sha256
from flag2027.infer import CELLS, validate_archive
from flag2027.train import load_model
from .features import verify_selection
from .mapping import map_features, per_trial_scores
from .transductive import cluster, score


def infer(validation='research_v5/runs/validation_preserved_baseline', features='research_v5/runs/evaluation_features_preserved_baseline',
          output='research_v5/runs/NONCOMPLIANT_submission', device='mps'):
    selected, spec = verify_selection(validation)
    cfg = spec['config']
    system_path = ROOT/cfg['system']
    system = json.loads(system_path.read_text())
    if sha256(system_path) != spec['baseline_manifest_sha256']:
        raise ValueError('Original system changed')
    for name, expected in {**system['source_hashes'], **system['pretrained_registries']}.items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Original source/encoder registry changed: '+name)
    members = []
    d = device_for(device)
    for item in system['members']:
        path = ROOT/item['checkpoint']
        if sha256(path) != item['sha256']:
            raise ValueError('Original checkpoint changed')
        model, ck = load_model(path, d)
        members.append((model, ck['gender_score_weight']))
    fp = Path(features)
    meta = json.loads((fp/'manifest.json').read_text())
    if meta['flag_eligible'] or meta['contains_reference_labels']:
        raise ValueError('Unexpected feature metadata')
    if sha256(fp/'pool.npz') != meta['pack_sha256'] or sha256(fp/'provenance.json') != meta['provenance_sha256']:
        raise ValueError('Unlabeled pool changed')
    provenance = json.loads((fp/'provenance.json').read_text())
    if provenance['selection_sha256'] != sha256(Path(validation)/'selected.json'):
        raise ValueError('Selection changed after feature extraction')
    data = load_pack(fp/'pool.npz')
    if 'labels' in data:
        raise ValueError('Reference labels must never enter evaluation scorer')
    faces, voices = data['faces'], data['voices']
    option = selected['selected_config']
    fg = cluster(faces[:, 4096:4608], option['face_threshold'], option['linkage'])
    vg = cluster(voices[:, 192:704], option['voice_threshold'], option['linkage'])
    mapped = map_features(members, faces, voices, d)
    args = (mapped, fg, vg, data['face_index'], data['voice_index'], data['tracks'] == 'no_gender')
    fi, vi = data['face_index'], data['voice_index']
    trial_faces = np.concatenate([data['trial_official_faces'], faces[fi, 4096:]], axis=1)
    trial_voices = np.concatenate([data['trial_official_voices'], voices[vi, 192:]], axis=1)
    baseline = per_trial_scores(members, trial_faces, trial_voices, data['tracks'] == 'no_gender', d)
    scores = score(*args, pool_weight=option['pool_weight'], graph_weight=option['graph_weight'],
                   pseudocount=option['graph_pseudocount'], independent_scores=baseline)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    notice = ('NONCOMPLIANT FLAG RESEARCH ARCHIVE\n'
              'Uses evaluation-wide content linking, clustering, cluster averaging and/or trial co-occurrence score adjustment.\n'
              'These procedures violate the FLAG organizer rules supplied by the user.\n'
              'Do not describe or submit this as a compliant FLAG entry.\n'
              'Hidden evaluation labels were unavailable. Evaluation EER, including the 15% target, is NOT VERIFIED.\n')
    (out/'COMPLIANCE_NOTICE.txt').write_text(notice)
    record = {'role': 'NONCOMPLIANT_TRANSDUCTIVE_RESEARCH_ARCHIVE', 'flag_eligible': False,
              'evaluation_eer': None, 'evaluation_eer_verified': False, 'upload_performed': False,
              'selection_sha256': sha256(Path(validation)/'selected.json'),
              'system_sha256': sha256(system_path), 'feature_pack_sha256': meta['pack_sha256'],
              'configuration': option, 'face_clusters': int(fg.max()+1), 'voice_clusters': int(vg.max()+1),
              'pooling': 'All four unlabeled cells jointly; additional cross-language behavior unvalidated locally',
              'cells': {}}
    for track, lang, condition in CELLS:
        mask = (data['tracks'] == track) & (data['languages'] == lang)
        filename = f'{track}/sub_score_v4_{lang}_{condition}.txt'
        dest = out/filename
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(''.join(f'{key} {-float(s):.10f}\n' for key, s in zip(data['keys'][mask], scores[mask])))
        original = ROOT/'artifacts/submission'/filename
        old = np.asarray([float(line.split()[1]) for line in original.read_text().splitlines()])
        if len(old) != mask.sum():
            raise ValueError('Original score count differs')
        difference = np.abs(old+baseline[mask])
        if float(difference.max()) > .002:
            raise ValueError('Recomputed original baseline differs unexpectedly; inspect extraction before packaging')
        record['cells'][f'{track}/{lang}'] = {'trials': int(mask.sum()), 'score_file_sha256': sha256(dest),
            'original_baseline_max_absolute_difference': float(difference.max()),
            'mean_absolute_score_change': float(np.mean(np.abs(scores[mask]-baseline[mask])))}
    archive = out/'submissions.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for track, lang, condition in CELLS:
            name = f'{track}/sub_score_v4_{lang}_{condition}.txt'
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (out/name).read_bytes())
    validate_archive(archive, ROOT/cfg['evaluation_root'])
    record['archive_sha256'] = sha256(archive)
    record['format_validation_passed'] = True
    np.savez_compressed(out/'research_scores_and_clusters.npz', similarities=scores, baseline=baseline,
                        face_clusters=fg, voice_clusters=vg)
    if sha256(ROOT/'artifacts/submission/submission.zip') != spec['baseline_zip_sha256']:
        raise ValueError('Original submission ZIP changed')
    save_json(out/'NONCOMPLIANT_manifest.json', record)
    print(f'Saved NONCOMPLIANT research archive: {archive}. Format checked; evaluation EER unknown. No upload.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--validation', default='research_v5/runs/validation_preserved_baseline')
    p.add_argument('--features', default='research_v5/runs/evaluation_features_preserved_baseline')
    p.add_argument('--output', default='research_v5/runs/NONCOMPLIANT_submission')
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    infer(a.validation, a.features, a.output, a.device)
