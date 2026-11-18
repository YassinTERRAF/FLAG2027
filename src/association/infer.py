"""Generate a separate EGMA candidate ZIP, preserving the user's current ZIP."""
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from flag2027.common import ROOT, device_for, load_pack, save_json, sha256
from flag2027.infer import CELLS, validate_archive
from flag2027.train import load_model
from .evidence import prepare, to_device
from .train import restore


def main():
    final = ROOT/'research_v6/runs/final'
    meta = json.loads((final/'manifest.json').read_text())
    spec = json.loads((final/'final_config.json').read_text())
    if sha256(final/'best.pt') != meta['checkpoint_sha256'] or sha256(final/'final_config.json') != meta['final_config_sha256']:
        raise ValueError('Candidate checkpoint/config changed')
    for name, expected in spec['source_hashes'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Candidate source changed after refit: '+name)
    original_zip_hash = sha256(ROOT/'submissions.zip')
    if original_zip_hash != spec['preserved_zip_sha256']:
        raise ValueError('Previous user ZIP changed')
    device = device_for(spec['config']['device'])
    system = json.loads((ROOT/'artifacts/final/system.json').read_text())
    for name, expected in {**system['source_hashes'], **system['pretrained_registries']}.items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Original frozen system source changed')
    members = []
    for item in system['members']:
        if sha256(ROOT/item['checkpoint']) != item['sha256']:
            raise ValueError('Original frozen mapper changed')
        mapper, ck = load_model(ROOT/item['checkpoint'], device)
        members.append((mapper, ck['gender_score_weight']))
    fp = ROOT/'research_v5/runs/evaluation_features_preserved_baseline'
    fm = json.loads((fp/'manifest.json').read_text())
    if sha256(fp/'pool.npz') != fm['pack_sha256']:
        raise ValueError('Cached unlabeled evaluation features changed')
    pool = load_pack(fp/'pool.npz')
    if 'labels' in pool:
        raise ValueError('No reference labels may enter inference')
    f, v, fi, vi = pool['faces'], pool['voices'], pool['face_index'], pool['voice_index']
    edge = (np.concatenate([pool['trial_official_faces'], f[fi, 4096:]], axis=1),
            np.concatenate([pool['trial_official_voices'], v[vi, 192:]], axis=1))
    inputs = to_device(prepare(f, v, fi, vi, pool['tracks'], edge, members, device, spec['prior']), device)
    model, _ = restore(final/'best.pt', device)
    with torch.inference_mode():
        output = model(inputs)
    score, prior = output['score'].cpu().numpy(), output['prior'].cpu().numpy()
    previous = load_pack(ROOT/'research_v5/runs/NONCOMPLIANT_submission/research_scores_and_clusters.npz')['similarities']
    if np.max(np.abs(prior-previous)) > 2e-5:
        raise ValueError('Prior does not reproduce the preserved V5 score')
    out = ROOT/'research_v6/runs/submission'
    out.mkdir(parents=True, exist_ok=True)
    record = {'role': 'EGMA_architecture_candidate', 'hidden_evaluation_eer': None,
              'local_development_english_eer_percent': meta['local_development_mean_english_eer_percent'],
              'local_score_role': meta['score_role'],
              'checkpoint_sha256': meta['checkpoint_sha256'], 'feature_pool_sha256': fm['pack_sha256'],
              'previous_zip_preserved_sha256': original_zip_hash,
              'rules_status': meta['rules_status'], 'upload_performed': False,
              'prior_max_absolute_difference': float(np.max(np.abs(prior-previous))), 'cells': {}}
    for track, lang, condition in CELLS:
        mask = (pool['tracks'] == track) & (pool['languages'] == lang)
        name = f'{track}/sub_score_v4_{lang}_{condition}.txt'
        dest = out/name
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(''.join(f'{key} {-float(s):.10f}\n' for key, s in zip(pool['keys'][mask], score[mask])))
        record['cells'][f'{track}/{lang}'] = {'trials': int(mask.sum()), 'sha256': sha256(dest)}
    archive = out/'submissions.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for track, lang, condition in CELLS:
            name = f'{track}/sub_score_v4_{lang}_{condition}.txt'
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (out/name).read_bytes())
    validate_archive(archive, ROOT/'data/evaluation')
    record['archive_sha256'] = sha256(archive)
    record['format_valid'] = True
    np.savez_compressed(out/'scores_and_attention.npz', similarities=score, prior=prior,
                        attention=output['attention'].cpu().numpy(), gate=output['gate'].cpu().numpy(),
                        correction=output['correction'].cpu().numpy())
    save_json(out/'manifest.json', record)
    (out/'README.txt').write_text('EGMA candidate. Local English nested EER is not hidden multilingual evaluation EER.\n'
        'Uses transductive cross-trial graph evidence. User reports updated organizer rules permit this.\n'
        'Previous top-level submissions.zip is preserved. No upload performed.\n')
    if sha256(ROOT/'submissions.zip') != original_zip_hash:
        raise ValueError('Previous user ZIP was changed')
    print('Saved separate EGMA candidate:', archive, flush=True)


if __name__ == '__main__':
    main()
