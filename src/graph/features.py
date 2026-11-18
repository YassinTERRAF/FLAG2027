"""NONCOMPLIANT research pool: encode and link unlabeled evaluation samples."""
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from flag2027.common import ROOT, save_json, save_npy, sha256, unit
from flag2027.infer import CELLS, cell_files, parse_pairs
from flag2027.data import resolve_media


def verify_selection(directory):
    directory = Path(directory)
    selected = json.loads((directory/'selected.json').read_text())
    if selected['flag_eligible'] or not selected['selected_before_score_archive_export']:
        raise ValueError('Expected explicitly noncompliant, validation-first selection')
    if sha256(directory/'summary.json') != selected['validation_summary_sha256']:
        raise ValueError('Local validation summary changed')
    if sha256(directory/'run_config.json') != selected['run_config_sha256']:
        raise ValueError('Validation provenance changed')
    spec = json.loads((directory/'run_config.json').read_text())
    for name, expected in spec['source_hashes'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Research source changed after validation: '+name)
    if not selected['improved_outer_mean']:
        raise ValueError('No local improvement; evaluation generation is gated off')
    return selected, spec


def extract(validation='research_v5/runs/validation_preserved_baseline', output='research_v5/runs/evaluation_features_preserved_baseline', device='mps'):
    selected, validation_spec = verify_selection(validation)
    cfg = validation_spec['config']
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out/'cache'
    cache.mkdir(exist_ok=True)
    faces, voices, rows, inputs = {}, {}, [], {}
    trial_official_faces, trial_official_voices = [], []
    for track, lang, condition in CELLS:
        pair_path, features = cell_files(ROOT/cfg['evaluation_root'], track, lang)
        pairs = parse_pairs(pair_path)
        f = pd.read_csv(features['faces'], header=None).to_numpy(dtype=np.float32)
        v = pd.read_csv(features['voices'], header=None).to_numpy(dtype=np.float32)
        if f.shape != (len(pairs), 4096) or v.shape != (len(pairs), 192):
            raise ValueError('Unlabeled CSV/trial dimension mismatch')
        inputs[f'{track}/{lang}'] = {'pair_list_sha256': sha256(pair_path),
            'face_csv_sha256': sha256(features['faces']), 'voice_csv_sha256': sha256(features['voices']), 'trials': len(pairs)}
        trial_official_faces.append(unit(f))
        trial_official_voices.append(unit(v))
        for i, (key, audio, image) in enumerate(pairs):
            fp, vp = resolve_media(pair_path.parent, image), resolve_media(pair_path.parent, audio)
            fh, vh = sha256(fp), sha256(vp)
            # Exact content hashing is cross-trial linking, explicitly prohibited by FLAG.
            faces.setdefault(fh, (fp, unit(f[i:i+1])[0]))
            voices.setdefault(vh, (vp, unit(v[i:i+1])[0]))
            rows.append((track, lang, condition, key, fh, vh))
    spec = {'role': 'NONCOMPLIANT_EVALUATION_FEATURE_POOL', 'flag_eligible': False,
            'contains_reference_labels': False, 'inputs': inputs,
            'selection_sha256': sha256(Path(validation)/'selected.json'),
            'encoders_sha256': sha256(ROOT/'models/registry.json'),
            'demographics_sha256': sha256(ROOT/'models/demographics_registry.json'),
            'face_contents': sorted(faces), 'voice_contents': sorted(voices),
            'preprocessing': 'same frozen encoders; independent full clip; content pool plus preserved per-trial official CSV blocks v5.2'}
    marker = out/'provenance.json'
    if marker.exists() and json.loads(marker.read_text()) != spec:
        raise ValueError('Unlabeled data or encoder/selection provenance changed')
    save_json(marker, spec)
    pack_path = out/'pool.npz'
    if pack_path.exists() and (out/'manifest.json').exists():
        meta = json.loads((out/'manifest.json').read_text())
        if sha256(pack_path) != meta['pack_sha256'] or sha256(marker) != meta['provenance_sha256']:
            raise ValueError('Cached evaluation feature pool changed')
        print('Verified completed evaluation feature pool.', flush=True)
        return
    print(f'Unlabeled pool: {len(rows)} trials, {len(faces)} unique images, {len(voices)} unique audio files', flush=True)
    from flag2027.encoders import FrozenEncoders
    from flag2027.demographics import FrozenDemographics
    os.environ['TORCH_HOME'] = str(ROOT/'models/torch')
    encoders, demographics = FrozenEncoders(device), FrozenDemographics(device)
    for model in [encoders.face, encoders.voice, demographics.face, demographics.voice]:
        assert not any(p.requires_grad for p in model.parameters())
    missing = [h for h in sorted(faces) if not (cache/(h+'_f.npy')).exists()]
    for start in tqdm(range(0, len(missing), 16), desc='Unique evaluation faces'):
        batch = missing[start:start+16]
        paths = [faces[h][0] for h in batch]
        f = np.concatenate([np.stack([faces[h][1] for h in batch]), unit(encoders.face_embeddings(paths)),
                            unit(demographics.face_embeddings(paths))], axis=1)
        for h, feature in zip(batch, f):
            save_npy(cache/(h+'_f.npy'), feature)
    groups = {}
    for h in sorted(voices):
        if (cache/(h+'_v.npy')).exists():
            continue
        info = sf.info(voices[h][0])
        seconds = math.ceil(math.ceil(info.frames*16000/info.samplerate)/16000)
        groups.setdefault(seconds, []).append(h)
    with tqdm(total=sum(len(x) for x in groups.values()), desc='Unique evaluation voices') as progress:
        for seconds, hashes in sorted(groups.items()):
            size = 4 if seconds <= 20 else 2 if seconds <= 40 else 1
            for start in range(0, len(hashes), size):
                batch = hashes[start:start+size]
                paths = [voices[h][0] for h in batch]
                v = np.concatenate([np.stack([voices[h][1] for h in batch]), unit(encoders.voice_embeddings(paths)),
                                    unit(demographics.voice_embeddings(paths))], axis=1)
                for h, feature in zip(batch, v):
                    save_npy(cache/(h+'_v.npy'), feature)
                progress.update(len(batch))
            if device == 'mps':
                import torch
                torch.mps.empty_cache()
    fk, vk = sorted(faces), sorted(voices)
    flookup, vlookup = {h: i for i, h in enumerate(fk)}, {h: i for i, h in enumerate(vk)}
    arrays = {'faces': np.stack([np.load(cache/(h+'_f.npy')) for h in fk]),
              'voices': np.stack([np.load(cache/(h+'_v.npy')) for h in vk]),
              'face_index': np.asarray([flookup[r[4]] for r in rows]), 'voice_index': np.asarray([vlookup[r[5]] for r in rows]),
              'tracks': np.asarray([r[0] for r in rows]), 'languages': np.asarray([r[1] for r in rows]),
              'conditions': np.asarray([r[2] for r in rows]), 'keys': np.asarray([r[3] for r in rows]),
              'trial_official_faces': np.concatenate(trial_official_faces),
              'trial_official_voices': np.concatenate(trial_official_voices)}
    if not np.isfinite(arrays['faces']).all() or not np.isfinite(arrays['voices']).all():
        raise ValueError('Non-finite evaluation features')
    np.savez_compressed(pack_path, **arrays)
    save_json(out/'manifest.json', {'flag_eligible': False, 'contains_reference_labels': False,
        'pack_sha256': sha256(pack_path), 'provenance_sha256': sha256(marker),
        'trials': len(rows), 'unique_faces': len(faces), 'unique_voices': len(voices)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--validation', default='research_v5/runs/validation_preserved_baseline')
    p.add_argument('--output', default='research_v5/runs/evaluation_features_preserved_baseline')
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    extract(a.validation, a.output, a.device)
