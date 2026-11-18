"""Frozen original architecture, with independent face/audio mapping on MPS."""
import numpy as np
import torch
from flag2027.model import Association
from flag2027.train import load_model


def load_research_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['kind'] != 'control_v1':
        raise ValueError('Expected the preserved original architecture control')
    model = Association(5376, 1728, len(ck['classes']), 128, 256, .6).to(device)
    model.load_state_dict(ck['state_dict'])
    return model.eval(), .25, ck['classes']


@torch.inference_mode()
def map_features(members, faces, voices, device):
    result = []
    for model, weight in members:
        model.eval()
        output = []
        for features, branch, gender in [(faces, model.face, model.face_gender),
                                         (voices, model.voice, model.voice_gender)]:
            embeds, sexes = [], []
            for start in range(0, len(features), 256):
                x = torch.as_tensor(features[start:start+256], dtype=torch.float32, device=device)
                embeds.append(branch(x).cpu().numpy())
                z = torch.nn.functional.normalize(x, dim=-1)*np.sqrt(x.shape[-1])
                sexes.append(gender(z).softmax(-1).cpu().numpy())
            output.append((np.concatenate(embeds), np.concatenate(sexes)))
        (f, gf), (v, gv) = output
        result.append((f, v, gf, gv, float(weight)))
    return result


def per_trial_scores(members, faces, voices, no_gender, device):
    mapped = map_features(members, faces, voices, device)
    scores = []
    for f, v, gf, gv, weight in mapped:
        s = np.einsum('ij,ij->i', f, v)
        s += np.asarray(no_gender, dtype=bool)*weight*np.einsum('ij,ij->i', gf, gv)
        scores.append(s)
    return np.mean(scores, axis=0)
