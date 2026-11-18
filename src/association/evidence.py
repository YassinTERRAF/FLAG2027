"""Label-free query/support geometry. A current media pair never supports itself."""
import numpy as np
import torch
from flag2027.common import unit
from research_v5.transductive import cluster
from research_v5.mapping import map_features, per_trial_scores


def prepare(faces, voices, fi, vi, tracks, edge_features, members, device, prior):
    fi, vi = np.asarray(fi), np.asarray(vi)
    support = np.unique(np.column_stack([fi, vi]), axis=0)
    sf, sv = support[:, 0], support[:, 1]
    geometry = []
    for raw, query, context, begin, end in [(faces, fi, sf, 4096, 4608),
                                            (voices, vi, sv, 192, 704),
                                            (faces, fi, sf, 0, 4096),
                                            (voices, vi, sv, 0, 192)]:
        x = torch.as_tensor(unit(raw[:, begin:end]), device=device)
        with torch.no_grad():
            d = (1-x[query]@x[context].T).clamp(0, 2).cpu().numpy()
        geometry.append(d.astype(np.float32))
    fg = cluster(faces[:, 4096:4608], prior['face_threshold'], prior['linkage'])
    vg = cluster(voices[:, 192:704], prior['voice_threshold'], prior['linkage'])
    hf = (fg[fi, None] == fg[sf][None, :]).astype(np.float32)
    hv = (vg[vi, None] == vg[sv][None, :]).astype(np.float32)
    self_pair = (fi[:, None] == sf[None, :]) & (vi[:, None] == sv[None, :])
    keep = (~self_pair).astype(np.float32)
    ng = np.asarray(tracks) == 'no_gender'
    independent = per_trial_scores(members, *edge_features, ng, device)
    mapped = map_features(members, *edge_features, device)
    pair = []
    for f, v, gf, gv, weight in mapped:
        cosine = np.sum(f*v, axis=1)
        gender_similarity = np.sum(gf*gv, axis=1)
        ef = -(gf*np.log(np.maximum(gf, 1e-8))).sum(1)
        ev = -(gv*np.log(np.maximum(gv, 1e-8))).sum(1)
        pair.append(np.column_stack([cosine, gender_similarity, ef, ev]))
    pair = np.column_stack([independent, np.mean(pair, axis=0), ng.astype(np.float32)]).astype(np.float32)
    return {'distances': np.stack(geometry), 'hard_face': hf, 'hard_voice': hv,
            'keep': keep, 'pair': pair, 'voice_index': vi.astype(np.int64),
            'no_gender': ng, 'independent': independent.astype(np.float32)}


def to_device(data, device):
    return {k: torch.as_tensor(v, device=device) for k, v in data.items()}


def support_statistics(face_affinity, voice_affinity, keep, pseudocount=.5):
    """All tensors [heads, queries, unique support pairs] or broadcastable."""
    face = face_affinity*keep
    voice = voice_affinity*keep
    observed = (face_affinity*voice_affinity*keep).sum(-1)
    face_mass, voice_mass = face.sum(-1), voice.sum(-1)
    total = keep.sum(-1).clamp_min(1)
    expected = face_mass*voice_mass/total
    evidence = ((observed+pseudocount)/(expected+pseudocount)).log()
    reliability = observed/(observed+1)
    tokens = torch.stack([evidence, observed.log1p(), expected.log1p(),
                          face_mass.log1p(), voice_mass.log1p(), reliability], dim=-1)
    return tokens, observed
