import torch
from torch.nn import functional as F


def same_gender_rank(scores, labels, genders, temperature=0.1, margin=0.15):
    """Smooth hard-negative ranking; negatives are different IDs, same gender.

    This is a research combination of established ranking/contrastive ideas,
    not a claim of a new published loss or a guaranteed EER reduction.
    """
    positive = labels[:, None] == labels[None, :]
    negative = (~positive) & (genders[:, None] == genders[None, :])
    if not negative.any(dim=1).all():
        raise ValueError('Every training anchor needs a different identity of its gender')
    def direction(s, pos, neg):
        pos_mean = (s * pos).sum(dim=1) / pos.sum(dim=1)
        neg_soft = temperature * (torch.logsumexp((s / temperature).masked_fill(~neg, -1e4), dim=1)
                                  - neg.sum(dim=1).float().log())
        return F.softplus((margin + neg_soft - pos_mean) / temperature).mean()
    return 0.5 * (direction(scores, positive, negative) + direction(scores.T, positive.T, negative.T))


def relational_loss(student, raw, dims):
    # Frozen unimodal teacher geometry; TRAINING BATCH ONLY, never inference.
    blocks = raw.split(dims, dim=-1)
    with torch.no_grad():
        teachers = [F.normalize(b, dim=-1) for b in blocks[:2]]
        target = sum(t @ t.T for t in teachers) / len(teachers)
    mask = ~torch.eye(len(raw), dtype=torch.bool, device=raw.device)
    return F.smooth_l1_loss((student @ student.T)[mask], target[mask])


def loss(model, output, face, voice, labels, genders, cfg, relation=False):
    from flag2027.model import contrastive_loss
    from .model import FACE_DIMS, VOICE_DIMS
    f, v, _, _, gf, gv = output
    rank = same_gender_rank(model.pair_matrix(output), labels, genders, cfg['temperature'], cfg['ranking_margin'])
    contrast = contrastive_loss(f, v, labels, genders, cfg['temperature'])
    proxy = 0.5 * (F.cross_entropy(16 * (f @ model.proxies.T), labels) +
                   F.cross_entropy(16 * (v @ model.proxies.T), labels))
    gender = 0.5 * (F.cross_entropy(gf, genders) + F.cross_entropy(gv, genders))
    total = rank + cfg['contrastive_weight']*contrast + cfg['proxy_weight']*proxy + cfg['gender_weight']*gender
    if relation:
        total = total + cfg['relation_weight']*0.5*(relational_loss(f, face, FACE_DIMS) + relational_loss(v, voice, VOICE_DIMS))
    return total
