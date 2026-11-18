import math

import torch
from torch import nn
from torch.nn import functional as F


class Branch(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, dropout):
        super().__init__()
        layers = [nn.Dropout(dropout)]
        if hidden_dim:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim or input_dim, embed_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Per-example L2 normalization is part of the model, never a cohort operation.
        x = F.normalize(x, dim=-1) * math.sqrt(x.shape[-1])
        return F.normalize(self.layers(x), dim=-1)


class Association(nn.Module):
    def __init__(self, face_dim, voice_dim, n_classes, embedding_dim=128, hidden_dim=0, dropout=0.3):
        super().__init__()
        self.face = Branch(face_dim, embedding_dim, hidden_dim, dropout)
        self.voice = Branch(voice_dim, embedding_dim, hidden_dim, dropout)
        self.proxies = nn.Parameter(torch.randn(n_classes, embedding_dim) * 0.01)
        # Auxiliary models are trained only on English official training identities.
        self.face_gender = nn.Linear(face_dim, 2)
        self.voice_gender = nn.Linear(voice_dim, 2)

    def forward(self, faces, voices):
        f, v = self.face(faces), self.voice(voices)
        gf = self.face_gender(F.normalize(faces, dim=-1) * math.sqrt(faces.shape[-1]))
        gv = self.voice_gender(F.normalize(voices, dim=-1) * math.sqrt(voices.shape[-1]))
        return f, v, gf, gv

    def angular_logits(self, embeddings, labels, margin=0.15, scale=16):
        cosine = (embeddings @ F.normalize(self.proxies, dim=-1).T).clamp(-1 + 1e-6, 1 - 1e-6)
        target = torch.cos(torch.acos(cosine) + margin)
        # Monotonic correction for angles approaching pi (ArcFace-style fallback).
        target = torch.where(cosine > math.cos(math.pi - margin), target,
                             cosine - math.sin(math.pi - margin) * margin)
        mask = F.one_hot(labels, cosine.shape[1]).float()
        return scale * (cosine * (1 - mask) + target * mask)


def contrastive_loss(f, v, labels, genders, temperature):
    logits = f @ v.T / temperature
    positives = (labels[:, None] == labels[None, :]).float()
    targets = positives / positives.sum(dim=1, keepdim=True)
    loss = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss += -(targets.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    # Same-gender contrast prevents solving the objective using gender alone.
    same_gender = genders[:, None] == genders[None, :]
    hard_logits = logits.masked_fill(~same_gender, -1e4)
    hard_loss = -(targets * F.log_softmax(hard_logits, dim=1)).sum(dim=1).mean()
    hard_loss += -(targets.T * F.log_softmax(hard_logits.T, dim=1)).sum(dim=1).mean()
    return 0.25 * (loss + hard_loss)


def joint_loss(model, outputs, labels, genders, cfg):
    f, v, gf, gv = outputs
    contrast = contrastive_loss(f, v, labels, genders, cfg["temperature"])
    proxy = (F.cross_entropy(model.angular_logits(f, labels), labels) +
             F.cross_entropy(model.angular_logits(v, labels), labels)) * 0.5
    gender = (F.cross_entropy(gf, genders) + F.cross_entropy(gv, genders)) * 0.5
    return contrast + cfg["proxy_weight"] * proxy + cfg["gender_weight"] * gender
