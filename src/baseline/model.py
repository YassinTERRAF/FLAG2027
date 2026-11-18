"""Separate identity/context feature paths and a fixed shared simplex classifier.

All pretrained encoders remain frozen; this module consumes cached training features.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

FACE_DIMS = (4096, 512, 768)
VOICE_DIMS = (192, 512, 1024)


class BlockProjection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, output_dim),
                                 nn.LayerNorm(output_dim), nn.GELU())

    def forward(self, x):
        return self.net(F.normalize(x, dim=-1) * math.sqrt(x.shape[-1]))


class Modality(nn.Module):
    def __init__(self, dims, block_dim, embed_dim, context_dim, dropout, block_dropout):
        super().__init__()
        self.dims = dims
        self.identity_blocks = nn.ModuleList([BlockProjection(d, block_dim, dropout) for d in dims[:2]])
        self.context = BlockProjection(dims[2], context_dim, dropout)
        self.fuse = nn.Linear(block_dim * 2, embed_dim)
        self.gates = nn.Parameter(torch.zeros(2))
        self.block_dropout = block_dropout

    def forward(self, x):
        blocks = x.split(self.dims, dim=-1)
        identity = torch.stack([layer(block) for layer, block in zip(self.identity_blocks, blocks[:2])], dim=1)
        if self.training and self.block_dropout:
            keep = (torch.rand(x.shape[0], 2, 1, device=x.device) >= self.block_dropout).to(x.dtype)
            # At least one identity feature block remains for every example.
            both_missing = keep.sum(dim=1, keepdim=True) == 0
            keep[:, :1] = torch.where(both_missing, torch.ones_like(keep[:, :1]), keep[:, :1])
            identity = identity * keep
        identity = identity * (2 * self.gates.softmax(dim=0))[None, :, None]
        z = F.normalize(self.fuse(identity.flatten(start_dim=1)), dim=-1)
        context = F.normalize(self.context(blocks[2]), dim=-1)
        return z, context


def simplex(classes, dimension):
    if classes < 2 or dimension < classes - 1:
        raise ValueError("Fixed simplex needs >=2 classes and dimension >= classes-1")
    # Columns of centered I form a regular simplex. Project its rank-(C-1)
    # subspace into the requested embedding dimension without distorting angles.
    centered = torch.eye(classes) - torch.ones(classes, classes) / classes
    _, _, vh = torch.linalg.svd(centered)
    coordinates = vh[:classes-1].T
    basis, _ = torch.linalg.qr(torch.randn(dimension, classes-1), mode='reduced')
    return F.normalize(coordinates @ basis.T, dim=-1)


class BlockAssociation(nn.Module):
    def __init__(self, n_classes, embedding_dim=96, block_dim=48, context_dim=24,
                 dropout=0.3, block_dropout=0.25):
        super().__init__()
        args = (block_dim, embedding_dim, context_dim, dropout, block_dropout)
        self.face = Modality(FACE_DIMS, *args)
        self.voice = Modality(VOICE_DIMS, *args)
        self.register_buffer('proxies', simplex(n_classes, embedding_dim))
        self.context_gate = nn.Parameter(torch.tensor(-2.0))
        self.face_gender = nn.Linear(sum(FACE_DIMS), 2)
        self.voice_gender = nn.Linear(sum(VOICE_DIMS), 2)

    def forward(self, face, voice):
        f, cf = self.face(face)
        v, cv = self.voice(voice)
        gf = self.face_gender(F.normalize(face, dim=-1) * math.sqrt(face.shape[-1]))
        gv = self.voice_gender(F.normalize(voice, dim=-1) * math.sqrt(voice.shape[-1]))
        return f, v, cf, cv, gf, gv

    def pair_matrix(self, output):
        f, v, cf, cv, _, _ = output
        return f @ v.T + 0.25 * self.context_gate.sigmoid() * (cf @ cv.T)

    def pair_scores(self, output):
        f, v, cf, cv, _, _ = output
        return (f*v).sum(-1) + 0.25 * self.context_gate.sigmoid() * (cf*cv).sum(-1)
