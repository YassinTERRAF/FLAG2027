"""Evidence-Gated Multiscale Association: proposed task-specific combination."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .evidence import support_statistics


class EvidenceAssociation(nn.Module):
    def __init__(self, variant='multiscale_attention', hidden=24, dropout=.1, max_correction=3.):
        super().__init__()
        if variant not in ['hard_residual', 'multiscale_attention']:
            raise ValueError('Unknown architecture')
        self.variant = variant
        self.max_correction = max_correction
        # Three FaceNet/WavLM scales plus a complementary VGG/ECAPA scale.
        self.register_buffer('initial_temperature', torch.tensor([[.1, .025], [.2, .05], [.4, .1], [.3, .1]]))
        self.temperature_offset = nn.Parameter(torch.zeros(4, 2), requires_grad=variant=='multiscale_attention')
        self.pair_encoder = nn.Sequential(nn.Linear(6, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.evidence_encoder = nn.Sequential(nn.Linear(6, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head_embedding = nn.Parameter(torch.randn(5, hidden)*.01)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.correction = nn.Sequential(nn.Linear(2*hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.gate = nn.Linear(2*hidden, 1)
        self.loss_log_scale = nn.Parameter(torch.tensor(math.log(2.)))
        self.loss_bias = nn.Parameter(torch.tensor(-.5))

    def temperatures(self):
        # Bounded to half/twice the prior to avoid collapsing to self-only/all-one kernels.
        return self.initial_temperature*torch.exp(math.log(2)*self.temperature_offset.tanh())

    def forward(self, data, support_mask=None):
        keep = data['keep']
        if support_mask is not None:
            keep = keep*support_mask[None, :]
        hard, hard_support = support_statistics(data['hard_face'][None], data['hard_voice'][None], keep)
        hard = hard[0]
        prior = data['independent']+hard[:, 0]
        pair = self.pair_encoder(data['pair'])
        if self.variant == 'multiscale_attention':
            distances = data['distances']
            tau = self.temperatures()
            df = torch.stack([distances[0], distances[0], distances[0], distances[2]])
            dv = torch.stack([distances[1], distances[1], distances[1], distances[3]])
            af = torch.exp(-df/tau[:, 0, None, None])
            av = torch.exp(-dv/tau[:, 1, None, None])
            soft, support = support_statistics(af, av, keep)
            tokens = torch.cat([hard[:, None], soft.permute(1, 0, 2)], dim=1)
            embedded = self.evidence_encoder(tokens)+self.head_embedding[None]
            attention = ((self.query(pair)[:, None]*self.key(embedded)).sum(-1)/math.sqrt(pair.shape[-1])).softmax(-1)
            evidence = (attention[:, :, None]*self.value(embedded)).sum(1)
            reliability = (torch.cat([hard_support, support], dim=0)/(torch.cat([hard_support, support], dim=0)+1)).max(0).values
        else:
            evidence = self.evidence_encoder(hard)
            attention = torch.ones((len(prior), 1), device=prior.device)
            reliability = hard_support[0]/(hard_support[0]+1)
        joined = torch.cat([pair, evidence], dim=-1)
        gate = self.gate(joined).sigmoid().squeeze(-1)*reliability
        correction = self.max_correction*self.correction(joined).squeeze(-1).tanh()*gate
        return {'score': prior+correction, 'prior': prior, 'correction': correction,
                'gate': gate, 'attention': attention, 'temperatures': self.temperatures()}

    def loss_logits(self, scores):
        return self.loss_log_scale.clamp(-2, 3).exp()*scores+self.loss_bias


def objective(model, output, labels, data, cfg, dropped=None):
    logits = model.loss_logits(output['score'])
    verification = 0.
    for track in [True, False]:
        mask = data['no_gender'] == track
        verification = verification+.5*F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
    # Same voice, positive-vs-same-gender-negative ranking, within supervised episodes.
    same_voice = data['voice_index'][:, None] == data['voice_index'][None, :]
    pair = same_voice & (labels[:, None] > .5) & (labels[None, :] < .5)
    pair = pair & (~data['no_gender'][:, None]) & (~data['no_gender'][None, :])
    difference = output['score'][:, None]-output['score'][None, :]
    ranking = F.softplus((cfg['ranking_margin']-difference[pair])/cfg['ranking_temperature']).mean() if pair.any() else logits.sum()*0
    residual = output['correction'].square().mean()
    consistency = logits.sum()*0
    if dropped is not None:
        consistency = F.smooth_l1_loss(dropped['correction'], output['correction'].detach())
    kernel = model.temperature_offset.square().mean()
    total = verification+cfg['ranking_weight']*ranking+cfg['residual_weight']*residual
    total = total+cfg['consistency_weight']*consistency+cfg['kernel_regularization']*kernel
    return total
