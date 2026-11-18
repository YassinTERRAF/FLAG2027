"""NONCOMPLIANT: context-dependent clustering, pooling and trial-graph scores.

This module deliberately accepts NO ground-truth labels, names, or identities.
Trial co-occurrence is a benchmark sampling signal, not independent matching.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from flag2027.common import unit


def cluster(features, threshold, method='average'):
    features = unit(features)
    if len(features) == 0:
        raise ValueError('No samples to cluster')
    if not np.isfinite(features).all() or (np.linalg.norm(features, axis=1) < .5).any():
        raise ValueError('Non-finite or zero identity features')
    if len(features) == 1:
        return np.zeros(1, dtype=np.int64)
    distances = np.clip(pdist(features.astype(np.float64), metric='cosine'), 0, 2)
    tree = linkage(distances, method=method)
    return fcluster(tree, t=float(threshold), criterion='distance').astype(np.int64)-1


def means(values, groups, normalize=False):
    result = np.zeros((int(groups.max())+1, values.shape[1]), dtype=np.float32)
    np.add.at(result, groups, values)
    result /= np.bincount(groups)[:, None]
    return unit(result) if normalize else result


def graph_evidence(face_groups, voice_groups, face_index, voice_index, pseudocount=.5):
    """Leave-one-unique-sample-pair-out, smoothed log observed/independent count.

    Repeated copies of the exact same face/audio edge have a single vote. The
    current unique edge is removed completely before scoring its group pair.
    Thus a singleton edge supplies no positive evidence for its own label.
    """
    unique_edges = np.unique(np.column_stack([face_index, voice_index]), axis=0)
    nf, nv = int(face_groups.max())+1, int(voice_groups.max())+1
    counts = np.zeros((nf, nv), dtype=np.float64)
    np.add.at(counts, (face_groups[unique_edges[:, 0]], voice_groups[unique_edges[:, 1]]), 1)
    a, b = face_groups[face_index], voice_groups[voice_index]
    observed = counts[a, b]-1
    face_degree = counts.sum(1)[a]-1
    voice_degree = counts.sum(0)[b]-1
    expected = face_degree*voice_degree/max(len(unique_edges)-1, 1)
    evidence = np.log((observed+pseudocount)/(expected+pseudocount))
    return evidence.astype(np.float32)


def score(mapped, face_groups, voice_groups, face_index, voice_index, no_gender,
          pool_weight, graph_weight, pseudocount=.5, independent_scores=None):
    """Mapped members: (face_vectors, voice_vectors, gender_f, gender_v, weight)."""
    face_index = np.asarray(face_index, dtype=np.int64)
    voice_index = np.asarray(voice_index, dtype=np.int64)
    no_gender = np.asarray(no_gender, dtype=bool)
    if not (len(face_index) == len(voice_index) == len(no_gender)):
        raise ValueError('Trial arrays differ in length')
    scores = []
    for f, v, gf, gv, gender_weight in mapped:
        baseline = np.einsum('ij,ij->i', f[face_index], v[voice_index])
        baseline += no_gender*gender_weight*np.einsum('ij,ij->i', gf[face_index], gv[voice_index])
        if pool_weight:
            pf, pv = means(f, face_groups, True), means(v, voice_groups, True)
            pgf, pgv = means(gf, face_groups), means(gv, voice_groups)
            a, b = face_groups[face_index], voice_groups[voice_index]
            pooled = np.einsum('ij,ij->i', pf[a], pv[b])
            pooled += no_gender*gender_weight*np.einsum('ij,ij->i', pgf[a], pgv[b])
            baseline = (1-pool_weight)*baseline+pool_weight*pooled
        scores.append(baseline)
    result = np.mean(scores, axis=0)
    if independent_scores is not None:
        independent_scores = np.asarray(independent_scores, dtype=np.float32)
        if independent_scores.shape != result.shape:
            raise ValueError('Independent trial baseline has the wrong shape')
        # Keep supplied per-trial official features for the independent term.
        # Content-identical audio may have slightly different official CSV rows.
        canonical = []
        for f, v, gf, gv, weight in mapped:
            s = np.einsum('ij,ij->i', f[face_index], v[voice_index])
            s += no_gender*weight*np.einsum('ij,ij->i', gf[face_index], gv[voice_index])
            canonical.append(s)
        result += (1-pool_weight)*(independent_scores-np.mean(canonical, axis=0))
    if graph_weight:
        result += graph_weight*graph_evidence(face_groups, voice_groups, face_index, voice_index, pseudocount)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite transductive scores')
    return result
