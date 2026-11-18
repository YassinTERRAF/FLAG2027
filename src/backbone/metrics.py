import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve


def verification_metrics(labels, similarities):
    labels, scores = np.asarray(labels), np.asarray(similarities)
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Metrics need explicit binary labels and finite scores")
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return {"eer_percent": float(eer * 100), "auc": float(roc_auc_score(labels, scores)),
            "trials": len(labels), "positives": int(labels.sum())}


def submission_distance(similarity):
    # CodaBench's live Overview + official scorer negate submitted distances.
    return -np.asarray(similarity)
