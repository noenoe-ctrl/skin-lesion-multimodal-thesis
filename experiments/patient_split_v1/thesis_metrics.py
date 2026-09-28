"""Common prediction metrics and fast weighted patient-bootstrap statistics."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from protocol import CLASSES


def target(frame, task):
    return frame.label.to_numpy(dtype=int) if task == 'binary' else frame.diagnostic.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(dtype=int)


def prediction_frame(frame, prob, task):
    out = frame[['img_id','patient_id','lesion_id','diagnostic']].copy()
    out['target'] = target(frame, task)
    out['prediction'] = (prob >= .5).astype(int) if task == 'binary' else prob.argmax(axis=1)
    if task == 'binary':
        out['prob_malignant'] = prob
    else:
        for i,c in enumerate(CLASSES):
            out[f'prob_{c}'] = prob[:,i]
    return out


def probabilities(frame, task):
    return frame.prob_malignant.to_numpy() if task == 'binary' else frame[[f'prob_{c}' for c in CLASSES]].to_numpy()


def evaluate_probabilities(y, prob, task):
    binary = task == 'binary'
    pred = (prob >= .5).astype(int) if binary else prob.argmax(axis=1)
    avg = 'binary' if binary else 'macro'
    onehot = np.eye(2 if binary else 6)[y]
    full = np.column_stack([1-prob, prob]) if binary else prob
    assert np.isfinite(full).all() and np.all((full >= 0) & (full <= 1))
    np.testing.assert_allclose(full.sum(axis=1), 1, atol=1e-6)
    cm = confusion_matrix(y, pred, labels=np.arange(full.shape[1]))
    metric = {'auc': float(roc_auc_score(y, prob) if binary else roc_auc_score(y, prob, multi_class='ovr', average='macro')),
              'accuracy': float(accuracy_score(y,pred)), 'balanced_accuracy': float(balanced_accuracy_score(y,pred)),
              'precision': float(precision_score(y,pred,average=avg,zero_division=0)),
              'recall': float(recall_score(y,pred,average=avg,zero_division=0)),
              'f1': float(f1_score(y,pred,average=avg,zero_division=0)),
              'average_precision': float(average_precision_score(y,prob) if binary else average_precision_score(onehot,full,average='macro')),
              'brier': float(np.mean((prob-y)**2) if binary else np.mean(np.sum((full-onehot)**2,axis=1))),
              'confusion_matrix': cm.tolist(), 'class_recall': [float(cm[i,i]/cm[i].sum()) for i in range(len(cm))]}
    if binary:
        metric['specificity'] = float(cm[0,0]/cm[0].sum())
        confidence = prob; observed = y
        metric['calibration_definition'] = 'malignant probability vs malignant frequency'
    else:
        confidence = full.max(axis=1); observed = (pred == y).astype(float)
        metric['calibration_definition'] = 'top-label confidence vs accuracy'
    bins = np.minimum((confidence*10).astype(int),9)
    calibration = []
    ece = 0.
    for i in range(10):
        mask = bins == i
        if mask.any():
            p = float(confidence[mask].mean()); o = float(observed[mask].mean())
            ece += float(mask.mean()) * abs(p-o)
            calibration.append({'bin':i, 'count':int(mask.sum()), 'mean_probability':p, 'observed':o})
        else:
            calibration.append({'bin':i, 'count':0, 'mean_probability':None, 'observed':None})
    metric['ece_10_bins'] = ece; metric['calibration_bins'] = calibration
    return metric


def weighted_primary(y, prob, task, weights):
    """Each row of weights is a bootstrap draw; exact weighted ROC-AUC/BACC."""
    weights = np.asarray(weights, float)
    if task != 'binary':
        pred = prob.argmax(axis=1)
        return np.mean([weights[:,(y==c)&(pred==c)].sum(axis=1)/weights[:,y==c].sum(axis=1) for c in range(6)],axis=0)
    order = np.argsort(prob, kind='stable')
    scores = prob[order]; truths = y[order]; w = weights[:,order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores))+1]
    pos = np.add.reduceat(w*(truths==1), starts, axis=1)
    neg = np.add.reduceat(w*(truths==0), starts, axis=1)
    below = np.cumsum(neg,axis=1)-neg
    return np.sum(pos*(below+.5*neg),axis=1)/(pos.sum(axis=1)*neg.sum(axis=1))
