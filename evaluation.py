import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import collate_fn
from sklearn.metrics import roc_auc_score


def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict


def evaluate_ood(in_scores, out_scores):
    keys = ['softmax', 'maha', 'cosine', 'energy']

    in_scores = merge_keys(in_scores, keys)

    out_scores = merge_keys(out_scores, keys)

    outputs = {}
    for key in keys:

        if key == "mix":
            auroc_in, aupr_in, fpr_95_in = get_measures(
                np.array([1 if i != 150 else 0 for i in in_scores[key]], dtype=np.float64),
                np.array([0 if i == 150 else 1 for i in out_scores[key]], dtype=np.float64))
            auroc_out, aupr_out, fpr_95_out = get_measures(
                np.array([1 if i == 150 else 0 for i in out_scores[key]], dtype=np.float64) * -1,
                np.array([0 if i != 150 else 1 for i in in_scores[key]], dtype=np.float64) * -1)

        else:
            auroc_in, aupr_in, fpr_95_in = get_measures(np.array(in_scores[key], dtype=np.float64),
                                                        np.array(out_scores[key], dtype=np.float64))
            # auroc_out, aupr_out, fpr_95_out = get_measures(np.array(out_scores[key], dtype=np.float64) * -1,
            #                                                np.array(in_scores[key], dtype=np.float64) * -1)

        # auroc, fpr_95_in, fpr_95_out = get_auroc(labels_in, scores), get_fpr_95(labels_in, scores), get_fpr_95(
        #     labels_out, scores)

        # outputs[tag + "_" + key + "_auroc_IN"] = auroc_in
        # outputs[tag + "_" + key + "_fpr95_IN"] = fpr_95_in
        # outputs[tag + "_" + key + "_aupr_IN"] = aupr_in
        outputs[key + "_auroc_IN"] = auroc_in
        outputs[key + "_fpr95_IN"] = fpr_95_in
        outputs[key + "_aupr_IN"] = aupr_in

    return outputs


def get_auroc(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    return roc_auc_score(new_key, prediction)


def get_fpr_95(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    score = fpr_and_fdr_at_recall(new_key, prediction)
    return score


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1.):
    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


import sklearn.metrics as sk


def get_measures(_pos, _neg):
    # pos = np.array(_pos[:]).reshape((-1, 1))
    # neg = np.array(_neg[:]).reshape((-1, 1))
    # examples = np.squeeze(np.vstack((pos, neg)))
    # labels = np.zeros(len(examples), dtype=np.int32)
    # labels[:len(pos)] += 1
    ins = _pos
    outs = _neg
    inl = np.ones_like(ins).astype(np.int64)
    outl = np.zeros_like(outs).astype(np.int64)
    scores = np.concatenate([ins, outs], axis=0)
    labels = np.concatenate([inl, outl], axis=0)

    auroc = get_auroc(labels, scores)
    aupr = sk.average_precision_score(labels, scores)
    fpr = get_fpr_95(labels, scores)

    return auroc, aupr, fpr
