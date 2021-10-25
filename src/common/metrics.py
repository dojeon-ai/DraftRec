import numpy as np
from sklearn.metrics import ndcg_score


def precision_at_k(y_true, y_score, k, reduction=True):
    """
    :param y_score: type:(np.array), shape:(n,d), desc: predicted relevance
    :param y_true: type: (np.array), shape:(n,d), desc: ground-truth relevance
    :param k: type:(int), desc:number of retrieved items
    :return: pr@k
    """
    y_true = np.sign(y_true)
    n_positives = np.sum(y_true, 1)
    assert np.sum(n_positives > 0) == len(y_true)

    order = np.argsort(y_score, 1)[:, ::-1]
    y_true = np.take_along_axis(y_true, order[:, :k], 1)
    n_relevant = np.sum(y_true, 1)

    # Todo: check which one is the standard practice if k is larger than the possible relevant items
    # pr = n_relevant / min(n_positives, k)
    pr = (n_relevant / k)
    if reduction:
        return np.mean(pr)
    else:
        return pr


def average_precision_at_k(y_true, y_score, k, reduction=True):
    """
    equivalent to reciprocal-rank if there exists one ground truth item

    :param y_score: type:(np.array), shape:(n,d), desc: predicted relevance
    :param y_true: type: (np.array), shape:(n,d), desc: ground-truth relevance
    :param k: type:(int), desc:number of retrieved items
    :return: AP@k or RR@k
    """
    y_true = np.sign(y_true)
    n_positives = np.sum(y_true, 1)
    assert np.sum(n_positives > 0) == len(y_true)

    order = np.argsort(y_score, 1)[:, ::-1]
    y_true = np.take_along_axis(y_true, order[:, :k], 1)

    PR = np.zeros(len(y_true))
    cnt = np.zeros(len(y_true))
    for idx in range(1, k+1):
        n_relevant = np.sum(y_true[:, :idx], 1)
        pr = (n_relevant / idx)
        PR += y_true[:, idx-1] * pr
        cnt += y_true[:, idx-1]

    eps = 1e-45
    ap = PR / (cnt+eps)
    if reduction:
        return np.mean(ap)
    else:
        return ap


def recall_at_k(y_true, y_score, k, reduction=True):
    """
    equivalent to hit-ratio if there exists one ground truth item

    :param y_score: type:(np.array), shape:(n,d), desc: predicted relevance
    :param y_true: type: (np.array), shape:(n,d), desc: ground-truth relevance
    :param k: type:(int), desc:number of retrieved items
    :return: Recall@k or HR@k
    """
    y_true = np.sign(y_true)
    n_positives = np.sum(y_true, 1)
    assert np.sum(n_positives > 0) == len(y_true)

    order = np.argsort(y_score, 1)[:, ::-1]
    y_true = np.take_along_axis(y_true, order[:, :k], 1)
    n_relevant = np.sum(y_true, 1)

    recall = n_relevant / n_positives
    if reduction:
        return np.mean(recall)
    else:
        return recall


def ndcg_at_k(y_true, y_score, k, reduction=True):
    """
    :param y_score: type:(np.array), shape:(n,d), desc: predicted relevance
    :param y_true: type: (np.array), shape:(n,d), desc: ground-truth relevance
    :param k: type:(int), desc:number of retrieved items
    :return: NDCG@k
    """
    if reduction:
        return ndcg_score(y_true, y_score, k=k)
    else:
        raise NotImplementedError
        

def get_recommendation_metrics_for_ks(y_score, y_true, ks=[]):
    metrics = {}
    cnt = len(y_score)
    for k in ks:
        metrics['HR@' + str(k)] = (recall_at_k(y_true, y_score, k), cnt)
        metrics['NDCG@' + str(k)] = (ndcg_at_k(y_true, y_score, k), cnt)
    
    return metrics

def get_win_prediction_metrics(y_score, y_true, is_draft_finished):
    """
    :param y_score: type:(np.array), shape:(n,), desc: predicted win probability
    :param y_true: type: (np.array), shape:(n,), desc: ground-truth win-rate
    :param is_draft_finished: type: (np.array), shape:(n,)
    :return metrics: {}
    """
    metrics = {}
    cnt = len(y_true)
    metrics['ACC'] = (np.mean((y_score >= 0.5) == y_true), cnt)
    metrics['MAE'] = (np.mean(np.abs(y_score - y_true)), cnt)
    metrics['MSE'] = (np.mean(np.square(y_score - y_true)), cnt)

    cnt = np.sum(is_draft_finished)
    y_score = y_score[is_draft_finished]
    y_true = y_true[is_draft_finished]
    metrics['FIN_ACC'] = (np.mean((y_score >= 0.5) == y_true), cnt)
    metrics['FIN_MAE'] = (np.mean(np.abs(y_score - y_true)), cnt)
    metrics['FIN_MSE'] = (np.mean(np.square(y_score - y_true)), cnt)
        
    return metrics

if __name__=='__main__':
    # [1. Test with a single interaction]
    y_true = np.array([[0, 1, 0]])
    y_score = np.array([[0.7, 0.5, 0.2]])

    assert precision_at_k(y_true, y_score, k=1) == (0/1)
    assert precision_at_k(y_true, y_score, k=2) == (1/2)
    assert precision_at_k(y_true, y_score, k=3) == (1/3)
    # recall@k = hr@k
    assert recall_at_k(y_true, y_score, k=1) == (0/1)
    assert recall_at_k(y_true, y_score, k=2) == (1/1)
    assert recall_at_k(y_true, y_score, k=3) == (1/1)
    # ap@k = rr@k
    assert average_precision_at_k(y_true, y_score, k=1) == 0
    assert average_precision_at_k(y_true, y_score, k=2) == 1/2
    assert average_precision_at_k(y_true, y_score, k=3) == 1/2

    # ndcg@k = dcg@k / idcg@k
    assert abs(ndcg_at_k(y_true, y_score, k=1) - (0/np.log(2)) / (1/np.log(2))) < 1e-5
    assert abs(ndcg_at_k(y_true, y_score, k=2) - (1/np.log(3)) / (1/np.log(2))) < 1e-5
    assert abs(ndcg_at_k(y_true, y_score, k=3) - (1/np.log(3)) / (1/np.log(2))) < 1e-5

    # [2. Test with a multiple interactions]
    y_true = np.array([[0.8, 0.0, 0.0, 1.0]])
    y_score = np.array([[0.3, 0.5, 0.2, 0.7]])

    assert precision_at_k(y_true, y_score, k=1) == (1/1)
    assert precision_at_k(y_true, y_score, k=2) == (1/2)
    assert precision_at_k(y_true, y_score, k=3) == (2/3)
    assert precision_at_k(y_true, y_score, k=4) == (2/4)

    assert recall_at_k(y_true, y_score, k=1) == (1/2)
    assert recall_at_k(y_true, y_score, k=2) == (1/2)
    assert recall_at_k(y_true, y_score, k=3) == (2/2)
    assert recall_at_k(y_true, y_score, k=4) == (2/2)

    assert average_precision_at_k(y_true, y_score, k=1) == 1/1
    assert average_precision_at_k(y_true, y_score, k=2) == 1/1
    assert average_precision_at_k(y_true, y_score, k=3) == (1/1 + 2/3)/2
    assert average_precision_at_k(y_true, y_score, k=4) == (1/1 + 2/3)/2

    # [3. Test with a batch-wise input]
    y_true = np.array([[0.8,   0,   0,  1.0],
                       [0  ,   1,   0,    0]])
    y_score = np.array([[0.3, 0.5, 0.2, 0.7],
                        [0.7, 0.5, 0.2, 0.3]])

    assert precision_at_k(y_true, y_score, k=1) == (1/1 + 0/1)/2
    assert precision_at_k(y_true, y_score, k=2) == (1/2 + 1/2)/2
    assert precision_at_k(y_true, y_score, k=3) == (2/3 + 1/3)/2
    assert precision_at_k(y_true, y_score, k=4) == (2/4 + 1/4)/2

    assert recall_at_k(y_true, y_score, k=1) == (1/2 + 0/1)/2
    assert recall_at_k(y_true, y_score, k=2) == (1/2 + 1/1)/2
    assert recall_at_k(y_true, y_score, k=3) == (2/2 + 1/1)/2
    assert recall_at_k(y_true, y_score, k=4) == (2/2 + 1/1)/2

    assert average_precision_at_k(y_true, y_score, k=1) == (1/1)/2
    assert average_precision_at_k(y_true, y_score, k=2) == ((1/1) + (1/2))/2
    assert average_precision_at_k(y_true, y_score, k=3) == ((1/1 + 2/3)/2 + (1/2))/2
    assert average_precision_at_k(y_true, y_score, k=4) == ((1/1 + 2/3)/2 + (1/2))/2
