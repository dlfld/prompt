from typing import List, Dict

from sklearn import metrics


def get_prf(y_true: List[str], y_pred: List[str]) -> Dict[str,float]:
    """
        计算prf值
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :return prf值 结构为map key为 recall、f1、precision、accuracy
    """
    res = dict({})
    res["recall"] = metrics.recall_score(y_true, y_pred, average='micro')
    res["f1"] = metrics.f1_score(y_true, y_pred, average='micro')
    res["precision"] = metrics.precision_score(y_true, y_pred, average='micro')
    res["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    return res
