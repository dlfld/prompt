from typing import List, Dict

import torch
from sklearn import metrics


def get_prf(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
        计算prf值
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :return prf值 结构为map key为 recall、f1、precision、accuracy
    """
    res = dict({
        "recall": 0,
        "f1": 0,
        "precision": 0,
        "acc": 0
    })
    res["recall"] = metrics.recall_score(y_true, y_pred, average='weighted')
    res["f1"] = metrics.f1_score(y_true, y_pred, average='weighted')
    res["precision"] = metrics.precision_score(y_true, y_pred, average='weighted')
    res["acc"] = metrics.accuracy_score(y_true,y_pred)
    return res


import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and self.patience == 5:
                self.early_stop = True
        else:
            self.best_score = score
            if len(self.save_path) > 1:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # path = os.path.join(self.save_path, 'best_network.pth')
        path = self.save_path
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
