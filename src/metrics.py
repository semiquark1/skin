# metrics support

# common numerical and scientific libraries
#import numpy as np
#from numpy import exp, log
#from scipy.optimize import curve_fit
#import h5py
from sklearn.metrics import (balanced_accuracy_score,
        confusion_matrix, roc_curve, auc, log_loss, brier_score_loss,
        precision_recall_curve)

def eval_metrics(y_true, y_proba, threshold):
    eps = 1e-15
    # metrics
    filt_n = (y_true == 0)
    filt_p = (y_true == 1)
    # balanced log score
    log_n = log_loss(y_true[filt_n], y_proba[filt_n], 
            labels=(0,1), eps=eps)
    log_p = log_loss(y_true[filt_p], y_proba[filt_p], 
            labels=(0,1), eps=eps)
    BLogS = 0.5 * (log_n + log_p)
    # balanced Brier score
    brier_n = brier_score_loss(y_true[filt_n], y_proba[filt_n])
    brier_p = brier_score_loss(y_true[filt_p], y_proba[filt_p])
    BBS = 0.5 * (brier_n + brier_p)
    # AUC of ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    AUROC = auc(fpr, tpr)
    # AUC of precision-recall curve
    prec, recall, thresholds = precision_recall_curve(y_true, y_proba)
    AUPR = auc(recall, prec)
    # binary metrics, using opt_ba_thres
    y_pred = (y_proba > threshold)
    th_BA = balanced_accuracy_score(y_true, y_pred)
    cmat = confusion_matrix(y_true, y_pred)
    th_FNR = cmat[1,0] / (cmat[1,1] + cmat[1,0])
    th_FPR = cmat[0,1] / (cmat[0,0] + cmat[0,1])
    return dict(
            BLogS=BLogS,
            BBS=BBS,
            AUROC=AUROC,
            AUPR=AUPR,
            th_BA=th_BA,
            th_FNR=th_FNR,
            th_FPR=th_FPR,
            thres=threshold,
            )


class PrintStat:
    def __init__(self,
            h_label,
            h_data,
            h_label_format,
            h_data_format,
            label_format,
            data_format,
            indent=0,
            sep=' ',
            end='',
            ):
        self.label_format = label_format
        self.data_format = data_format
        self.indent = indent
        self.sep = sep
        self.end = end
        self.data = []
        print(' ' * indent, end='')
        print(h_label_format.format(h_label), end='')
        for i in range(len(h_data)):
            print(self.sep + h_data_format[i].format(h_data[i]), end='')
        print(self.end)

    def append(self, label, data, do_append=True):
        if do_append:
            self.data.append(data)
        if isinstance(data, dict):
            data = [data[k_] for k_ in self.headers]
        print(' ' * self.indent, end='')
        print(self.label_format.format(label), end=self.sep)
        print(self.sep.join([self.data_format[i_].format(data[i_])
                for i_ in range(len(data))]), end='')
        print(self.end)

    def print_mean(self):
        data = np.array(self.data).mean(axis=0)
        self.append('mean', data, do_append=False)

    def print_median(self):
        data = np.median(np.array(self.data), axis=0)
        self.append('median', data, do_append=False)


class PrintMetrics(PrintStat):
    headers = ['BLogS', 'BBS', 'AUROC', 'AUPR',
            'th_BA', 'th_FNR', 'th_FPR', 'thres']

    def __init__(self, **kwargs):
        super().__init__(
                'model', self.headers,
                '{:18s}', ['{:>6s}']*8,
                '{:18s}', ['{:6.4f}']*8,
                indent=2, 
                **kwargs)

# vim: set sw=4 sts=4 expandtab :
