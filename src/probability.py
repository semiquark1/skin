# support for calibrated probability

# standard lib
import sys

# common numerical and scientific libraries
import numpy as np
from numpy import log, exp
from scipy.optimize import curve_fit
import h5py
from sklearn.metrics import roc_curve

# local
sys.path.append('../src')

def sigma(x, k, x0):
    return 1/(1 + exp(-k*(x-x0)))

def inv_sigma(x, k, x0):
    xx = np.clip(x.astype(np.float64), 1e-16, 1-1e-16)
    return -(1/k) * log(1/xx - 1) + x0



class ComputeProbability:
    """Class for computing probability and indications from committee machine
    output."""

    def __init__(self, model=None):
        """Initialize ComputeProbability class instance

        Parameters:
        -----------
        model: multiple types:
            None: leave model uninitialized, use load_model_from_path() later.
            str: call load_model_from_path() with this argument.
        """
        self.k = None
        self.n_bins = None
        self.x0 = None
        self.w = None
        self.thres0 = None
        self.thres1 = None
        self.opt_ba_thres = None
        if isinstance(model, str):
            self.load_model_from_path(model)

    def set(self, k, n_bins, x0, w, thres0, thres1, opt_ba_thres):
        self.k = k
        self.n_bins = n_bins
        self.x0 = x0
        self.w = w
        self.thres0 = thres0
        self.thres1 = thres1
        self.opt_ba_thres = opt_ba_thres

    def load_model_from_path(self, path:str) -> None:
        """Load data from path

        Raises:
        -------
        ValueError
            If file does not exist, or data already loaded.
        """
        if self.k is None:
            try:
                with h5py.File(path, 'r') as h5f:
                    self.k = h5f.attrs['k']
                    self.n_bins = h5f.attrs['n_bins']
                    self.x0 = h5f.attrs['x0']
                    self.w = h5f.attrs['w']
                    self.thres0 = h5f.attrs['thres0']
                    self.thres1 = h5f.attrs['thres1']
                    self.opt_ba_thres = h5f.attrs['opt_ba_thres']
            except (FileNotFoundError, OSError):
                raise ValueError(f'Unknown model path: {path}')
        else:
            raise ValueError(f'Data already loaded.')

    def save(self, path):
        with h5py.File(path, 'w') as h5f:
            h5f.attrs['k'] = self.k
            h5f.attrs['n_bins'] = self.n_bins
            h5f.attrs['x0'] = self.x0 
            h5f.attrs['w'] = self.w
            h5f.attrs['thres0'] = self.thres0 if self.thres0 is not None else -1
            h5f.attrs['thres1'] = self.thres1 if self.thres1 is not None else -1
            h5f.attrs['opt_ba_thres'] = self.opt_ba_thres

    def predict(self, input_data) -> np.ndarray:
        """Predict probability

        Parameters:
        -----------
        input_data: np.ndarray with shape=(n_samples,), or list of floats.
            Sigmoid output of committee machine.

        Returns:
        --------
        probability: np.ndarray with shape=(n_samples,)
            Probability of melanoma (between 0 and 1)
        """
        assert self.k is not None
        input_data = np.array(input_data)
        assert input_data.ndim == 1, 'wrong input_data dimensions'
        assert input_data.size > 0, 'no samples'

        proxy = sigma(inv_sigma(input_data, self.k, 0), 1, 0)
        contbin = proxy * self.n_bins + 0.5
        proba = sigma(contbin, 1/self.w, self.x0)
        return proba


    def compute(self, input_data) -> np.ndarray:
        """Compute probability and indication

        Parameters:
        -----------
        input_data: np.ndarray with shape=(n_samples,), or list of floats.
            Sigmoid output of committee machine.

        Returns:
        --------
        (probability, indication)
            probability: Probability of melanoma (between 0 and 1)
            indication: chance of melanoma: 'low', 'medium', 'high'
        """
        assert self.k is not None
        input_data = np.array(input_data)
        assert input_data.ndim == 1, 'wrong input_data dimensions'
        assert input_data.size > 0, 'no samples'

        proxy = sigma(inv_sigma(input_data, self.k, 0), 1, 0)
        contbin = proxy * self.n_bins
        proba = sigma(contbin, 1/self.w, self.x0)
        #print('softmax, proxy, contbin, proba', input_data, proxy, contbin, proba)

        proba = proba[-1]       #

        if proba <= self.thres0:
            indication = 'low'
        elif proba <= self.thres1:
            indication = 'medium'
        else:
            indication = 'high'
        return proba, indication


def fit_proba(y_true, y_pred, beta, n_bins):
    """returns w, x0"""
    # augment to balanced data: oversample minority class (=1)
    n_major = (y_true == 0).sum()
    n_minor = (y_true == 1).sum()
    assert n_major > n_minor, 'expected minority class is 1'
    y_true_aug = np.array(list(y_true[y_true == 0])
            + (list(y_true[y_true == 1]) * int(np.ceil(n_major / n_minor)))
                [:n_major])
    y_pred_aug = np.array(list(y_pred[y_true == 0])
            + (list(y_pred[y_true == 1]) * int(np.ceil(n_major / n_minor)))
                [:n_major])
    assert (y_true_aug == 0).sum() == (y_true_aug == 1).sum()
    #    
    y_pred_aug_proxy = sigma(inv_sigma(y_pred_aug, 1, 0), beta, 0)

    y_pred_aug_proxy_bin = np.round(y_pred_aug_proxy * n_bins).astype(int)
    bin_proba = []
    bin_size = []
    for i in range(n_bins+1):
        selected = y_true_aug[y_pred_aug_proxy_bin == i]
        bin_size.append(len(selected))
        if len(selected):
            bin_proba.append(selected.sum() / len(selected))
        else:
            bin_proba.append(np.nan)
    bin_proba = np.array(bin_proba)
    bin_size = np.array(bin_size)
    filt = np.isfinite(bin_proba)
    bins = np.arange(n_bins + 1)
    bins_mid = np.array([bins.min(), bins.max()]).mean()
    p0 = (1, bins_mid)
    #yerr = 1 / np.sqrt(bin_size[filt])
    yerr = None
    popt, pcov = curve_fit(sigma, bins[filt], bin_proba[filt], p0, sigma=yerr)
    # w=1/k, x0, misc
    return 1/popt[0], popt[1], (y_pred_aug_proxy, bin_proba)

def get_model_proba(y_true, y, beta, n_bins, return_misc=False):
    w, x0, misc = fit_proba(y_true, y, beta, n_bins)
    model_proba = ComputeProbability()
    model_proba.k = 1./beta
    model_proba.n_bins = n_bins
    model_proba.w = w
    model_proba.x0 = x0
    y_proba = model_proba.predict(y)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    i = np.argmax(tpr - fpr)
    model_proba.opt_ba_thres = thresholds[i]
    if return_misc:
        return model_proba, misc
    return model_proba

# vim: set sw=4 sts=4 expandtab :
