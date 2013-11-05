"""
Local Component Analysis, from [1]

[1] Le Roux, N. and Bach Francis, Local Component Analysis, September 29, 2011,
arXiv:1109.0093v3

"""

import numpy as np
from scipy import linalg
from sklearn.utils.extmath import fast_logdet
from sklearn.metrics.pairwise import distance
from base import AdvancedOutlierDetectionMixin
from nonparametric_tools import LCA


class EnvelopeLCA(AdvancedOutlierDetectionMixin, LCA):
    """
    """
    def __init__(self, contamination=0.1, pvalue_correction="fwer",
                 verbose=False):
        AdvancedOutlierDetectionMixin.__init__(
            self, contamination=contamination,
            pvalue_correction=pvalue_correction)
        LCA(self, verbose=verbose)

    def decision_function(self, X, raw_values=True):
        """
        """
        n_features = self.cov_.shape[0]
        prec_ = linalg.pinv(self.cov_)
        dist = np.zeros((X.shape[0], self.support.shape[0]))
        for i, x in enumerate(X):
            for j, t in enumerate(self.support):
                dist[i, j] = distance.mahalanobis(x, t, prec_)
        a = fast_logdet(self.cov_)
        density = np.log(np.ravel(np.exp(-.5 * dist).mean(1))) \
            - 0.5 * a - (.5 * n_features) * np.log(2. * np.pi)
        return -density

    # def predict(self, X):
    #     """
    #     """
    #     if self.threshold is None:
    #         raise Exception("Please set a threshold (see help_predict)")
    #     return self.predict_with_l(X, self.threshold)

    # def predict_with_l(self, X, l):
    #     """
    #     """
    #     l = float(l)
    #     D = self.D
    #     U = self.U
    #     nonnul_mask = D > l
    #     Dl = np.zeros(D.size)
    #     Dl[nonnul_mask] = 1. - l / D[nonnul_mask]
    #     Zl = np.dot(U, (Dl * U).T)
    #     self.Zl = Zl
    #     decision = np.sum(Zl, 1)
    #     pred = np.ones(X.shape[0])
    #     pred[decision < 0.5] = -1
    #     return pred, decision
