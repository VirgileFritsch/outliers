"""
Local Component Analysis, from [1]

[1] Le Roux, N. and Bach Francis, Local Component Analysis, September 29, 2011,
arXiv:1109.0093v3

"""

import numpy as np
from scipy import linalg
from sklearn.covariance import empirical_covariance
from sklearn.utils.extmath import fast_logdet
from sklearn.metrics.pairwise import distance
from ..covariance import EllipticEnvelopeRMCDl2


class LCA():
    """
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X):
        """
        """
        rmcd = EllipticEnvelopeRMCDl2(shrinkage='lw')
        rmcd.set_optimal_shrinkage_amount(X, method="lw")
        self.shrinkage = rmcd.shrinkage
        self.support = X.copy()
        self._lca(X, regularization=self.shrinkage)
        return self

    def _lca(self, X, max_iter=100, regularization=0, tol=1e-10):
        """
        """
        n_samples, n_features = X.shape
        if regularization == np.inf:
            # Use identity matrix if Ledoit-Wolf shrinkage == 1
            print "/!\ use identity matrix"
            coeff = np.trace(empirical_covariance(X)) / float(n_features)
            self.cov_ = coeff * np.eye(n_features)
            prec_ = self.cov_
            # learn the kernel
            dist = np.zeros((n_samples, self.support.shape[0]))
            for i, x in enumerate(X):
                for j, t in enumerate(self.support):
                    dist[i, j] = distance.mahalanobis(x, t, prec_)
            self.kernel = np.exp(-.5 * dist)
            # decompose the kernel
            U, D, V = linalg.svd(self.kernel)
            self.U = U
            self.D = D
            return self.cov_

        # LCA algorithm starts
        cov_gauss = empirical_covariance(X)
        cov_gauss.flat[::n_features + 1] += regularization
        # EM loop
        # The last iteration is there to compute the final log-likelihood
        mean_loglike = -np.inf
        for l in xrange(max_iter + 1):
            xax = np.dot(X, np.dot(linalg.pinv(cov_gauss), X.T))
            dxax = np.diag(xax).reshape((-1, 1))

            logK = -.5 * (dxax + dxax.T - 2. * xax)
            # each datapoint cannot use itself
            logK.flat[::n_samples + 1] = -np.inf
            K = np.exp(logK)

            loglik1 = -.5 * fast_logdet(cov_gauss)
            loglik2 = np.log(np.sum(K)) - np.log(n_samples - 1)
            loglik3 = -n_features / (2. * np.log(2. * np.pi))
            loglike = loglik1 + loglik2 + loglik3
            old_mean_loglike = mean_loglike
            mean_loglike = np.mean(loglike)
            if self.verbose:
                print "\tIteration %d, loglike = %g" % (l, mean_loglike)

            if l < max_iter:
                if mean_loglike - old_mean_loglike < tol:
                    #print "Convergence reached (iteration %d)" % l
                    break
                # row-normalize the responsibilities
                B = K / np.sum(K, 1)
                Bsum = np.sum(B, 0) + np.sum(B, 1)
                cov_gauss = np.dot(X.T, np.dot(np.diag(Bsum) - B - B.T, X)) \
                    / float(n_samples)
                cov_gauss.flat[::n_features + 1] += regularization
        self.responsibilities = K
        self.cov_ = cov_gauss

        # learn the kernel for further decision/prediction
        prec_ = linalg.pinv(self.cov_)
        dist = np.zeros((n_samples, self.support.shape[0]))
        for i, x in enumerate(X):
            for j, t in enumerate(self.support):
                dist[i, j] = distance.mahalanobis(x, t, prec_)
        self.kernel = np.exp(-.5 * dist)
        # decompose the kernel
        U, D, V = linalg.svd(self.kernel)
        self.U = U
        self.D = D

        return cov_gauss

    def transform(self, X):
        """
        """
        #D, V = linalg.eig(self.cov_)
        #transform_matrix = np.dot(np.dot(V, np.sqrt(np.diag(1. / D))), V.T)
        #return (np.dot(transform_matrix, X.T).T).astype(np.float64)
        transform_matrix = linalg.pinv(linalg.cholesky(self.cov_))
        return (np.dot(transform_matrix, X.T).T).astype(np.float64)
