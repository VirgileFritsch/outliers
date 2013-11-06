"""
Own implementation of an algorithm to compute the MCD estimator.
It relies on several random initialisations which can be chosen according
to various sampling schemes.

"""
# Author : Virgile Fritsch, <virgile.fritsch@inria.fr>, 2012

import numpy as np
import scipy as sp
from scipy import linalg
from sklearn.utils.extmath import pinvh, fast_logdet
from sklearn.covariance import empirical_covariance, EmpiricalCovariance
import warnings


###############################################################################
### Minimum Covariance Determinant
#   Implementing of an algorithm by Rousseeuw & Van Driessen described in
#   (A Fast Algorithm for the Minimum Covariance Determinant Estimator,
#   1999, American Statistical Association and the American Society
#   for Quality, TECHNOMETRICS)
###############################################################################
def c_step(X, h, objective_function, initial_estimates, verbose=False,
           cov_computation_method=empirical_covariance):
    """C_step procedure described in [1] aiming at computing the MCD

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
      Data set in which we look for the h observations whose scatter matrix
      has minimum determinant
    h: int, > n_samples / 2
      Number of observations to compute the ribust estimates of location
      and covariance from.
    remaining_iterations: int
      Number of iterations to perform.
      According to Rousseeuw [1], two iterations are sufficient to get close
      to the minimum, and we never need more than 30 to reach convergence.
    initial_estimates: 2-tuple
      Initial estimates of location and shape from which to run the c_step
      procedure:
      - initial_estimates[0]: an initial location estimate
      - initial_estimates[1]: an initial covariance estimate
    verbose: boolean
      Verbose mode

    Returns
    -------
    location: array-like, shape (n_features,)
      Robust location estimates
    covariance: array-like, shape (n_features, n_features)
      Robust covariance estimates
    support: array-like, shape (n_samples,)
      A mask for the `h` observations whose scatter matrix has minimum
      determinant

    Notes
    -----
    References:
    [1] A Fast Algorithm for the Minimum Covariance Determinant Estimator,
        1999, American Statistical Association and the American Society
        for Quality, TECHNOMETRICS

    """
    n_samples, n_features = X.shape
    n_iter = 30
    remaining_iterations = 30

    # Get initial robust estimates from the function parameters
    location = initial_estimates[0]
    covariance = initial_estimates[1]
    # run a special iteration for that case (to get an initial support)
    precision = pinvh(covariance)
    X_centered = X - location
    dist = (np.dot(X_centered, precision) * X_centered).sum(1)
    # compute new estimates
    support = np.zeros(n_samples).astype(bool)
    support[np.argsort(dist)[:h]] = True
    location = X[support].mean(0)
    covariance = cov_computation_method(X[support])
    previous_obj = np.inf

    # Iterative procedure for Minimum Covariance Determinant computation
    obj = objective_function(X[support], location, covariance)
    while (obj < previous_obj) and (remaining_iterations > 0):
        # save old estimates values
        previous_location = location
        previous_covariance = covariance
        previous_obj = obj
        previous_support = support
        # compute a new support from the full data set mahalanobis distances
        precision = pinvh(covariance)
        X_centered = X - location
        dist = (np.dot(X_centered, precision) * X_centered).sum(1)
        # compute new estimates
        support = np.zeros(n_samples).astype(bool)
        support[np.argsort(dist)[:h]] = True
        location = X[support].mean(axis=0)
        covariance = cov_computation_method(X[support])
        obj = objective_function(X[support], location, covariance)
        # update remaining iterations for early stopping
        remaining_iterations -= 1

    # Catch computation errors
    if np.isinf(obj):
        raise ValueError(
            "Singular covariance matrix. "
            "Please check that the covariance matrix corresponding "
            "to the dataset is full rank and that MCD is used with "
            "Gaussian-distributed data (or at least data drawn from a "
            "unimodal, symetric distribution.")
    # Check convergence
    if np.allclose(obj, previous_obj):
        # c_step procedure converged
        if verbose:
            print "Optimal couple (location, covariance) found before" \
                "ending iterations (%d left)" % (remaining_iterations)
        results = location, covariance, obj, support
    elif obj > previous_obj:
        # objective function has increased (should not happen)
        current_iter = n_iter - remaining_iterations
        warnings.warn("Warning! obj > previous_obj (%.15f > %.15f, iter=%d)" \
                          % (obj, previous_obj, current_iter), RuntimeWarning)
        results = previous_location, previous_covariance, \
            previous_obj, previous_support

    # Check early stopping
    if remaining_iterations == 0:
        if verbose:
            print 'Maximum number of iterations reached'
        obj = fast_logdet(covariance)
        results = location, covariance, obj, support

    return results


def select_candidates(X, h, objective_function, verbose=False,
                      cov_computation_method=empirical_covariance):
    """Finds the best pure subset of observations to compute MCD from it.

    The purpose of this function is to find the best sets of h
    observations with respect to a minimization of their covariance
    matrix determinant. Equivalently, it removes n_samples-h
    observations to construct what we call a pure data set (i.e. not
    containing outliers). The list of the observations of the pure
    data set is referred to as the `support`.

    Starting from a support estimated with a Parzen density estimator,
    the pure data set is found by the c_step procedure introduced by
    Rousseeuw and Van Driessen in [1].

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
      Data (sub)set in which we look for the h purest observations
    h: int, [(n + p + 1)/2] < h < n
      The number of samples the pure data set must contain.
    select: int, int > 0
      Number of best candidates results to return.

    See
    ---
    `c_step` function

    Returns
    -------
    best_locations: array-like, shape (select, n_features)
      The `select` location estimates computed from the `select` best
      supports found in the data set (`X`)
    best_covariances: array-like, shape (select, n_features, n_features)
      The `select` covariance estimates computed from the `select`
      best supports found in the data set (`X`)
    best_supports: array-like, shape (select, n_samples)
      The `select` best supports found in the data set (`X`)

    Notes
    -----
    References:
    [1] A Fast Algorithm for the Minimum Covariance Determinant Estimator,
        1999, American Statistical Association and the American Society
        for Quality, TECHNOMETRICS

    """
    n_samples, n_features = X.shape

    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.svm import OneClassSVM
    pairwise_distances = np.ravel(euclidean_distances(X))
    delta = sp.stats.scoreatpercentile(pairwise_distances, 10)
    gamma = 0.01 / delta
    clf = OneClassSVM(kernel='rbf', gamma=gamma)
    clf.fit(X)
    in_support = np.argsort(
        -np.ravel(clf.decision_function(X)))[-(n_samples / 2):]
    support = np.zeros(n_samples, dtype=bool)
    support[in_support] = True
    location = X[support].mean(0)
    covariance = cov_computation_method(X[support])
    initial_estimates = (location, covariance)
    best_location, best_covariance, _, best_support = c_step(
        X, h, objective_function, initial_estimates, verbose=verbose,
        cov_computation_method=cov_computation_method)

    return best_location, best_covariance, best_support


def fast_mcd(X, objective_function, h=None,
             cov_computation_method=empirical_covariance):
    X = np.asanyarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. " \
                          "You may want to reshape your data array")
        n_samples = 1
        n_features = X.size
    else:
        n_samples, n_features = X.shape

    # minimum breakdown value
    if h is None:
        h = int(np.ceil(0.5 * (n_samples + n_features + 1)))
    else:
        h = int(h * n_samples)

    # 1-dimensional case quick computation
    # (Rousseeuw, P. J. and Leroy, A. M. (2005) References, in Robust
    #  Regression and Outlier Detection, John Wiley & Sons, chapter 4)
    if n_features == 1:
        # find the sample shortest halves
        X_sorted = np.sort(np.ravel(X))
        diff = X_sorted[h:] - X_sorted[:(n_samples - h)]
        halves_start = np.where(diff == np.min(diff))[0]
        # take the middle points' mean to get the robust location estimate
        location = 0.5 * \
            (X_sorted[h + halves_start] + X_sorted[halves_start]).mean()
        support = np.zeros(n_samples).astype(bool)
        support[np.argsort(np.abs(X - location), axis=0)[:h]] = True
        covariance = np.asarray([[np.var(X[support])]])
        location = np.array([location])
    else:
        location, covariance, support = select_candidates(
            X, h, objective_function,
            cov_computation_method=cov_computation_method)

    return location, covariance, support


class MCD(EmpiricalCovariance):
    """sklearn's MinCovDet + correction methods.

    """
    _nonrobust_covariance = staticmethod(empirical_covariance)

    def __init__(self, store_precision=True, assume_centered=False, h=None,
                 correction=None):
        EmpiricalCovariance.__init__(
            self, store_precision=store_precision,
            assume_centered=assume_centered)
        self.h = h
        self.correction = correction

    def fit(self, X, y=None):
        """Fits a Minimum Covariance Determinant with the FastMCD algorithm.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples
          and n_features is the number of features.
        y: not used, present for API consistence purpose.

        Returns
        -------
        self: object
          Returns self.

        """
        n_samples, n_features = X.shape
        # check that the empirical covariance is full rank
        if (linalg.svdvals(np.dot(X.T, X)) > 1e-8).sum() != n_features:
            warnings.warn("The covariance matrix associated to your dataset "
                          "is not full rank")
        # compute and store raw estimates
        raw_location, raw_covariance, raw_support = fast_mcd(
            X, objective_function=self.objective_function,
            h=self.h, cov_computation_method=self._nonrobust_covariance)
        if self.h is None:
            self.h = int(np.ceil(0.5 * (n_samples + n_features + 1))) \
                / float(n_samples)
        if self.assume_centered:
            raw_location = np.zeros(n_features)
            raw_covariance = self._nonrobust_covariance(
                X[raw_support], assume_centered=True)
        # get precision matrix in an optimized way
        precision = pinvh(raw_covariance)
        raw_dist = np.sum(np.dot(X, precision) * X, 1)
        self.raw_location_ = raw_location
        self.raw_covariance_ = raw_covariance
        self.raw_support_ = raw_support
        self.location_ = raw_location
        self.support_ = raw_support
        self.dist_ = raw_dist
        # obtain consistency at normal models
        self.correct_covariance(X)

        return self

    def correct_covariance(self, data, method=None):
        """Apply a correction to raw Minimum Covariance Determinant estimates.

        Correction using the empirical correction factor suggested
        by Rousseeuw and Van Driessen in [Rouseeuw1984]_.

        Parameters
        ----------
        data: array-like, shape (n_samples, n_features)
          The data matrix, with p features and n samples.
          The data set must be the one which was used to compute
          the raw estimates.

        Returns
        -------
        covariance_corrected: array-like, shape (n_features, n_features)
          Corrected robust covariance estimate.

        """
        if method is "empirical":
            X_c = data - self.raw_location_
            dist = np.sum(
                np.dot(X_c, pinvh(self.raw_covariance_)) * X_c, 1)
            correction = np.median(dist) / sp.stats.chi2(
                data.shape[1]).isf(0.5)
            covariance_corrected = self.raw_covariance_ * correction
        elif method is "theoretical":
            n, p = data.shape
            c = sp.stats.chi2(p + 2).cdf(sp.stats.chi2(p).ppf(self.h)) / self.h
            covariance_corrected = self.raw_covariance_ * c
        else:
            covariance_corrected = self.raw_covariance_
        self._set_covariance(covariance_corrected)

        return covariance_corrected

    def objective_function(self, data, location, covariance):
        """
        """
        det = fast_logdet(covariance)
        return det


def pison_correction(n, p):
    """

    """
    repeat = 100
    pth_roots = np.zeros(repeat)
    for i in range(repeat):
        print i
        data = np.dot(np.random.randn(n, p), np.eye(p))
        mcd = MCD(h=None).fit(data)
        covariance = mcd.raw_covariance_
        pth_roots[i] = np.exp(fast_logdet(covariance))

    res_inv = (1. / repeat) * np.sum(pth_roots ** (1. / p))

    return 1. / res_inv
