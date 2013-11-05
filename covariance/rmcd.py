"""
Robust, high-dimensional location and covariance estimators.

Here are implemented:
 * RMCD estimator objects with l1 or l2

"""
# Author: Virgile Fritsch <virgile.fritsch@inria.fr>
#
# License: BSD Style.
import numpy as np
from sklearn.utils.extmath import pinvh, fast_logdet
from sklearn.covariance import empirical_covariance, GraphLassoCV, graph_lasso

from mcd import fast_mcd, MCD


class RMCDl2(MCD):
    """Regularized MCD, robust and high-dimensional estimator of covariance.

    The Minimum Covariance Determinant estimator is a robust estimator
    of a data set's covariance introduced by P.J.Rousseuw in [1].
    The idea is to find a given proportion of "good" observations which
    are not outliers and compute their empirical covariance matrix.
    This empirical covariance matrix is then rescaled to compensate the
    performed selection of observations ("consistency step").

    Rousseeuw and Van Driessen [2] developed the FastMCD algorithm in order
    to compute the Minimum Covariance Determinant. This algorithm is used
    when fitting an MCD object to data.
    The FastMCD algorithm also computes a robust estimate of the data set
    location at the same time.

    Parameters
    ----------
    store_precision: bool
        Specify if the estimated precision is stored

    Attributes
    ----------
    `location_`: array-like, shape (n_features,)
        Estimated robust location

    `covariance_`: array-like, shape (n_features, n_features)
        Estimated robust covariance matrix

    `precision_`: array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    `support_`: array-like, shape (n_samples,)
        A mask of the observations that have been used to compute
        the robust estimates of location and shape.

    `correction`: str
        Improve the covariance estimator consistency at gaussian models
          - "empirical" (default): correction using the empirical correction
            factor suggested by Rousseeuw and Van Driessen in [2]
          - "theoretical": correction using the theoretical correction factor
            derived in [3]
          - else: no correction

    [1] P. J. Rousseeuw. Least median of squares regression. J. Am
        Stat Ass, 79:871, 1984.
    [2] A Fast Algorithm for the Minimum Covariance Determinant Estimator,
        1999, American Statistical Association and the American Society
        for Quality, TECHNOMETRICS
    [3] R. W. Butler, P. L. Davies and M. Jhun,
        Asymptotics For The Minimum Covariance Determinant Estimator,
        The Annals of Statistics, 1993, Vol. 21, No. 3, 1385-1400

    """
    def __init__(self, store_precision=True, assume_centered=False,
                 h=0.5, correction=None, shrinkage=None,
                 cov_computation_method=None):
        """
        assume_centered: Boolean
          If True, the support of robust location and covariance estimates
          is computed, and a covariance estimate is recomputed from it,
          without centering the data.
          Useful to work with data whose mean is significantly equal to
          zero but is not exactly zero.
          If False, the robust location and covariance are directly computed
          with the FastMCD algorithm without additional treatment.
        correction: str
          Improve the covariance estimator consistency at Gaussian models
            - "empirical" (default): correction using the empirical correction
              factor suggested by Rousseeuw and Van Driessen in [1]
            - "theoretical": correction using the theoretical correction factor
              derived in [2]
            - else: no correction
        shrinkage: float or {"cv", "oas", "lw"}
          The amount of shrinkage (ridge regularization) for ill-conditionned
          matrices.
          If shrinkage is in {"cv", "oas", "lw"}, the shrinkage amount
          will be selected according to the whosen criterion.

        """
        MCD.__init__(
            self, store_precision=store_precision,
            assume_centered=assume_centered, h=0.5, correction=correction)
        if shrinkage == "cv" or shrinkage is None:
            self.adapt_shrinkage = "cv"
            self.shrinkage = None
        elif shrinkage == "oas":
            self.adapt_shrinkage = "oas"
            self.shrinkage = None
        elif shrinkage == "lw":
            self.adapt_shrinkage = "lw"
            self.shrinkage = None
        else:
            self.adapt_shrinkage = False
            self.shrinkage = shrinkage
        self.h = h  # always use 50% observations with the RMCD
        self.cov_computation_method = cov_computation_method

    def fit(self, X, n_jobs=1):
        """Compute the Minimum Covariance Determinant estimate.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples
          and n_features is the number of features.

        Returns
        -------
        self : object
          Returns self.

        """
        n_samples, n_features = X.shape
        self.set_optimal_shrinkage_amount(X, method=self.adapt_shrinkage)
        if np.isinf(self.shrinkage):
            self.cov_computation_method = "diag"
            self.shrinkage = self.std_shrinkage * n_features
        # compute and store raw estimates
        raw_location, raw_covariance, raw_support = fast_mcd(
            X, objective_function=self.objective_function,
            h=self.h, cov_computation_method=self._nonrobust_covariance,)
        if self.h is None:
            self.h = raw_support.sum() / float(raw_support.size)
        if self.assume_centered:
            raw_location = np.zeros(n_features)
            raw_covariance = self._nonrobust_covariance(
                    X[raw_support], assume_centered=True)
        self.raw_location_ = raw_location
        self.raw_covariance_ = raw_covariance
        self.raw_support_ = raw_support
        self.location_ = raw_location
        self.support_ = raw_support
        # obtain consistency at normal models
        self.correct_covariance(X, method=self.correction)
        self._set_covariance(self.covariance_)
        return self

    def _nonrobust_covariance(self, data, assume_centered=False):
        """Non-robust estimation of the covariance to be used within MCD.

        Parameters
        ----------
        data: array_like, shape (n_samples, n_features)
          Data for which to compute the non-robust covariance matrix.
        assume_centered: Boolean
          Whether or not the observations should be considered as centered.

        Returns
        -------
        nonrobust_covariance: array_like, shape (n_features, n_features)
          The non-robust covariance of the data.

        """
        if self.cov_computation_method is None:
            cov = empirical_covariance(data, assume_centered=assume_centered)
            cov.flat[::data.shape[1] + 1] += self.shrinkage
        elif self.cov_computation_method == "diag":
            cov = np.diag(np.var(data, 0)) / self.shrinkage
        else:
            raise NotImplemented
        return cov

    def set_optimal_shrinkage_amount(self, X, method="cv", verbose=False):
        """Set optimal shrinkage amount according to chosen method.

        /!\ Could be rewritten with GridSearchCV.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples
          and n_features is the number of features.
        method: float or str in {"cv", "lw", "oas"},
          The method used to set the shrinkage. If a floating value is provided
          that value is used. Otherwise, the selection is made according to
          the selected method.
          "cv" (default): 10-fold cross-validation.
                          (or Leave-One Out cross-validation if n_samples < 10)
          "lw": Ledoit-Wolf criterion
          "oas": OAS criterion
        verbose: bool,
          Verbose mode or not.

        Returns
        -------
        optimal_shrinkage: float,
          The optimal amount of shrinkage.

        """
        n_samples, n_features = X.shape
        if isinstance(method, str):
            std_shrinkage = np.trace(empirical_covariance(X)) / \
                (n_features * n_samples)
            self.std_shrinkage = std_shrinkage
        if method == "cv":
            from sklearn.covariance import log_likelihood
            n_samples, n_features = X.shape
            shrinkage_range = np.concatenate((
                    [0.], 10. ** np.arange(-n_samples / n_features, -1, 0.5),
                    np.arange(0.05, 1., 0.05),
                    np.arange(1., 20., 1.), np.arange(20., 100, 5.),
                    10. ** np.arange(2, 7, 0.5)))
            # get a "pure" active set with a standard shrinkage
            active_set_estimator = RMCDl2(shrinkage=std_shrinkage)
            active_set_estimator.fit(X)
            active_set = np.where(active_set_estimator.support_)[0]
            # split this active set in ten parts
            active_set = active_set[np.random.permutation(active_set.size)]
            if active_set.size >= 10:
                # ten fold cross-validation
                n_folds = 10
                fold_size = active_set.size / 10
            else:
                n_folds = active_set.size
                fold_size = 1
            log_likelihoods = np.zeros((shrinkage_range.size, n_folds))
            if verbose:
                print "*** Cross-validation"
            for trial in range(n_folds):
                if verbose:
                    print trial / float(n_folds)
                # define train and test sets
                train_set_indices = np.concatenate(
                    (np.arange(0, fold_size * trial),
                     np.arange(fold_size * (trial + 1), n_folds * fold_size)))
                train_set = X[active_set[train_set_indices]]
                test_set = X[active_set[np.arange(
                            fold_size * trial, fold_size * (trial + 1))]]
                # learn location and covariance estimates from train set
                # for several amounts of shrinkage
                for i, shrinkage in enumerate(shrinkage_range):
                    location = test_set.mean(0)
                    cov = empirical_covariance(train_set)
                    cov.flat[::(n_features + 1)] += shrinkage * std_shrinkage
                    # compute test data likelihood
                    log_likelihoods[i, trial] = log_likelihood(
                       empirical_covariance(test_set - location,
                                            assume_centered=True), pinvh(cov))
            optimal_shrinkage = shrinkage_range[
                np.argmax(log_likelihoods.mean(1))]
            self.shrinkage = optimal_shrinkage * std_shrinkage
            self.shrinkage_cst = optimal_shrinkage
            if verbose:
                print "optimal shrinkage: %g (%g x lambda(= %g))" \
                    % (self.shrinkage, optimal_shrinkage, std_shrinkage)
            self.log_likelihoods = log_likelihoods
            self.shrinkage_range = shrinkage_range

            return shrinkage_range, log_likelihoods
        elif method == "oas":
            from sklearn.covariance import OAS
            rmcd = self.__init__(shrinkage=std_shrinkage)
            support = rmcd.fit(X).support_
            oas = OAS().fit(X[support])
            if oas.shrinkage_ == 1:
                self.shrinkage_cst = np.inf
            else:
                self.shrinkage_cst = oas.shrinkage_ / (1. - oas.shrinkage_)
            self.shrinkage = self.shrinkage_cst * std_shrinkage * n_features
        elif method == "lw":
            from sklearn.covariance import LedoitWolf
            rmcd = RMCDl2(self, h=self.h, shrinkage=std_shrinkage)
            support = rmcd.fit(X).support_
            lw = LedoitWolf().fit(X[support])
            if lw.shrinkage_ == 1:
                self.shrinkage_cst = np.inf
            else:
                self.shrinkage_cst = lw.shrinkage_ / (1. - lw.shrinkage_)
            self.shrinkage = self.shrinkage_cst * std_shrinkage * n_features
        else:
            pass
        return

    def objective_function(self, data, location, covariance):
        """Objective function minimized at each step of the MCD algorithm.
        """
        precision = pinvh(covariance)
        det = fast_logdet(precision)
        trace = np.trace(
            np.dot(empirical_covariance(data - location, assume_centered=True),
                   precision))
        pen = self.shrinkage * np.trace(precision)
        return -det + trace + pen


###############################################################################
class RMCDl1(MCD):
    """Regularized MCD, robust and high-dimensional estimator of covariance.

    The Minimum Covariance Determinant estimator is a robust estimator
    of a data set's covariance introduced by P.J.Rousseuw in [1].
    The idea is to find a given proportion of "good" observations which
    are not outliers and compute their empirical covariance matrix.
    This empirical covariance matrix is then rescaled to compensate the
    performed selection of observations ("consistency step").
    Having computed the Minimum Covariance Determinant estimator, one
    can give weights to observations according to their Mahalanobis
    distance, leading the a reweighted estimate of the covariance
    matrix of the data set.

    Rousseuw and Van Driessen [2] developed the FastMCD algorithm in order
    to compute the Minimum Covariance Determinant. This algorithm is used
    when fitting an MCD object to data.
    The FastMCD algorithm also computes a robust estimate of the data set
    location at the same time.

    Parameters
    ----------
    store_precision: bool
        Specify if the estimated precision is stored

    Attributes
    ----------
    `location_`: array-like, shape (n_features,)
        Estimated robust location

    `covariance_`: array-like, shape (n_features, n_features)
        Estimated robust covariance matrix

    `precision_`: array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    `support_`: array-like, shape (n_samples,)
        A mask of the observations that have been used to compute
        the robust estimates of location and shape.

    `correction`: str
        Improve the covariance estimator consistency at gaussian models
          - "empirical" (default): correction using the empirical correction
            factor suggested by Rousseeuw and Van Driessen in [2]
          - "theoretical": correction using the theoretical correction factor
            derived in [3]
          - else: no correction

    [1] P. J. Rousseeuw. Least median of squares regression. J. Am
        Stat Ass, 79:871, 1984.
    [2] A Fast Algorithm for the Minimum Covariance Determinant Estimator,
        1999, American Statistical Association and the American Society
        for Quality, TECHNOMETRICS
    [3] R. W. Butler, P. L. Davies and M. Jhun,
        Asymptotics For The Minimum Covariance Determinant Estimator,
        The Annals of Statistics, 1993, Vol. 21, No. 3, 1385-1400

    """
    def __init__(self, store_precision=True, assume_centered=False,
                 h=0.5, correction=None, shrinkage=None, contamination=0.1):
        """
        assume_centered: Boolean
          If True, the support of robust location and covariance estimates
          is computed, and a covariance estimate is recomputed from it,
          without centering the data.
          Useful to work with data whose mean is significantly equal to
          zero but is not exactly zero.
          If False, the robust location and covariance are directly computed
          with the FastMCD algorithm without additional treatment.
        correction: str
          Improve the covariance estimator consistency at Gaussian models
            - "empirical" (default): correction using the empirical correction
              factor suggested by Rousseeuw and Van Driessen in [1]
            - "theoretical": correction using the theoretical correction factor
              derived in [2]
            - else: no correction
        shrinkage: float
          The amount of shrinkage (ridge regularization) for ill-conditionned
          matrices.

        """
        MCD.__init__(
            self, store_precision=store_precision,
            assume_centered=assume_centered, h=0.5, correction=correction)
        if shrinkage is None:
            self.adapt_shrinkage = True
            self.shrinkage = None
        else:
            self.adapt_shrinkage = False
            self.shrinkage = shrinkage
        self.h = 0.5  # always use 50% observations with the RMCD

    def fit(self, X):
        """Fits a Minimum Covariance Determinant with the FastMCD algorithm.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples
          and n_features is the number of features.

        Returns
        -------
        self : object
            Returns self.

        """
        if self.adapt_shrinkage:
            self.set_optimal_shrinkage_amount(X)
        n_samples, n_features = X.shape
        # compute and store raw estimates
        raw_location, raw_covariance, raw_support = fast_mcd(
            X, objective_function=self.objective_function,
            h=self.h, cov_computation_method=self._nonrobust_covariance)
        if self.assume_centered:
            raw_location = np.zeros(n_features)
            raw_covariance = self._nonrobust_covariance(
                    X[raw_support], assume_centered=True)
        self.raw_location_ = raw_location
        self.raw_covariance_ = raw_covariance
        self.raw_support_ = raw_support
        self.location_ = raw_location
        self.support_ = raw_support
        self._set_covariance(raw_covariance)

        return self

    def _nonrobust_covariance(self, data, assume_centered=False):
        """Non-robust estimation of the covariance to be used within MCD.

        Parameters
        ----------
        data: array_like, shape (n_samples, n_features)
          Data for which to compute the non-robust covariance matrix.
        assume_centered: Boolean
          Whether or not the observations should be considered as centered.

        Returns
        -------
        nonrobust_covariance: array_like, shape (n_features, n_features)
          The non-robust covariance of the data.

        """
        try:
            cov, prec = graph_lasso(
                empirical_covariance(data, assume_centered=assume_centered),
                self.shrinkage)
        except:
            print " > Exception!"
            emp_cov = empirical_covariance(
                data, assume_centered=assume_centered)
            emp_cov.flat[::data.shape[1] + 1] += 1e-06
            cov, prec = graph_lasso(emp_cov, self.shrinkage)
        return cov

    def set_optimal_shrinkage_amount(self, X, verbose=False):
        """

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples
          and n_features is the number of features.

        Returns
        -------
        optimal_shrinkage: The optimal amount of shrinkage, chosen with a
        10-fold cross-validation. (or a Leave-One Out cross-validation
        if n_samples < 10).

        """
        n_samples, n_features = X.shape
        std_shrinkage = np.trace(empirical_covariance(X)) / \
            float(n_samples * n_features)
        # use L2 here? (was done during research work, changed for consistency)
        rmcd = RMCDl1(shrinkage=std_shrinkage).fit(X)
        cov = GraphLassoCV().fit(X[rmcd.raw_support_])
        self.shrinkage = cov.alpha_
        return cov.cv_alphas_, cov.cv_scores

    def objective_function(self, data, location, covariance):
        """Objective function minimized at each step of the MCD algorithm.
        """
        precision = pinvh(covariance)
        det = fast_logdet(precision)
        trace = np.trace(
            np.dot(empirical_covariance(data - location, assume_centered=True),
                   precision))
        pen = self.shrinkage * (np.sum(np.abs(precision))
                                 - np.abs(np.diag(precision)).sum())
        return -det + trace + pen
