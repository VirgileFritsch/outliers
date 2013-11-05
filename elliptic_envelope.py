"""
Outlier detection with covariance-based tools.

"""

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, EmpiricalCovariance

from covariance import RMCDl2, RMCDl1
from base import AdvancedOutlierDetectionMixin
import parietal.outliers_detection.data_generation as dg


class CovarianceOutlierDetectionMixin(AdvancedOutlierDetectionMixin):
    """Generic class for covariance-based outlier detection algorithms.

    """
    def __init__(self, contamination=0.1, pvalue_correction="fwer"):
        AdvancedOutlierDetectionMixin.__init__(
            self, contamination=contamination,
            pvalue_correction=pvalue_correction)

    def threshold_from_contamination(self, X):
        """Threshold from a fixed contamination amount.

        Parameter
        ---------
        X: array-like, shape=(n_samples, n_features)
          Dataset on which to find outliers.

        """
        values = self.decision_function(X, raw_values=True)
        threshold = sp.stats.scoreatpercentile(
            values, 100. * (1. - self.contamination))
        return threshold

    def compute_pvalues(self, X, method="experimental"):
        """Compute pvalues associated to "is inlier" test.

        Parameters
        ----------
        X: array-like, shape={n_samples, n_features}
          The observations for which to compute p-values.
        method: str, in {"experimental", "theoretical"}
          The reference distribution to be compared to to asses p-values.
          "experimental" is a tabulated distribution.
          "theoretical" is a theoretical distribution.

        Returns
        -------
        pvalues: array-like, shape=(1, n_samples)
          The p-values.

        """
        mahal = self.mahalanobis(X)
        if method == "experimental":
            pvalues = np.ones(X.shape[0])
            for i in range(X.shape[0]):
                pvalues[i] = 1. - sp.stats.percentileofscore(
                    self.dist_out, mahal[i]) / 100.
        elif method == "theoretical":
            pvalues = self.outliers_distribution.sf(
                mahal / self.outliers_coeff)
        else:
            raise("Wrong argument")
        return pvalues


###############################################################################
def tabulate_inliers_expelldata(n_samples, n_features, sigma_root, clf_init):
            X1, _ = dg.generate_gaussian(
                n_samples, n_features, np.zeros(n_features),
                cov_root=sigma_root)
            # learn location and shape
            clf = ExperimentalEllipticData(
                correction=clf_init.correction,
                reweighting=clf_init.reweighting,
                h=clf_init.support_.sum() / float(n_samples),
                algorithm=clf_init.algorithm,
                nonparametric_support=clf_init.nonparametric_support).fit(X1)
            dist_in = clf.decision_function(X1[clf.support_], raw_values=True)
            h = clf.h
            return dist_in, h


def tabulate_outliers_expelldata(n_samples, n_features, sigma_root, clf):
            X1, _ = dg.generate_gaussian(
                n_samples, n_features, np.zeros(n_features),
                cov_root=sigma_root)
            # learn location and shape
            clf = ExperimentalEllipticData(
                correction=clf.correction, reweighting=clf.reweighting,
                h=clf.support_.sum() / float(n_samples),
                algorithm=clf.algorithm,
                nonparametric_support=clf.nonparametric_support).fit(X1)
            dist_out = clf.decision_function(
                X1[~clf.support_], raw_values=True)
            h = clf.h
            return dist_out, h


class ExperimentalEllipticData(RMCDl2,
                               CovarianceOutlierDetectionMixin):
    """Outlier detection with Minimum Covariance Determinant.

    """
    def __init__(self, store_precision=True, assume_centered=False,
                 h=None, correction="empirical", reweighting=None,
                 algorithm=None, contamination=0.1, pvalue_correction="fwer",
                 nonparametric_support=False, cov_computation_method=None):
        """

        Parameters
        ----------
        store_precision: bool
          Specify if the estimated precision is stored
        assume_centered: Boolean
          If True, data are not centered before computation.
          Useful when working with data whose mean is almost, but not exactly
          zero.
          If False, data are centered before computation.
        h: float, 0 < support_fraction <= 1
          The proportion of points to be included in the support of the raw
          MCD estimate. Default is None, which implies that the minimum
          value of support_fraction will be used within the algorithm:
          [n_sample + n_features + 1] / 2
        correction: str, in {"empirical", "theoretical"}
        reweighting: str, in {"rousseeuw", None}
        algorithm: str, or None
          Algorithm to be used for the MCD computation.
          If "fastmcd", the algorithm from Rousseeuw and Van Driessen is used.
          If None (default), minimization procedures are made from several
          random initialization, but no division in subsets is made as in
          Rousseeuw and Van Driessen.
        contamination: float, 0 <= contamination <= 1,
          The thought amount of contamination in the dataset, or the pvalue at
          which it should be thresholded.
        pvalue_correction: {"fwer" | "fpr" | "fdr"},
          Correction to be applied to the pvalue when thresholding.

        """
        RMCDl2.__init__(
            self, store_precision=store_precision,
            assume_centered=assume_centered, h=h, shrinkage=0,
            correction=correction, reweighting=reweighting,
            algorithm=algorithm, nonparametric_support=nonparametric_support,
            cov_computation_method=cov_computation_method)
        CovarianceOutlierDetectionMixin.__init__(
            self, contamination=contamination,
            pvalue_correction=pvalue_correction)

    def correct_pvalue(self, n_tests):
        """Correct p-value according to a given number of samples

        Correction is made according to the object `contamination` and
        `pvalue_correction` parameters.
        Since the detection is made only on observations not lying in the
        support of the MCD, this must be taken into account by adjusting the
        pvalue.

        Parameters
        ----------
        n_tests: int, n > 0
          Number of tests to be performed. Is used for family-wise corrections
          or FDR correction.

        Returns
        -------
        corrected_pvalue: float, 0 <= corrected_pvalue <= 1
          P-value corrected according to `contamination` and
          `pvalue_correction` parameters.

        """
        res = CovarianceOutlierDetectionMixin.correct_pvalue(self, n_tests)
        return res / (1 - self.h)

    def threshold_from_simulations(self, X, precision=2000, verbose=False,
                                   n_jobs=1):
        """
        """
        import multiprocessing as mp
        n_samples, n_features = X.shape
        #lw = LedoitWolf()
        #ref_covariance = lw.fit(X[self.support_]).covariance_
        # TEST
        #c = st.chi2(n_features + 2).cdf(
        #    st.chi2(n_features).ppf(
        #        float(self.h) / n_samples)) / (float(self.h) / n_samples)
        #ref_covariance = self.covariance_ * st.chi2(n_features).isf(0.5) \
        #    / np.median(self.mahalanobis(X))
        ref_covariance = self.covariance_  # * c
        # /TEST
        #sigma_root = np.linalg.cholesky(ref_covariance)
        D, V = np.linalg.eig(ref_covariance)
        sigma_root = np.dot(V, np.diag(np.sqrt(D))).astype(float)
        #sigma_root = np.eye(n_features)

        # inliers distribution
        #dist_in = np.array([], ndmin=1)
        max_i = max(1, int(precision / float(self.support_.sum())))
        dist_in = []
        all_h = []
        res = []
        if n_jobs == 1:
            for i in range(max_i):
                res.append(tabulate_inliers_expelldata(
                    n_samples, n_features, sigma_root, self))
        else:
            pool = mp.Pool(processes=n_jobs)
            results = []
            for i in range(max_i):
                results.append(pool.apply_async(
                    tabulate_inliers_expelldata,
                    args=(n_samples, n_features, sigma_root, self)))
            res = [r.get() for r in results]
            pool.close()
            pool.join()
        for r in res:
            dist_in_, all_h_ = r
            dist_in.append(dist_in_)
            all_h.append(all_h_)
        dist_in = np.ravel(dist_in)

        # outliers distribution
        #dist_out = np.array([], ndmin=1)
        max_i = max(1, int(precision / float(n_samples - self.support_.sum())))
        dist_out = []
        all_h2 = []
        res = []
        if n_jobs == 1:
            for i in range(max_i):
                res.append(tabulate_outliers_expelldata(
                    n_samples, n_features, sigma_root, self))
        else:
            pool = mp.Pool(processes=n_jobs)
            results = []
            for i in range(max_i):
                results.append(pool.apply_async(
                    tabulate_outliers_expelldata,
                    args=(n_samples, n_features, sigma_root, self)))
            res = [r.get() for r in results]
            pool.close()
            pool.join()
        for r in res:
            dist_out_, all_h_ = r
            dist_out.append(dist_out_)
            all_h2.append(all_h_)
        dist_out = np.ravel(dist_out)

        all_h = all_h + all_h2
        self.dist_in = np.sort(dist_in)
        self.dist_out = np.sort(dist_out)
        self.h_mean = np.mean(all_h)

        return self.dist_out

    def threshold_from_distribution(self, n_samples, n_features):
        """Theoretical outlier detection threshold on Mahalanobis distance.

        Parameters
        ----------
        n_samples: int,
          Number of samples
        n_features: int
          Number of features

        """
        n = n_samples
        p = n_features
        h = self.support_.sum() / float(n)
        H = self.support_.sum()

        # inliers distribution
        c = sp.stats.chi2(p + 2).cdf(sp.stats.chi2(p).ppf(h)) / h
        inliers_distribution = sp.stats.beta(p / 2., (H - p - 1) / 2.)
        inliers_coeff = ((H - 1) ** 2) / float(c * H)

        # outliers distribution
        alpha = 1. - h
        q_alpha = sp.stats.chi2(p).isf(alpha)
        c_alpha = (1. - alpha) / sp.stats.chi2(p + 2).cdf(q_alpha)
        c_2 = -sp.stats.chi2(p + 2).cdf(q_alpha) * .5
        c_3 = -sp.stats.chi2(p + 4).cdf(q_alpha) * .5    * (p + 4) / (p + 2)
        c_4 = 3. * c_3
        b_1 = c_alpha * (c_3 - c_4) / (1. - alpha)
        b_2 = .5 + (c_alpha / (1. - alpha)) \
            * (c_3 - ((q_alpha / p) * (c_2 + (.5 * (1. - alpha)))))
        v_1 = (1. - alpha) * (b_1 ** 2) \
            * (alpha * ((c_alpha * q_alpha / p) - 1.) ** 2 - 1.) \
            - (2. * c_3 * (c_alpha ** 2) \
                   * (3. * (b_1 - p * b_2) ** 2 \
                          + (p + 2) * b_2 * (2. * b_1 - p * b_2)))
        v_2 = n * (b_1 * (b_1 - p * b_2) * (1. - alpha)) ** 2
        v = v_1 / v_2
        m = 2. / v
        outliers_distribution = sp.stats.f(p, m - p + 1)
        outliers_coeff = (p * m) / (c * float(m - p + 1.))
        self.inliers_distribution = inliers_distribution
        self.inliers_coeff = inliers_coeff
        self.outliers_distribution = outliers_distribution
        self.outliers_coeff = outliers_coeff

        corrected_pvalue = self.correct_pvalue(n_samples)
        res = self.outliers_coeff * \
            self.outliers_distribution.isf(corrected_pvalue)
        return res

    def plot_distribution(self, method="experimental"):
        """Plot Mahalanobis distances distribution.

        Parameters
        ----------
        method: str, {"experimental", "theoretical"}
          Whether the distribution comes from theoretical knowledge
          of monte-carlo simulations.
        """
        if method == "experimental":
            plt.plot(
                np.linspace(
                    0., self.h_mean, self.dist_in.size, endpoint=False),
                self.dist_in, c='m', label="inliers")
            plt.plot(
                np.linspace(self.h_mean, 1., self.dist_out.size),
                self.dist_out, c='r', label="outliers")
            plt.hlines(self.threshold, plt.xlim()[0], plt.xlim()[1],
                       color='m', linestyles='dashed', label="threshold")
        elif method == "theoretical":
            x = np.linspace(0., self.h, self.support_.sum(), endpoint=False)
            plt.plot(x, self.inliers_coeff * self.inliers_distribution.ppf(x),
                     c='red')
            x = np.linspace(self.h, 1., np.sum(~self.support_))
            plt.plot(x,
                self.outliers_coeff * self.outliers_distribution.ppf(x), c='m')
            plt.hlines(self.threshold, plt.xlim()[0], plt.xlim()[1],
                       color='m', linestyles='dashed', label="threshold")
        else:
            raise NotImplemented("Wrong argument.")


###############################################################################
def tabulate_outliers_expnaivelldata(n_samples, n_features, sigma_root):
    X1, _ = dg.generate_gaussian(
        n_samples, n_features, np.zeros(n_features),
        cov_root=sigma_root)
    # learn location and shape
    clf = ExperimentalNaiveEllipticData().fit(X1)
    X2 = X1 - clf.location_
    dist_out = clf.decision_function(X2, raw_values=True)
    return dist_out


class ExperimentalNaiveEllipticData(
    EmpiricalCovariance, CovarianceOutlierDetectionMixin):
    """Outlier detection with empirical covariance.

    """
    def __init__(self, store_precision=True, assume_centered=False,
                 h=None, nonparametric_support=False, contamination=0.1,
                 pvalue_correction="fwer"):
        """
        """
        EmpiricalCovariance.__init__(
            self, store_precision=store_precision,
            assume_centered=assume_centered)
        CovarianceOutlierDetectionMixin.__init__(
            self, contamination=contamination,
            pvalue_correction=pvalue_correction)

    def threshold_from_simulations(self, X, precision=2000, verbose=False,
                                   n_jobs=1):
        """
        """
        import multiprocessing as mp
        n_samples, n_features = X.shape
        D, V = np.linalg.eig(self.covariance_)
        sigma_root = np.dot(V, np.diag(np.sqrt(D))).astype(float)
        dist_out = []
        max_i = max(1, 2 * int(np.ceil(precision / n_samples)))
        res = []
        if n_jobs == 1:
            for i in range(max_i):
                res.append(tabulate_outliers_expnaivelldata(
                        n_samples, n_features, sigma_root, self))
        else:
            pool = mp.Pool(processes=n_jobs)
            results = []
            for i in range(max_i):
                results.append(pool.apply_async(
                    tabulate_inliers_expelldata,
                    args=(n_samples, n_features, sigma_root, self)))
            res = [r.get() for r in results]
            pool.close()
            pool.join()
        for r in res:
            dist_out_ = r
            dist_out.append(dist_out_)
        self.dist_out = np.ravel(dist_out)

        return self.dist_out

    def threshold_from_distribution(self, n_samples, n_features):
        """Theoretical outlier detection threshold on Mahalanobis distance.

        Parameters
        ----------
        n_samples: int,
          Number of samples
        n_features: int
          Number of features

        """
        corrected_pvalue = self.correct_pvalue(n_samples)
        self.outliers_coeff = ((n_samples - 1) ** 2) / (float(n_samples))
        self.outliers_distribution = sp.stats.beta(
            n_features / 2., (n_samples - n_features - 1) / 2.)
        res_threshold = self.outliers_coeff \
            * self.outliers_distribution.isf(corrected_pvalue)
        return res_threshold

    def plot_distribution(self, method="experimental"):
        """Plot Mahalanobis distances distribution.
        """
        if method == "experimental":
            x = np.linspace(0., 1., self.dist_out.size)
            plt.plot(
                x, self.dist_out, c='m')
            plt.hlines(self.threshold, plt.xlim()[0], plt.xlim()[1],
                       color='m', linestyles='dashed', label="threshold")
        else:
            x = np.linspace(0., 1., 1000)
            plt.plot(
                x, self.outliers_coeff * self.outliers_distribution.ppf(x),
                c='m')
            plt.hlines(self.threshold, plt.xlim()[0], plt.xlim()[1],
                       color='m', linestyles='dashed', label="threshold")


###############################################################################
def tabulate_inliers_expregelldata(n_samples, n_features, sigma_root,
                                   clf_init):
    X1, _ = dg.generate_gaussian(
        n_samples, n_features, np.zeros(n_features),
        cov_root=sigma_root)
    # learn location and shape
    clf = ExperimentalRegularizedEllipticData(
        correction=clf_init.correction, reweighting=clf_init.reweighting,
        shrinkage=clf_init.shrinkage, algorithm=clf_init.algorithm,
        h=clf_init.support_.sum() / float(n_samples),
        nonparametric_support=clf_init.nonparametric_support,
        cov_computation_method=clf_init.cov_computation_method).fit(X1)
    dist_in = clf.decision_function(X1[clf.support_], raw_values=True)
    if clf_init.h != 1:
        dist_out = clf.decision_function(X1[~clf.support_], raw_values=True)
    else:
        dist_out = None
    h = clf.h
    return dist_in, dist_out, h


class ExperimentalRegularizedEllipticData(
    RMCDl2, CovarianceOutlierDetectionMixin):
    """Outlier detection with Regularized Minimum Covariance Determinant (l2).

    """
    def __init__(self, store_precision=True, assume_centered=False,
                 h=0.5, correction="empirical", reweighting=None,
                 shrinkage=None, nonparametric_support=False, algorithm=None,
                 contamination=0.1, pvalue_correction="fwer",
                 cov_computation_method=None):
        """
        """
        RMCDl2.__init__(
            self, store_precision=store_precision,
            assume_centered=assume_centered, h=h,
            correction=correction, reweighting=reweighting,
            algorithm=algorithm, shrinkage=shrinkage,
            nonparametric_support=nonparametric_support,
            cov_computation_method=cov_computation_method)
        CovarianceOutlierDetectionMixin.__init__(
            self, contamination=contamination,
            pvalue_correction=pvalue_correction)

    def fit(self, X, n_jobs=1):
        return RMCDl2.fit(self, X, n_jobs=n_jobs)

    def correct_pvalue(self, n_tests):
        """Correct p-value according to a given number of samples

        Correction is made according to the object `contamination` and
        `pvalue_correction` parameters.
        Since the detection is made only on observations not lying in the
        support of the MCD, this must be taken into account by adjusting the
        pvalue.

        Parameters
        ----------
        n_tests: int, n > 0
          Number of tests to be performed. Is used for family-wise corrections
          or FDR correction.

        Returns
        -------
        corrected_pvalue: float, 0 <= corrected_pvalue <= 1
          P-value corrected according to `contamination` and
          `pvalue_correction` parameters.

        """
        res = CovarianceOutlierDetectionMixin.correct_pvalue(self, n_tests)
        return res / (1 - self.h)

    def threshold_from_simulations(self, X, precision=1000, verbose=False,
                                   n_jobs=1):
        """
        """
        import multiprocessing as mp
        n_samples, n_features = X.shape
        n, p = X.shape
        h = self.support_.sum() / float(self.support_.size)
        # First learn a RMCD-CV to get a shape-preserving generative covariance
        rmcd_cv = ExperimentalRegularizedEllipticData(
            correction=self.correction, shrinkage="cv",
            h=self.support_.sum() / float(n_samples)).fit(X)
        self.alt_shrinkage = rmcd_cv.shrinkage
        self.alt_cov = rmcd_cv
        #lw = LedoitWolf()
        #ref_covariance = lw.fit(X[self.support_]).covariance_
        #c = st.chi2(p + 2).cdf(st.chi2(p).ppf(float(h) / n)) / (float(h) / n)
        #sigma_root = np.linalg.cholesky(ref_covariance / c)
        # TEST
        ref_covariance = rmcd_cv.covariance_  # * c
        """
        ref_covariance = self.covariance_
        """
        # /TEST
        #sigma_root = np.linalg.cholesky(ref_covariance)
        D, V = np.linalg.eig(ref_covariance)
        sigma_root = np.dot(V, np.diag(np.sqrt(D))).astype(float)
        #sigma_root = np.eye(n_features)
        all_h = []

        # observations distribution
        dist_in = []
        dist_out = []
        if self.support_.sum() == n_samples:
            max_i = max(1, int(precision / float(self.support_.sum())))
        else:
            max_i = max(
                1, int(precision / float(self.support_.sum())),
                int(precision / float(n_samples - self.support_.sum())))
        res = []
        if n_jobs == 1:
            for i in range(max_i):
                res.append(tabulate_inliers_expregelldata(
                    n_samples, n_features, sigma_root, self))
        else:
            pool = mp.Pool(processes=n_jobs)
            results = []
            for i in range(max_i):
                results.append(pool.apply_async(
                    tabulate_inliers_expelldata,
                    args=(n_samples, n_features, sigma_root, self)))
            res = [r.get() for r in results]
            pool.close()
            pool.join()
        for r in res:
            dist_in_, dist_out_, all_h_ = r
            dist_in.append(dist_in_)
            dist_out.append(dist_out_)
            all_h.append(all_h_)
        self.dist_in = np.ravel(dist_in)
        self.dist_out = np.ravel(dist_out)
        self.h_mean = np.mean(all_h)
        return self.dist_out

    def plot_distribution(self):
        """
        """
        plt.plot(np.linspace(0., self.h_mean, self.dist_in.size),
                 np.sort(self.dist_in), c='m')
        plt.plot(np.linspace(self.h_mean, 1., self.dist_out.size),
                 np.sort(self.dist_out), c='r')
        plt.hlines(self.threshold, plt.xlim()[0], plt.xlim()[1], color='m')


###############################################################################
class EllipticEnvelopeRMCDl1(
    RMCDl1, CovarianceOutlierDetectionMixin):
    """Outlier detection with Regularized Minimum Covariance Determinant (l1).
    """
    def __init__(self, store_precision=True, assume_centered=False,
                 h=None, correction="empirical", reweighting=None,
                 shrinkage=None, algorithm=None,
                 contamination=0.1, pvalue_correction="fwer",
                 nonparametric_support=False):
        """
        """
        RMCDl1.__init__(
            self, store_precision=store_precision,
            assume_centered=assume_centered,
            h=h, correction=correction, reweighting=reweighting,
            shrinkage=shrinkage, algorithm=algorithm,
            nonparametric_support=nonparametric_support)
        CovarianceOutlierDetectionMixin.__init__(
            self, contamination=contamination,
            pvalue_correction=pvalue_correction)

    def threshold_from_simulations(self, X, precision=2000, verbose=True,
                                   n_jobs=1):
        """
        """
        n_samples, n_features = X.shape
        n = n_samples
        p = n_features
        h = self.support_.sum()
        lw = LedoitWolf()
        ref_covariance = lw.fit(X[self.support_]).covariance_
        c = sp.stats.chi2(p + 2).cdf(
            sp.stats.chi2(p).ppf(float(h) / n)) / (float(h) / n)
        sigma_root = np.linalg.cholesky(ref_covariance / c)
        all_h = []

        # inliers distribution
        dist_in = np.array([], ndmin=1)
        max_i = max(1, int(precision / float(self.support_.sum())))
        for i in range(max_i):
            if verbose and max_i > 4 and (i % (max_i / 4) == 0):
                print "\t", 50 * i / float(max_i), "%"
            #sigma_root = np.diag(np.sqrt(eigenvalues))
            #sigma_root = np.eye(n_features)
            X1, _ = dg.generate_gaussian(
                n_samples, n_features, np.zeros(n_features),
                cov_root=sigma_root)
            # learn location and shape
            clf = EllipticEnvelopeRMCDl1(
                correction=self.correction, shrinkage=self.shrinkage,
                h=self.support_.sum() / float(n_samples)).fit(X1)
            X2 = X1 - clf.location_
            dist_in = np.concatenate(
                (dist_in, clf.decision_function(
                        X2[clf.support_], raw_values=True)))
            all_h.append(clf.h)

        # outliers distribution
        dist_out = np.array([], ndmin=1)
        max_i = max(1, int(precision / float(n_samples - self.support_.sum())))
        for i in range(max_i):
            if verbose and max_i > 4 and (i % (max_i / 4) == 0):
                print "\t", 50 * (1. + i / float(max_i)), "%"
            X1, _ = dg.generate_gaussian(
                n_samples, n_features, np.zeros(n_features),
                cov_root=sigma_root)
            # learn location and shape
            clf = EllipticEnvelopeRMCDl1(
                correction=self.correction, shrinkage=self.shrinkage,
                h=self.support_.sum() / float(n_samples)).fit(X1)
            X2 = X1 - clf.location_
            dist_out = np.concatenate(
                (dist_out, clf.decision_function(
                        X2[~clf.support_], raw_values=True)))
            all_h.append(clf.h)
        self.dist_in = np.sort(dist_in)
        self.dist_out = np.sort(dist_out)
        self.h_mean = np.mean(all_h)

        return self.dist_out

    def plot_distribution(self):
        """
        """
        plt.plot(
            np.linspace(0., self.h_mean, self.dist_in.size, endpoint=False),
            self.dist_in, c='red', label="inliers")
        plt.plot(np.linspace(self.h_mean, 1., self.dist_out.size),
                 self.dist_out, c='m', label="outliers")
        plt.hlines(self.threshold, plt.xlim()[0], plt.xlim()[1],
                   color='m', linestyles='dashed', label="threshold")


###############################################################################
def fit_on_projection(X, clf, n_features, n_proj_dim):
    mcd = ExperimentalEllipticData(
        correction=None,
        contamination=clf.contamination,
        pvalue_correction=clf.pvalue_correction)
    projector = np.linalg.svd(
        np.random.normal(size=(n_features, n_features)))[2]
    Xi = np.dot(X, projector[:, 0:(n_proj_dim + 1)])
    mcd.fit(Xi)
    mahal = mcd.mahalanobis(Xi)
    return mahal, mcd, projector[:, 0:(n_proj_dim + 1)]


class EllipticEnvelopeRMCDRP(AdvancedOutlierDetectionMixin):
    """
    """
    def __init__(self, contamination=0.1, pvalue_correction="fwer",
                 n_projections=None, n_proj_dim=None):
        AdvancedOutlierDetectionMixin.__init__(
            self, contamination=contamination,
            pvalue_correction=pvalue_correction)
        self.n_proj_dim = n_proj_dim
        self.n_projections = n_projections

    def fit(self, X, n_jobs=1):
        """
        """
        import multiprocessing as mp
        self.mcds = []
        self.projectors = []
        if X.shape[1] <= 10:
            actual_n_proj_dim = 1
        else:
            actual_n_proj_dim = self.n_proj_dim
        n_samples, n_features = X.shape
        if actual_n_proj_dim is None:
            actual_n_proj_dim = n_features
        if self.n_projections is None:
            self.n_projections = n_features / 5

        self.mahal = []
        self.mcds = []
        self.projectors = []
        res = []
        if n_jobs == 1:
            for i in xrange(self.n_projections):
                res.append(fit_on_projection(
                    X, self, n_features, actual_n_proj_dim))
        else:
            pool = mp.Pool(processes=n_jobs)
            results = []
            for i in xrange(self.n_projections):
                results.append(pool.apply_async(
                    fit_on_projection,
                    args=(X, self, n_features, actual_n_proj_dim)))
            res = [r.get() for r in results]
            pool.close()
            pool.join()
        for r in res:
            mahal_, mcd_, projector_ = r
            self.mahal.append(mahal_)
            self.mcds.append(mcd_)
            self.projectors.append(projector_)
        self.mahal = np.asarray(self.mahal)
        return self

    def correct_pvalue(self, n_tests):
        """Correct p-value according to a given number of samples

        Correction is made according to the object `contamination` and
        `pvalue_correction` parameters.
        Since the detection is made only on observations not lying in the
        support of the MCD, this must be taken into account by adjusting the
        pvalue.

        Parameters
        ----------
        n_tests: int, n > 0
          Number of tests to be performed. Is used for family-wise corrections
          or FDR correction.

        Returns
        -------
        corrected_pvalue: float, 0 <= corrected_pvalue <= 1
          P-value corrected according to `contamination` and
          `pvalue_correction` parameters.

        """
        res = AdvancedOutlierDetectionMixin.correct_pvalue(self, n_tests)
        return res / float(self.n_projections)

    def decision_function(self, X, raw_values=True):
        """Compute the decision function of the given observations.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)

        Returns
        -------
        decision: array-like, shape (n_samples, )
            The values of the decision function for each observations.

        """
        # Check that the threshold has been defined
        if not raw_values:
            if self.threshold is None:
                # cannot compute decision function with threshold 0
                print "please set threshold first to get adjusted decision."

        consensus = np.amax(self.mahal, 0)
        # optionaly modify the decision function so that the threshold is 0
        if not raw_values:
            consensus = consensus - self.threshold
        return consensus

    def predict(self, X, precision=2000, n_jobs=1):
        mcd = self.mcds[0]
        X0 = np.dot(X, self.projectors[0])
        mcd.set_threshold(
            X0, method="simulation", precision=precision, n_jobs=n_jobs)
        corrected_pvalue = self.correct_pvalue(X.shape[0])
        threshold = sp.stats.scoreatpercentile(
            mcd.dist_out, 100 * (1. - (corrected_pvalue / float(1. - mcd.h))))
        is_inlier = -np.ones(X.shape[0], dtype=int)
        is_inlier[np.amax(self.mahal, 0) < threshold] = 1
        return is_inlier

    def compute_pvalues(self, n_jobs=1):
        """
        """
        import multiprocessing as mp
        n_samples = self.mahal.shape[1]
        mcd = self.mcds[0]
        # pvalues = np.ones(n_samples)
        mahal_max = np.amax(self.mahal, 0)
        res = []
        if n_jobs == 1:
            for i in range(n_samples):
                res.append(sp.stats.percentileofscore(
                    mcd.dist_out, mahal_max[i]))
        else:
            pool = mp.Pool(processes=n_jobs)
            results = []
            for i in range(n_samples):
                results.append(pool.apply_async(
                        sp.stats.percentileofscore,
                        args=(mcd.dist_out, mahal_max[i])))
            res = [r.get() for r in results]
            pool.close()
            pool.join()
        pvalues = np.array(res)
        pvalues = 1. - pvalues / 100.
        return pvalues

    def threshold_from_distribution(self, X):
        """
        """
        return self.correct_pvalue(X.shape[0])

    def threshold_from_simulations(self, X, precision=2000):
        """
        """
        return self.correct_pvalue(X.shape[0]) * np.ones(precision)
