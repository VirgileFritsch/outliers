"""
Class + utility functions based on sklearn to perform advanced outlier
detection (state-of-the-art algorithms, advanced data manipulations).

All outlier detection methods in the outlier detection module
should inherit from this class.
For a given outlier detection object, one should:
1. fit data
2. (if prediction needed) set the threshold (set_threshold method)
3. call predict/decision_function

"""

import numpy as np
import scipy as sp
from sklearn.metrics import roc_curve, auc
from sklearn.covariance.outlier_detection import OutlierDetectionMixin


def resample_roc(fp, tp, precision):
    """Resample a ROC curve so that it can be averaged with others.

    Parameters
    ----------
    fp: 1d-array,
      False positives, = abscisse values of the ROC curve
    tp: 1d-array,
      True positives, = ordinate values of the ROC curve
    precision: float,
      The precision at which the ROC must be resampled, i.e. the step between
      two graduations of the abscisse axis.

    Returns
    -------
    new_fp: 1d-array,
      The new abscisse values, resampled at `precision`
    new_tp: 1d-array,
      The new ordinate values, resampled at `precision`

    """
    from scipy.interpolate import interp1d
    new_fp = np.linspace(0., 1., 1. / precision)[:, np.newaxis]
    if fp[0] != 0:
        fp = np.concatenate(([0], np.ravel(fp)))
        tp = np.concatenate(([0], np.ravel(tp)))
    f = interp1d(np.ravel(fp), np.ravel(tp), bounds_error=True, fill_value=0.)
    new_tp = f(new_fp)
    new_tp[np.isnan(new_tp)] = 0.  # because 'fill_value' does not work
    # new_tp_ind = new_fp - fp
    # new_tp_ind[new_tp_ind < 0] = 1.
    # new_tp_ind = np.argmax(new_tp_ind, 1)
    # new_tp = tp[new_tp_ind]

    return new_fp, new_tp


def compute_auc(fp, tp, end=1.):
    """Compute the Area Under Curve up to fp = `end`.

    Parameters
    ----------
    fp: 1d-array,
      False positives, = abscisse values of the ROC curve
    tp: 1d-array,
      True positives, = ordinate values of the ROC curve
    end: float, 0 <= end <= 1,
      The value up to which to compute the AUC.

    Returns
    -------
    area: float,
      The Area Under the ROC Curve from fp = 0. to fp = `end`.

    """
    auc_stop = np.amax(np.where(fp <= end)) + 1
    return auc(np.ravel(fp[:auc_stop]), np.ravel(tp[:auc_stop]))


def resample_recall(rec, pr, precision):
    """
    """
    aux_sort = np.argsort(rec)
    rec_ = rec.copy()[aux_sort]
    pr_ = pr.copy()[aux_sort]
    new_rec = np.linspace(0., 1., 1. / precision, endpoint=True)[:, np.newaxis]

    new_pr = np.zeros(new_rec.size)
    for i, r in enumerate(new_rec[::-1]):
        id_rec = np.amin(np.where(rec_ >= r)[0])
        loc = new_pr.size - i - 1
        new_pr[loc] = np.amax(np.concatenate((
                    pr_[id_rec:], new_pr[loc:])))

    return new_rec, new_pr


class AdvancedOutlierDetectionMixin(OutlierDetectionMixin):
    """Generic class for outlier detection algorithms.

    `contamination`: float, 0 <= contamination <= 1,
      The thought amount of contamination in the dataset, or the pvalue at
      which it should be thresholded.

    `pvalue_correction`: {"fwer" | "fpr" | "fdr"},
      Correction to be applied to the pvalue when thresholding.

    """
    def __init__(self, contamination=0.1, pvalue_correction="fwer"):
        OutlierDetectionMixin.__init__(self, contamination)
        self.pvalue_correction = pvalue_correction

    def correct_pvalue(self, n_tests):
        """Correct p-value according to a given number of samples

        Correction is made according to the object `contamination` and
        `pvalue_correction` parameters.

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
        if self.pvalue_correction == "fwer":
            corrected_pvalue = self.contamination / float(n_tests)
        elif self.pvalue_correction == "fdr":
            corrected_pvalue = self.contamination * (n_tests + 1) \
                / (2 * float(n_tests))
        else:
            corrected_pvalue = self.contamination

        return corrected_pvalue

    def set_threshold(self, X, method="simulation", precision=2000, n_jobs=1):
        """Wrapper of the methods used to set the threshold.

        Parameters
        ----------
        X: array-like, shape=(n_samples, n_features),
          Dataset used to set the threshold (is "simulation" or "fixed"
          methods are used).
        method: str, {"simulation" | "fixed" | "distribution"}
          Threshold setting method.
          "simulation": monte-carlo simulation of the decision function
                        distribution
          "fixed": a fixed proportion of outliers is found.
          "distribution": detection based on a theoretical knowledge of the
                          decision function distribution.
        precision: int,
          precision of the decision function tabulation
          ("simulation" method only).
        n_jobs: int,
          Number of maximum parallel jobs.

        Require the existence of self.threshold_from_distribution,
        self.threshold_from_simulations and self.decision_function.

        Return
        ------
        res_threshold: float,
          Threshold used within the decision function to give observations
          an "outlyingness" score.

        """
        n_samples, n_features = X.shape
        corrected_pvalue = self.correct_pvalue(n_samples)
        if method == "simulation":
            tabulated_distribution = self.threshold_from_simulations(
                X, precision=precision, n_jobs=n_jobs)
            threshold = sp.stats.scoreatpercentile(
                tabulated_distribution, 100. * (1. - corrected_pvalue))
        elif method == "distribution":
            threshold = self.threshold_from_distribution(n_samples, n_features)
        elif method == "fixed":
            values = self.decision_function(X, raw_values=True)
            threshold = sp.stats.scoreatpercentile(
                values, 100. * (1. - self.contamination))
        else:
            raise NotImplemented("Method '%s' not implemented yet" % method)
        self.threshold = threshold

        return threshold

    def roc_curve(self, X, ground_truth, precision=None):
        """Compute a ROC curve illustrating outlier detection accuracy.

        Parameters
        ----------
        X: array-like, shape=(n_samples, n_features),
          Dataset on which the accuracy of the method is assessed.
        ground_truth: array-like, shape=(n_samples,)
          ground_truth[i] is -1 if the i-th observation is an outlier,
          and 1 if it is not.

        Returns
        -------
        fp: 1d-array,
          False positives, = abscisse values of the ROC curve
        tp: 1d-array,
          True positives, = ordinate values of the ROC curve

        """
        # change ground truth values for compatibility with sklearn
        recoded_ground_truth = ground_truth.copy()
        recoded_ground_truth[recoded_ground_truth == 1] = 0
        recoded_ground_truth[recoded_ground_truth == -1] = 1
        fp, tp, _ = roc_curve(
            recoded_ground_truth, self.decision_function(X, raw_values=True))
        # optionaly resample ROC curve for further comparisons
        if precision is not None:
            fp, tp = resample_roc(fp, tp, precision)

        return fp, tp

    def predict(self, X):
        """
        """
        if self.threshold is None:
            # cannot compute decision function with threshold 0
            print "please set threshold first to obtain a prediction."
        is_inlier = -np.ones(X.shape[0], dtype=int)
        # compute decision with outliers corresponding to negative values
        decision = self.decision_function(X, raw_values=False)
        is_inlier[decision > 0] = 1
        return is_inlier

    def transform(self, X):
        """
        """
        return X[self.predict(X) == 1]
