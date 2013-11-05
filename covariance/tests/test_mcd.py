import numpy as np

from sklearn.covariance import empirical_covariance

from robust_cov.covariance import MCD


def test_mcd():
    """Tests the FastMCD algorithm implementation

    """
    ### Small data set
    # test without outliers (random independent normal data)
    launch_mcd_on_dataset(100, 5, 0)
    # test with a contaminated data set (medium contamination)
    launch_mcd_on_dataset(100, 5, 20)
    # test with a contaminated data set (strong contamination)
    launch_mcd_on_dataset(100, 5, 40)

    ### Medium data set
    launch_mcd_on_dataset(1000, 5, 450)

    ### Large data set
    launch_mcd_on_dataset(1700, 5, 800)

    ### 1D data set
    launch_mcd_on_dataset(500, 1, 100)

    ### "High-dimensional" dataset
    #launch_mcd_on_dataset(30, 10, 10)


def launch_mcd_on_dataset(n_samples, n_features, n_outliers):

    rand_gen = np.random.RandomState(0)
    data = rand_gen.randn(n_samples, n_features)
    # add some outliers
    outliers_index = rand_gen.permutation(n_samples)[:n_outliers]
    outliers_offset = 10. * \
        (rand_gen.randint(2, size=(n_outliers, n_features)) - 0.5)
    data[outliers_index] += outliers_offset
    inliers_mask = np.ones(n_samples).astype(bool)
    inliers_mask[outliers_index] = False

    pure_data = data[inliers_mask]
    # compute MCD by fitting an object
    mcd_fit = MCD().fit(data)
    T = mcd_fit.location_
    S = mcd_fit.covariance_
    # compare with the estimates learnt from the inliers
    error_location = np.mean((pure_data.mean(0) - T) ** 2)
    print error_location
    assert(error_location < 1.)
    error_cov = np.mean((empirical_covariance(pure_data) - S) ** 2)
    print error_cov
    assert(error_cov < 1.)
