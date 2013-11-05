import numpy as np
from sklearn.utils.extmath import pinvh

from robust_cov.covariance import RMCDl2, RMCDl1


def test_rmcdl2():
    """Tests the FastMCD algorithm implementation

    """
    ### Small data set
    # test without outliers (random independent normal data)
    launch_rmcdl2_on_dataset(100, 5, 0)
    # test with a contaminated data set (medium contamination)
    launch_rmcdl2_on_dataset(100, 5, 20)
    # test with a contaminated data set (strong contamination)
    launch_rmcdl2_on_dataset(100, 5, 40)

    ### Medium data set
    launch_rmcdl2_on_dataset(1000, 5, 450)

    ### Large data set
    launch_rmcdl2_on_dataset(1700, 5, 800)

    ### 1D data set
    #launch_rmcdl2_on_dataset(500, 1, 100)

    ### High-dimensional dataset
    launch_rmcdl2_on_dataset(30, 50, 10)


def launch_rmcdl2_on_dataset(n_samples, n_features, n_outliers):

    rand_gen = np.random.RandomState(0)
    data = rand_gen.randn(n_samples, n_features)
    # add some outliers
    outliers_index = rand_gen.permutation(n_samples)[:n_outliers]
    outliers_offset = 10. * \
        (rand_gen.randint(2, size=(n_outliers, n_features)) - 0.5)
    data[outliers_index] += outliers_offset
    inliers_mask = np.ones(n_samples).astype(bool)
    inliers_mask[outliers_index] = False

    # compute RMCD by fitting an object
    rmcd_fit = RMCDl2(shrinkage="lw").fit(data)
    T = rmcd_fit.location_
    S = rmcd_fit.covariance_
    # compare with the true location and precision
    error_location = np.mean(T ** 2)
    print error_location, rmcd_fit.shrinkage
    assert(error_location < 1.)
    error_cov = np.mean((np.eye(n_features) - pinvh(S)) ** 2)
    print error_cov
    assert(error_cov < 1.)


def test_rmcdl1():
    """Tests the FastMCD algorithm implementation

    """
    ### Small data set
    # test without outliers (random independent normal data)
    launch_rmcdl1_on_dataset(100, 5, 0)
    # test with a contaminated data set (medium contamination)
    launch_rmcdl1_on_dataset(100, 5, 20)
    # test with a contaminated data set (strong contamination)
    launch_rmcdl1_on_dataset(100, 5, 40)

    ### Medium data set
    launch_rmcdl1_on_dataset(1000, 5, 450)

    ### Large data set
    launch_rmcdl1_on_dataset(1700, 5, 800)

    ### 1D data set
    #launch_rmcdl1_on_dataset(500, 1, 100)

    ### High-dimensional dataset
    launch_rmcdl1_on_dataset(30, 20, 10)


def launch_rmcdl1_on_dataset(n_samples, n_features, n_outliers):

    rand_gen = np.random.RandomState(0)
    data = rand_gen.randn(n_samples, n_features)
    # add some outliers
    outliers_index = rand_gen.permutation(n_samples)[:n_outliers]
    outliers_offset = 10. * \
        (rand_gen.randint(2, size=(n_outliers, n_features)) - 0.5)
    data[outliers_index] += outliers_offset
    inliers_mask = np.ones(n_samples).astype(bool)
    inliers_mask[outliers_index] = False

    # compute RMCD by fitting an object
    rmcd_fit = RMCDl1().fit(data)
    T = rmcd_fit.location_
    S = rmcd_fit.covariance_
    # compare with the true location and precision
    error_location = np.mean(T ** 2)
    assert(error_location < 1.)
    error_cov = np.mean((np.eye(n_features) - pinvh(S)) ** 2)
    assert(error_cov < 1.)
