import numpy as np
from outliers import (
    EllipticEnvelope, EllipticEnvelopeNaive,
    EllipticEnvelopeRMCDl2, EllipticEnvelopeRMCDl1, EllipticEnvelopeRMCDRP)
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.datasets import load_boston

# Parameters
contamination = 0.1
pvalue_correction = "fwer"

# Get data
X1 = load_boston()['data'][:, [8, 10]]  # two clusters
X2 = load_boston()['data'][:, [5, 12]]  # "banana"-shaped

# Define "classifiers" to be used
classifiers = {
    # "Empirical Covariance": EllipticEnvelopeNaive(
    #     contamination=contamination, pvalue_correction=pvalue_correction),
    # "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(
    #     contamination=contamination, pvalue_correction=pvalue_correction),
    "RMCDl2": EllipticEnvelopeRMCDl2(
        contamination=contamination, pvalue_correction=pvalue_correction),
    # "RMCDl1": EllipticEnvelopeRMCDl1(
    #     contamination=contamination, pvalue_correction=pvalue_correction),
    # "RMCDRP": EllipticEnvelopeRMCDRP(
    #     contamination=contamination, pvalue_correction=pvalue_correction)
    }
colors = ['r', 'm', 'g', 'b', 'black']
legend1 = {}
legend2 = {}

# Learn a frontier for outlier detection with several classifiers
xx1, yy1 = np.meshgrid(np.linspace(-8, 28, 500), np.linspace(3, 40, 500))
xx2, yy2 = np.meshgrid(np.linspace(3, 10, 500), np.linspace(-5, 45, 500))
for i, (clf_name, clf) in enumerate(classifiers.iteritems()):
    plt.figure(1)
    clf.fit(X1)
    Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    Z1 = Z1.reshape(xx1.shape)
    legend1[clf_name] = plt.contour(
        xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])
    if clf_name == "RMCDl2":
        supportX1 = clf.support_.copy()
        plt.figure(3)
        clf.plot_distribution(X1)
    plt.figure(2)
    clf.fit(X2)
    Z2 = clf.decision_function(np.c_[xx2.ravel(), yy2.ravel()])
    Z2 = Z2.reshape(xx2.shape)
    legend2[clf_name] = plt.contour(
        xx2, yy2, Z2, levels=[0], linewidths=2, colors=colors[i])
    if clf_name == "RMCDl2":
        supportX2 = clf.support_.copy()
        plt.figure(4)
        clf.plot_distribution(X2)

# Plot the results (= shape of the data points cloud)
plt.figure(1)  # two clusters
plt.title("Outlier detection on a real data set (boston housing)")
plt.scatter(X1[:, 0], X1[:, 1], color='black')
plt.scatter(X1[supportX1, 0], X1[supportX1, 1], color='r')
bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")
plt.annotate("several confounded points", xy=(24, 19),
             xycoords="data", textcoords="data",
             xytext=(13, 10), bbox=bbox_args, arrowprops=arrow_args)
plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
plt.legend(
    [legend1.values()[i].collections[0] for i in range(len(classifiers))],
    [legend1.keys()[i] for i in range(len(classifiers))],
    loc="upper center",
    prop=matplotlib.font_manager.FontProperties(size=12))
plt.ylabel("accessibility to radial highways")
plt.xlabel("pupil-teatcher ratio by town")

plt.figure(2)  # "banana" shape
plt.title("Outlier detection on a real data set (boston housing)")
plt.scatter(X2[:, 0], X2[:, 1], color='black')
plt.scatter(X2[supportX2, 0], X2[supportX2, 1], color='r')
plt.xlim((xx2.min(), xx2.max()))
plt.ylim((yy2.min(), yy2.max()))
plt.legend(
    [legend2.values()[i].collections[0] for i in range(len(classifiers))],
    [legend2.keys()[i] for i in range(len(classifiers))],
    loc="upper center",
    prop=matplotlib.font_manager.FontProperties(size=12))
plt.ylabel("% lower status of the population")
plt.xlabel("average number of rooms per dwelling")

plt.show()
