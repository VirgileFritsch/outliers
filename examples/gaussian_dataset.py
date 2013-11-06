import numpy as np
from outliers import (
    EllipticEnvelope, EllipticEnvelopeNaive,
    EllipticEnvelopeRMCDl2, EllipticEnvelopeRMCDl1, EllipticEnvelopeRMCDRP)
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Parameters
contamination = 0.1
pvalue_correction = "fwer"

rng = np.random.RandomState(0)

# Get data
X = rng.randn(500, 2)
Xs = [rng.randn(500, 2) for i in range(42)]

# Define "classifiers" to be used
classifiers = {
    "Empirical Covariance": EllipticEnvelopeNaive(
        contamination=contamination, pvalue_correction=pvalue_correction),
    "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(
        contamination=contamination, pvalue_correction=pvalue_correction),
    "RMCDl2": EllipticEnvelopeRMCDl2(
        contamination=contamination, pvalue_correction=pvalue_correction),
    "RMCDl1": EllipticEnvelopeRMCDl1(
         contamination=contamination, pvalue_correction=pvalue_correction),
    "RMCDRP": EllipticEnvelopeRMCDRP(
        contamination=contamination, pvalue_correction=pvalue_correction)
    }
colors = ['r', 'm', 'g', 'b', 'black']
legend = {}

Zj = []
# Learn a frontier for outlier detection with several classifiers
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
for i, (clf_name, clf) in enumerate(classifiers.iteritems()):
    plt.figure(1)
    clf.fit(X)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Zj.append(Z.copy())
    legend[clf_name] = plt.contour(
        xx, yy, Z, levels=[0], linewidths=2, colors=colors[i])

# Plot the results (= shape of the data points cloud)
plt.figure(1)
plt.title("original dataset")
plt.scatter(X[:, 0], X[:, 1], color='black')
plt.xlim((xx.min(), xx.max()))
plt.ylim((yy.min(), yy.max() + 3))  # "+ 3" for legend
plt.legend(
    [legend.values()[i].collections[0] for i in range(len(classifiers))],
    [legend.keys()[i] for i in range(len(classifiers))], loc="upper center",
    prop=matplotlib.font_manager.FontProperties(size=12), frameon=False)

fig = plt.figure()
for i in range(42):
    splt = fig.add_subplot(6, 7, i + 1)
    splt.scatter(Xs[i][:, 0], Xs[i][:, 1], color='black')
    for j, (clf_name, clf) in enumerate(classifiers.iteritems()):
        # if np.sum(clf.predict(Xs[i]) == 1) > 0:
        #     splt.set_axis_bgcolor('gray')
        splt.contour(xx, yy, Zj[j], levels=[0], linewidths=2, colors=colors[j])
    splt.set_xlim((xx.min(), xx.max()))
    splt.set_ylim((yy.min(), yy.max()))

plt.subplots_adjust(0.02, 0.03, 0.99, 0.99)
plt.show()
