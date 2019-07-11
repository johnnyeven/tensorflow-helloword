from sklearn.datasets import load_boston, make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

expected_x, expected_y = make_regression(n_samples=100, n_features=3, noise=0)
boston = load_boston()

print(boston.feature_names, boston.data[1, :], boston.target[1])

plt.plot()
plt.scatter(boston.data[:, 0], boston.target)
plt.scatter(boston.data[:, 1], boston.target)
plt.scatter(boston.data[:, 2], boston.target)
plt.scatter(boston.data[:, 3], boston.target)
plt.scatter(boston.data[:, 4], boston.target)
plt.scatter(boston.data[:, 5], boston.target)
plt.scatter(boston.data[:, 6], boston.target)
plt.scatter(boston.data[:, 7], boston.target)
plt.scatter(boston.data[:, 8], boston.target)
plt.scatter(boston.data[:, 9], boston.target)
plt.scatter(boston.data[:, 10], boston.target)
plt.scatter(boston.data[:, 11], boston.target)
plt.scatter(boston.data[:, 12], boston.target)
plt.legend(boston.feature_names)
plt.show()
