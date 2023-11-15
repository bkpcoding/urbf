import numpy as np
import matplotlib.pyplot as plt

# Define the number of Gaussians
m = 3

# Define the mean and covariance matrix for each Gaussian
mu_list = [np.array([1, 2]), np.array([-2, 3]), np.array([0, 0])]
cov_list = [np.array([[1, 0.5], [0.5, 1]]),
            np.array([[2, -1], [-1, 2]]),
            np.array([[0.5, 0], [0, 0.5]])]

# Define the range of the xy plane
x_min, x_max = -5, 5
y_min, y_max = -5, 5

# Create a grid of points for the xy plane
X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
XY = np.vstack([X.ravel(), Y.ravel()]).T

# Calculate the value of each Gaussian at each point in the grid
Z = np.zeros_like(X)
for mu, cov in zip(mu_list, cov_list):
    Z += np.exp(-0.5 * ((XY - mu) @ np.linalg.inv(cov) * (XY - mu)).sum(axis=1)).reshape(X.shape)

# Plot the xy plane and the Gaussian contours
fig, ax = plt.subplots()
ax.contour(X + Y, Z, levels=np.linspace(0, 1, 10 * m))
ax.set_xlim(x_min + y_min, x_max + y_max)
ax.set_ylim(y_min, y_max)
plt.show()
