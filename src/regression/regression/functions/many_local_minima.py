import numpy as np

# see https://www.sfu.ca/~ssurjano/ackley.html for more information about the function
def ackley_function(x, y):
    a = 2
    b = 1
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = (
                - a * np.exp(-b * np.sqrt(0.5 * (x[i] ** 2 + y[j] ** 2)))
                - np.exp(0.5 * (np.cos(2 * np.pi * x[i]) + np.cos(2 * np.pi * y[j])))
                + np.e
                + a
            )
    return z
