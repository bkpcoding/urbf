import numpy as np


def styblinski_function(x, y):
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = (
                x[i] ** 4
                - 16 * x[i] ** 2
                + 5 * x[i]
                + y[j] ** 4
                - 16 * y[j] ** 2
                + 5 * y[j]
            )
    return z
