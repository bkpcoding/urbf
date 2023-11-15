import numpy as np


def staircase(x, y, config):
    z = np.zeros((len(x), len(y)))
    #m = 1 => one discountinuity at x = 0
    if config.m == 1:
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] <= 0:
                    z[i, j] = x[i] + y[j]
                else:
                    z[i,j] = x[i] + y[j] + 8
    #m = 3 => three discountinuity at x = 0 or x > 0 and y < -1 or x < 0 and y > 1 
    if config.m == 3:
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] <= 0 and y[j] > 1:
                    z[i,j] = x[i] + y[j] - 4
                elif x[i] <= 0 and y[j] <= 1:
                    z[i, j] = x[i] + y[j] + 5
                elif x[i] > 0 and y[j] < -1:
                    z[i, j] = x[i] + y[j] + 7
                else:
                    z[i, j] = x[i] + y[j]

    if config.m == 5:
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] <= 0 and y[j] > 1:
                    z[i,j] = x[i] + y[j] - 4
                elif x[i] <= 0 and y[j] < 1 and y[j] > -2:
                    z[i, j] = x[i] + y[j] + 5
                elif x[i] <= 0 and y[j] <= -2:
                    z[i, j] = x[i] + y[j] + 12
                elif x[i] > 0 and y[j] < -1:
                    z[i, j] = x[i] + y[j] + 7
                elif x[i] > 0 and y[j] >= -1 and y[j] < 3:
                    z[i, j] = x[i] + y[j] - 2
                else:
                    z[i, j] = x[i] + y[j]
    if config.m == 0:
        for i in range(len(x)):
            for j in range(len(y)):
                z[i,j] = x[i] + y[j]   
    return z
