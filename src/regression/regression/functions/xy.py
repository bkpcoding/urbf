import numpy
import torch
def xy(X, Y):
    Z = torch.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            Z[i, j] = X[i]*Y[j]
    return Z