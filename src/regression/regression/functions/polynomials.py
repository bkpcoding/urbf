import numpy as np
import torch
import random
import itertools
def poly(x, reshape_param, config):
    # here n is the number of terms in my polynomial
    # for example n = 3 => polynomial = x^2*y + x*y^2 + x^3
    n = 5
    # I choose x to be a uniform samples in the range -2 to +2 with 10 points
    x = torch.tensor(x, dtype= torch.float32)
    x = torch.transpose(x, 0, 1)
    # m: degree of the polynomial
    # m = 4 => polynomial = x^2 * y^2 + x^4 + y^ 4 + ..  
    m = config.m
    # ll is my k x n matrix, where each column represents power of my variable
    # examples for the polynomial: x^2 * y^2 + x^4
    # ll would be
    # [2 4]
    # [2 0]
    ll = [[0]*n for _ in range(config.k)]
    for i in range(n):
        m  = config.m
        for j in range(config.k):
            rn = random.randint(0, m)
            ll[j][i] = rn
            m = m - rn
            if m == 0:
                break
    # here complicated is my datapoints matrix
    # it is initialized to (k, n, 10^k) zero matrix
    complicated = torch.zeros((config.k, n, reshape_param))
    for i in range(config.k):
        for j in range(n):
            complicated[i,j] = torch.float_power(x[i], torch.tensor(ll[i][j], dtype=torch.float32))
    # after raising each of the point to it's respective power defined in ll[i][j]
    # the columns are collapsed to have a 2dimensional matrix with [n, 10^k] dimension
    my_prod = torch.prod(complicated, 0)
    # after collapsing the matrix, all the terms belonging to the same polynomial are added
    # to give a matrix with a dimension of [1, number of datapoints(10^k)]
    my_sum = torch.sum(my_prod, 0)
    # small random noise is added (-0.1, 0.1)
    my_sum += (torch.rand(1) - 0.5) * 0.1

    return my_sum

def sin(input):
    return torch.sin(input)


def fixed_polynomial(x, config):
    random.seed(config.seed)
    # here n is the number of terms in my polynomial
    # for example n = 3 => polynomial = x^2*y + x*y^2 + x^3
    n = config.n
    # m is the degree of the polynomial
    m = config.m
    # ll is k x n matrix, where each column represents power of my variable
    ll = [[0]*n for _ in range(config.k)]
    for i in range(n):
        m  = config.m
        for j in range(config.k):
            rn = random.randint(0, m)
            if i % 2 == 0:
                ll[j][i] = rn
            else:
                ll[config.k - j - 1][i] = rn
            m = m - rn
            if m == 0:
                break
    # output is a input.shape[0] x 1 matrix
    output = torch.zeros((config.k, n, config.size))
    for i in range(config.k):
        for j in range(n):
            output[i, j] = torch.float_power(x[i], torch.tensor(ll[i][j], dtype=torch.float32))
    output = torch.prod(output, 0)
    output = torch.sum(output, 0)
    # add some random noise to the output
    #output += (torch.rand(1) - 0.5)*10
    return output
