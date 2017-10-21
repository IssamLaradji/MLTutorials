import sys
import argparse

import numpy as np

def f(x):
    return x**3

def g(x):
    return 3*x**2

def forward_diff(x, f):
    # Approximates gradient
    eps = 1e-4
    g_approx = f(x + eps) - f(x)
    g_approx /= eps

    return g_approx

if __name__ == "__main__":
    x = 3
    f(x)
    import pdb;pdb.set_trace()