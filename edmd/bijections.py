import numpy as np


def h_sin(x, theta):
    """h(x) = x + θ sin(πx),  |θ| < 1/π ≈ 0.318."""
    return x + theta * np.sin(np.pi * x)
