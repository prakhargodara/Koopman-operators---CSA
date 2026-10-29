
import numpy as np


def logistic(x, k):
    return k * x * (1.0 - x)


def generate_time_series(map_fn, param, x0, n_steps):
    
    ts = np.empty(n_steps)
    ts[0] = x0
    for t in range(1, n_steps):
        ts[t] = map_fn(ts[t - 1], param)
    return ts
