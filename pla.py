import numpy as np

def sign(s):
    if s >=0:
        return 1
    return -1
        
def pick_sample(xs, ys, N, w):
    """
    Pick a misclassified sample from training samples

    Args:
        xs: samples (numpy array of arrays, or matrix)
        ys: correspond sample labels (numpy array)
        N: number of samples
        w: weight (numpy array)
    Return:
        a misclassified sample and its label: (x, y) or None
    """
    for i in range(0, N):
        x = xs[i]
        y = ys[i]
        s = np.sum(w * x)
        if sign(s) != y:
            return (x, y)
        
    return None


def pla(xs, ys, N, start_w):
    """
    PLA learning algorithm

    Args:
        xs: samples (numpy array of arrays, or matrix)
        ys: correspond sample labels (numpy array)
        N: number of samples
    Return:
        final optimal weight
    """
    w = start_w
    while True:
        r = pick_sample(xs, ys, N, w)
        if r is None:
            return w
        x = r[0]
        y = r[1]
        w = w + y * x

POCKET_ITERATION = 1000

def in_sample_error(xs, ys, N,  w) -> float:
    error = 0
    for i in range(0, N):
        x = xs[i]
        y = ys[i]
        s = np.sum(w * x)
        if sign(s) != y :
            error += 1
    return error / N

def pocket(samples, N, start_w): 
    xs, ys = samples
    w = start_w
    final_weight = w
    current_err = in_sample_error(xs, ys, N, w)
    for _ in range (1, POCKET_ITERATION):
        x = pick_sample(xs, ys, N, w)
        if x is None:
            return final_weight
        w = pla(xs, ys, N, w)
        new_in_sample_err = in_sample_error(xs, ys, N, w)
        if new_in_sample_err < current_err:
            current_err = new_in_sample_err
            final_weight = w
        
    return final_weight