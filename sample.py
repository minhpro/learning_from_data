from pla import sign
import random
import numpy as np

VALUE_RANGE = 100

def generate_target_function(d):
    """
    Generate a target function w0 + w1x1 + ... + wdxd = 0
    Returns:
        - A numpy ndarray contains (w0, w1, .., wd)
    """
    line = np.empty(d+1)
    for i in range (d+1):
        line[i] = random.randrange(-VALUE_RANGE, VALUE_RANGE)
    return line

def generate_samples(d, line, size):
    """
    Generate the training samples

    Returns:
        A tuple (X, Y)
        X is a numpy ndarray contains samples
        Y is a numpy ndarray contains corresponding sample output 
    """
    samples = np.empty((size, d+1))
    lables = np.empty(size)
    for i in range(size):
        samples[i][0] = 1
        for j in range(1, d+1):
            samples[i][j] = random.randrange(-VALUE_RANGE, VALUE_RANGE)
        s = np.sum(line * samples[i])
        lables[i] = sign(s)
    return (samples, lables)
