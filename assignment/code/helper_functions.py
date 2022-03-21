import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, n=3) :
    steps = len(a)-n
    ma = np.full(steps, np.nan)
    for i in range(steps):
        ma[i] = np.mean(a[i:i+n])
    return ma, np.arange(steps)


def exponential_moving_average(array, alpha=0.001):
    """
    Calculate exponential moving average of an array
    """
    exponential_average = np.full(len(array), np.nan)
    exponential_average[0] = array[0]
    for i in range(1, len(array)):
        exponential_average[i] = alpha * array[i] + (1 - alpha) * exponential_average[i-1]
    return exponential_average, np.arange(len(array))
