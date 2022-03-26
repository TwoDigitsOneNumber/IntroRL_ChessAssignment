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
    ema = np.full(len(array), np.nan)
    ema[0] = array[0]
    for i in range(1, len(array)):
        ema[i] = alpha * array[i] + (1 - alpha) * ema[i-1]
    return ema, np.arange(len(array))


def save_avg_statistics(histories, method):
    # unpack histories
    R_histories = [history[0] for history in histories]
    N_moves_histories = [history[1] for history in histories]

    # turn into numpy arrays
    R_histories = np.hstack(R_histories)
    N_moves_histories = np.hstack(N_moves_histories)

    # compute mean, standard deviation and 95% empirical confidence interval boundaries for each row
    R_mean = np.mean(R_histories, axis=1)
    R_std = np.std(R_histories, axis=1)

    N_moves_mean = np.mean(N_moves_histories, axis=1)
    N_moves_std = np.std(N_moves_histories, axis=1)

    # save to file
    np.save(f"statistics/{method}_R_mean.npy", R_mean)
    np.save(f"statistics/{method}_R_std.npy", R_std)

    np.save(f"statistics/{method}_N_moves_mean.npy", N_moves_mean)
    np.save(f"statistics/{method}_N_moves_std.npy", N_moves_std)


def load_avg_statistics(method):
    R_mean = np.load(f"statistics/{method}_R_mean.npy")
    R_std = np.load(f"statistics/{method}_R_std.npy")

    N_moves_mean = np.load(f"statistics/{method}_N_moves_mean.npy")
    N_moves_std = np.load(f"statistics/{method}_N_moves_std.npy")

    return R_mean, R_std, N_moves_mean, N_moves_std