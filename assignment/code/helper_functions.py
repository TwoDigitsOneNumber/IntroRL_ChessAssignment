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
    return ema


def save_avg_statistics(histories, method):
    # unpack histories
    R_histories = [history[0] for history in histories]
    N_moves_histories = [history[1] for history in histories]
    training_times = [history[2] for history in histories]
    layer1_gradient_norms_histories = [history[3] for history in histories]
    layer2_gradient_norms_histories = [history[4] for history in histories]

    # turn into numpy arrays
    R_histories = np.hstack(R_histories)
    N_moves_histories = np.hstack(N_moves_histories)
    training_times = np.hstack(training_times)
    layer1_gradient_norms_histories = np.hstack(layer1_gradient_norms_histories)
    layer2_gradient_norms_histories = np.hstack(layer2_gradient_norms_histories)

    # compute mean and standard deviation for each row of the histories
    R_mean = np.mean(R_histories, axis=1)
    R_std = np.std(R_histories, axis=1)

    N_moves_mean = np.mean(N_moves_histories, axis=1)
    N_moves_std = np.std(N_moves_histories, axis=1)

    layer1_gradient_norms_mean = np.mean(layer1_gradient_norms_histories, axis=1)
    layer1_gradient_norms_std = np.std(layer1_gradient_norms_histories, axis=1)

    layer2_gradient_norms_mean = np.mean(layer2_gradient_norms_histories, axis=1)
    layer2_gradient_norms_std = np.std(layer2_gradient_norms_histories, axis=1)

    # save to file
    np.save(f"statistics/{method}_R_mean.npy", R_mean)
    np.save(f"statistics/{method}_R_std.npy", R_std)

    np.save(f"statistics/{method}_N_moves_mean.npy", N_moves_mean)
    np.save(f"statistics/{method}_N_moves_std.npy", N_moves_std)

    np.save(f"statistics/{method}_training_times.npy", training_times)

    np.save(f"statistics/{method}_layer1_gradient_norms_mean.npy", layer1_gradient_norms_mean)
    np.save(f"statistics/{method}_layer1_gradient_norms_std.npy", layer1_gradient_norms_std)

    np.save(f"statistics/{method}_layer2_gradient_norms_mean.npy", layer2_gradient_norms_mean)
    np.save(f"statistics/{method}_layer2_gradient_norms_std.npy", layer2_gradient_norms_std)


def load_avg_statistics(method):
    R_mean = np.load(f"statistics/{method}_R_mean.npy")
    R_std = np.load(f"statistics/{method}_R_std.npy")

    N_moves_mean = np.load(f"statistics/{method}_N_moves_mean.npy")
    N_moves_std = np.load(f"statistics/{method}_N_moves_std.npy")

    training_times = np.load(f"statistics/{method}_training_times.npy")

    layer1_gradient_norms_mean = np.load(f"statistics/{method}_layer1_gradient_norms_mean.npy")
    layer1_gradient_norms_std = np.load(f"statistics/{method}_layer1_gradient_norms_std.npy")

    layer2_gradient_norms_mean = np.load(f"statistics/{method}_layer2_gradient_norms_mean.npy")
    layer2_gradient_norms_std = np.load(f"statistics/{method}_layer2_gradient_norms_std.npy")

    return R_mean, R_std, N_moves_mean, N_moves_std, training_times, layer1_gradient_norms_mean, layer1_gradient_norms_std, layer2_gradient_norms_mean, layer2_gradient_norms_std


def printable_name(method):
    if method == "sarsa":
        return "SARSA"
    elif method == "qlearning":
        return "Q-Learning"
    elif method == "dqn":
        return "DQN"
    else:
        return None