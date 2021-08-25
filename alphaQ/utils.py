"""Utility functions used in various modules."""

from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def download_ticker_data(tickers, start='2000-01-01', end=datetime.today().date(), columns=['Adj Close']):
    """Download price data for multiple tickers in bulk."""
    data = yf.download(tickers, start=start, end=end)
    return data[columns]


def train_test_split(data, n_train=None, train_years=10, validation_set=True):
    """Split data into train, test (and validation) sets."""
    if n_train is None:
        if train_years is None:
            raise "Either `n_train` or `train_years` is required."
        n_train = int(train_years * 252)

    if n_train > data.shape[0]:
        raise "Invalid train size selected."

    train = data[:n_train]
    test = data[n_train:]

    if validation_set:
        test_size = int((data.shape[0] - n_train)/2)
        val, test = test[:test_size], test[test_size:]

        return train, val, test

    return train, test


def get_action_space(M, N):
    """Discretize and return action space for DQN.

    Parameters
    ----------
    M: int
        Number of traded assets.
    N: int
        Number of divisions/slices allowed in
        discretized action space.

    Returns
    -------
    A: list
        list of all unique actions in the action space
        after discretization.
    """
    A = []
    NUM = M + 1 + N - 1
    seq = list(range(NUM))

    for c in combinations(seq, M):
        c = list(c)
        action = np.zeros(M+1)
        for i in range(len(c)-1):
            action[i+1] = c[i+1] - c[i] - 1

        action[0] = c[0]
        action[M] = NUM - c[M-1] - 1
        for j in range(M+1):
            action[j] = action[j]/N

        A.append(action)

    return A


def plot_episodes(env):
    """Plot episode summaries and performance on env."""
    # create dataframe with episode stats
    episodes = pd.DataFrame(env.record.episodes)

    print("mean:")
    print(episodes.mean())

    _, axes = plt.subplots(figsize=(20, 5))

    # plot cumulative reward per-episode
    episodes.rewards.plot.bar(color='orange', ax=axes)
    episodes.rewards.rolling(window=10).mean().plot(ax=axes)
    # df_episodes.total_wealth.plot(color='r', ax=axes)

    # axes settings
    plt.xticks(rotation=90)
    axes.set(title='Reward per Episode', ylabel='Reward', xlabel='Episode')
    plt.show()


def sharpe(returns, freq=252, rfr=0):
    """Calculate sharpe ratio for returns."""
    eps = np.finfo(float).eps
    return np.sqrt(freq) * np.mean(returns - rfr) / np.std(returns - rfr + eps)


# def max_drawdown(returns):
#     """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
#     peak = returns.max()
#     trough = returns[returns.argmax():].min()
#     return (trough - peak) / (peak + eps)
