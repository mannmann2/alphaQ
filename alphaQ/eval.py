"""Functions for evaluation and comparison of agent performance."""

import pandas as pd
from universal.algos import BAH

from config import WINDOW_LENGTH, COMMISSION_RATE, ATTRIBUTES, OLPS_STRATEGIES


def evalu8(agent, env):
    """Closure runner for each episode."""
    # reset environment and fetch observation
    obs = env.reset()
    # completion flag
    done = False

    # until episode completes
    while (not done):
        # determine action
        action, _states = agent.predict(obs, deterministic=True)
        # environment: take action
        new_obs, reward, done, info = env.step(action)
        # update current observation
        obs = new_obs


def get_stats(result):
    # TODO assign with dict
    alpha, beta = result.alpha_beta()
    stats = [
        result.total_wealth,
        (result.total_wealth - 1) * 100,
        result.annualized_return * 100,
        result.sharpe,
        result.max_drawdown * 100,
        result.drawdown_period,
        result.ulcer,
        result.profit_factor,
        result.winning_pct * 100,
        beta,
        alpha,
        result.appraisal_ucrp,
        result.information,
        result.annualized_volatility,
        result.turnover,
    ]
    return stats


def evaluate_baselines(data, custom_strategies=[], market_data={}, attributes=None):
    """Evaluate and compare baseline strategies (with agent strategies)."""
    strategies = OLPS_STRATEGIES.copy()
    strategy_names = [algo.__class__.__name__ for algo in strategies]

    # if not list, convert to list
    if not isinstance(custom_strategies, list):
        custom_strategies = [custom_strategies]

    # append agent strategies to benchmarks/baselines
    strategies = custom_strategies + strategies
    strategy_names = list(map(lambda x: x.name, custom_strategies)) + strategy_names

    df_metrics = pd.DataFrame(index=strategy_names, columns=ATTRIBUTES)
    results = pd.Series(index=strategy_names)

    for name, algo in zip(df_metrics.index, strategies):
        print(name, end='.')
        result = algo.run(data['Close'][WINDOW_LENGTH-1:])
        result.fee = COMMISSION_RATE

        df_metrics.loc[name] = get_stats(result)
        results.loc[name] = result

    for name, data in market_data.items():
        print(name, end='.')
        result = BAH().run(data[WINDOW_LENGTH-1:])
        result.fee = COMMISSION_RATE

        df_metrics.loc[name] = get_stats(result)
        results.loc[name] = result

    if attributes:
        df_metrics = df_metrics[attributes]
    return results, df_metrics
