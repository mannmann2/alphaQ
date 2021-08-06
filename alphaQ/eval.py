"""Functions for evaluation and comparison of agent performance."""

import pandas as pd
from config import WINDOW_LENGTH, COMMISSION_RATE, OLPS_STRATEGIES


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


def evaluate_baselines(data, agent_strategy=None):
    """Evaluate and compare baseline strategies (with agent strategies)."""
    strategies = OLPS_STRATEGIES.copy()
    strategy_names = [algo.__class__.__name__ for algo in strategies]

    if agent_strategy:
        if not isinstance(agent_strategy, list):
            # convert to list
            agent_strategy = [agent_strategy]

        # append agent strategies to benchmarks/baselines
        strategies = agent_strategy + strategies
        strategy_names = list(map(lambda x: x.name, agent_strategy)) + strategy_names

    attributes = [
        'total_wealth',
        'profit_factor',
        'sharpe',
        'annualized_return',
        'max_drawdown',
        'drawdown_period',
        'information',
        'winning_pct',
        'annual_turnover'
    ]

    df_metrics = pd.DataFrame(index=strategy_names, columns=attributes)
    results = pd.Series(index=strategy_names)

    for name, algo in zip(df_metrics.index, strategies):
        print(name, end='.')
        result = algo.run(data['Close'][WINDOW_LENGTH-1:])
        result.fee = COMMISSION_RATE

        df_metrics.loc[name]['total_wealth'] = result.total_wealth
        df_metrics.loc[name]['profit_factor'] = result.profit_factor
        df_metrics.loc[name]['sharpe'] = result.sharpe
        df_metrics.loc[name]['information'] = result.information
        df_metrics.loc[name]['annualized_return'] = result.annualized_return * 100
        df_metrics.loc[name]['max_drawdown'] = result.max_drawdown * 100
        df_metrics.loc[name]['drawdown_period'] = result.drawdown_period
        df_metrics.loc[name]['winning_pct'] = result.winning_pct * 100
        df_metrics.loc[name]['annual_turnover'] = result.turnover

        results.loc[name] = result

    return results, df_metrics
