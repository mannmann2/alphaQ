import pandas as pd
from config import WINDOW_LENGTH, COMMISSION_RATE, OLPS_STRATEGIES


def evalu8(agent, env):
    """Closure runner for each episode."""
    # initialize rewards local buffer
    # rewards = []
    # initialize actions local buffer
    # actions = []
    # environment: reset & fetch observation
    ob = env.reset()
    # initialize reward
    reward = 0.0
    # termination flag
    done = False
    # environment state information
    info = {}
    # iterator for maximum episode steps
    j = 0
    # interaction loop
    while (not done):
        # agent closure: determine action
        action, _ = agent.predict(ob)
        # environment: take action
        new_ob, reward, done, info = env.step(action)
        # store reward
        # rewards.append(reward)
        # store action
        # actions.append(action)
        # agent closure: observe
        # set new observation to current
        ob = new_ob
        # increment iterator
        j = j + 1

#     return _rewards, _actions


def evaluate_baselines(data, agent_strategy=None):

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

    strategies = OLPS_STRATEGIES.copy()
    strategy_names = [algo.__class__.__name__ for algo in strategies]
    if agent_strategy:
        strategies.insert(0, agent_strategy)
        strategy_names.insert(0, agent_strategy.name)

    # 'algo', 'results',
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
