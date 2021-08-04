import numpy as np
import pandas as pd

from universal.algo import Algo


class AgentStrategy(Algo):
    """Universal Portfolio interface for RL agent to
    allow strategy comparison.
    """
    PRICE_TYPE = 'raw'

    def __init__(self, actions, name='PORTFOLIO'):
        """:params b: Portfolio weights at start. Default are uniform."""
        super().__init__()
        self.actions = actions
        self.name = name

    def weights(self, S):
        return self.actions


class Record:
    """Local data structure for actions and rewards records."""

    def __init__(self, index, columns):
        # records of actions
        columns = columns + ['CASH']
        self.actions = pd.DataFrame(columns=columns, index=index, dtype=float)
        self.actions.iloc[0] = np.zeros(len(columns))
        self.actions.iloc[0]['CASH'] = 1.0
        # records of rewards
        self.rewards = pd.DataFrame(columns=columns, index=index, dtype=float)
        self.rewards.iloc[0] = np.zeros(len(columns))
