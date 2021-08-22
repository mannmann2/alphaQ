"""Agent utilities."""

import numpy as np
import pandas as pd

from universal.algo import Algo
from stable_baselines3 import DQN, DDPG


class AgentStrategy(Algo):
    """Universal Portfolio interface for RL agent strategy comparison."""

    PRICE_TYPE = 'raw'

    def __init__(self, actions, name='PORTFOLIO'):
        super().__init__()
        self.actions = actions
        self.name = name

    def weights(self, S):
        """Return weights."""
        return self.actions


class Record:
    """Local data structure for actions and rewards records."""

    def __init__(self, index, columns):
        # records of actions
        columns = columns + ['CASH']
        self.actions = pd.DataFrame(columns=columns, index=index, dtype=float)
        self.actions.iloc[0] = np.zeros(len(columns))
        self.actions.iloc[0]['CASH'] = 1.0
        # record of episode summaries
        self.episodes = []


def load_model(path):
    """Load agent from path based on name."""
    name = path.strip('/').split('/')[-1]
    if name.startswith('DQN'):
        return 'dqn', DQN.load(path)
    elif name.startswith('DDPG'):
        return 'ddpg', DDPG.load(path)
    else:
        print(' - Could not load model. Check name...')
        return None


def display_attributes(agent, obs_space=None):
    """Display selected agent attributes."""
    print('learning_rate:', agent.learning_rate)
    print('gamma:', agent.gamma)
    print('batch_size:', agent.batch_size)
    print('buffer_size:', agent.buffer_size)
    print('polyak_update:', agent.tau)

    if agent.action_noise:
        print('action_noise:', agent.action_noise)

    eps = agent.__dict__.get('exploration_initial_eps')
    if eps:
        print('epsilon_initial:', eps,
              '\tepislon_final:', agent.exploration_final_eps,
              '\tepislon_fraction:', agent.exploration_fraction)

    if 'features_extractor_class' in agent.policy_kwargs:
        x = agent.policy_kwargs['features_extractor_class'](obs_space)
        print('feature_extractor:')
        print(x.cnn)

    if 'net_arch' in agent.policy_kwargs:
        print('net_arch:', agent.policy_kwargs['net_arch'])

    print()
