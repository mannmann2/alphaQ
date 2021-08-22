"""Feature extractor for RL model."""

import gym
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
from torch import nn


class FeatureExtractor(BaseFeaturesExtractor):
    """CNN based feature extractor for stock price data.

    Inspired by the CNN implementation by Jiang et al. (2017)
    "A deep reinforcement learning framework for the financial portfolio
    management problem"

    Parameters
    ----------
        observation_space: gym.spaces.Dict
            Assumes CxHxW shape (channels first).

        features_dim: int
            Number of units in the last layer.

        multiplier: int
            Multiplication factor for layer size.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512, multiplier: int = 1):
        super(FeatureExtractor, self).__init__(observation_space, features_dim)

        # history_shape = observation_space['history'].shape
        n_input_channels = observation_space['history'].shape[0]
        # n_instruments = observation_space['history'].shape[1]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 4*multiplier, kernel_size=(1, 8), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(4*multiplier, 8*multiplier, kernel_size=(1, 16), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8*multiplier, 16*multiplier, kernel_size=(1, 28), stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, observations: TensorDict) -> th.Tensor:
        """Forward pass of the neural network."""
        # print(observations['history'].shape, observations['weights'].shape)
        a = self.cnn(observations['history'])
        # concatenate weights to network output
        k = th.cat((a, observations['weights']), dim=1)

        return k
