"""Feature extractor for RL models."""

import gym
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
from torch import nn


class FeatureExtractor(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super(FeatureExtractor, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        # history = observation_space['history'].shape
        n_input_channels = observation_space['history'].shape[0]
        self.n_instruments = observation_space['history'].shape[1]
        mult = 1

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8*mult, kernel_size=(1, 8), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8*mult, 16*mult, kernel_size=(1, 16), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16*mult, 32*mult, kernel_size=(1, 28), stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 8*mult, kernel_size=(1, 5), stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(8*mult, 16*mult, kernel_size=(1, 46), stride=1, padding=0),
        #     nn.ReLU(),
        # )
        # self.temp = nn.Sequential(
        #     # nn.Conv2d(16*mult + 1, 32*mult, kernel_size=1, stride=1, padding=0),
        #     # nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(self.n_instruments*(16*mult + 1), features_dim),
        #     nn.ReLU()
        # )

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 3, kernel_size=(1, 3), stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(3, 10, kernel_size=(1, 48), stride=1, padding=0),
        #     nn.ReLU(),
        # )
        # self.temp = nn.Sequential(
        #     nn.Conv2d(11, 1, kernel_size=(1, 1), stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(self.n_flatten, features_dim),
        #     nn.ReLU()
        # )

    def forward(self, observations: TensorDict) -> th.Tensor:
        a = self.cnn(observations['history'])
        # print(observations['history'].shape, observations['weights'].shape)
        # concatenate weights to network output
        k = th.cat((a, observations['weights']), dim=1)

        return k
