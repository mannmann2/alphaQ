import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
from universal.algos import BAH, BCRP

import config
from alphaQ.agent.utils import AgentStrategy, Record
from alphaQ.utils import get_action_space, download_ticker_data

# TODO
# summary method?
# EPISODE #


class PortfolioEnv(gym.Env):
    """OpenAI Gym Base Trading Environment."""

    def __init__(self, **env_config):

        tickers = env_config['tickers']
        prices = env_config['prices']
        market_prices = env_config['market_prices']
        trading_period = env_config.get('trading_period')

        # set window length for observations
        self.window_length = env_config.get('window_length', config.WINDOW_LENGTH)
        # set commission fee rate
        self.trading_cost = env_config.get('trading_cost', config.COMMISSION_RATE)
        self.observation_with_weights = env_config.get('observation_with_weights', True)
        self.action_space_type = env_config.get('action_space_type', 'discrete')
        self.render_enabled = env_config.get('render', True)
        self.render_mode = env_config.get('render_mode', 'train')

        # <prices> not provided
        if prices is None and tickers is not None:
            # fetch prices
            print('downloading data...')
            prices = download_ticker_data(tickers, **env_config)

        # resample prices into _prices table
        if not trading_period:
            self.prices = prices.copy()
        else:
            self.prices = prices.resample(trading_period).last()

        if 'Adj Close' in self.features:
            self.close = 'Adj Close'
        else:
            self.close = 'Close'

        # relative (percentage) returns
        self.returns = self.prices[self.close].pct_change()  # self._prices/self._prices.shift(1) - 1

        # price relative vector
        self.Y = self.returns + 1  # self._prices/self._prices.shift(1) (eq 1)
        # add cash column
        self.Y['CASH'] = 1
        self.Y['CASH'].iloc[0] = np.nan

        if self.action_space_type == 'discrete':
            # discretize and set action space
            self.action_set = get_action_space(self.n_instruments, env_config.get('slices', config.SLICES))
            self.action_space = gym.spaces.Discrete(len(self.action_set))
        elif self.action_space_type == 'continuous':
            self.action_space = gym.spaces.Box(0, 1, (self.n_instruments+1,), dtype=np.float32)

        # define observation space
        history_space = gym.spaces.Box(-np.inf,  np.inf, (self.n_features, self.n_instruments, self.window_length), dtype=np.float32)
        if self.observation_with_weights:
            weight_space = gym.spaces.Box(0, 1, (self.n_instruments+1,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict({'history': history_space, 'weights': weight_space})
        else:
            self.observation_space = history_space

        # set counters to track total steps and number of episodes
        self.step_count = 0
        self.episode_count = 0

        # create an object to register agent stats
        self.record = Record(columns=self.instruments, index=self.dates[self.window_length-1:])
        self.result = None

        # calculate market rate
        if market_prices is not None:
            self.market = BAH().run(market_prices[self.window_length-1:])
        # best constant rebalanced portfolio
        self.bcrp = BCRP().run(self.prices[self.close][self.window_length-1:])
        self.bcrp.fee = self.trading_cost

    @property
    def instruments(self) -> list:
        """List of non cash asset instrument universe."""
        return self.prices.columns.unique(level=1).tolist()

    @property
    def n_instruments(self) -> int:
        """Count of portfolio assets."""
        return len(self.instruments)

    @property
    def features(self) -> list:
        """List of features used to train on."""
        return self.prices.columns.unique(level=0).tolist()

    @property
    def n_features(self) -> int:
        """Count of features."""
        return len(self.features)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Dates of the environment prices."""
        return self.prices.index

    @property
    def index(self) -> pd.Timestamp:
        """Return current index."""
        return self.dates[self.window_length + self._counter - 1]

    def _get_observation(self):
        """Build current observation to send to the agent."""
        window_data = self.prices[self._counter:self._counter + self.window_length]
        # divide price vector by latest close price for each asset
        obs = window_data.values.reshape(self.window_length, self.n_features, self.n_instruments)/window_data[self.close].iloc[-1].values
        obs = obs.transpose(1, 2, 0)

        if self.observation_with_weights:
            if self.action_space_type == 'continuous':
                return {'history': obs, 'weights': self.action}
            else:
                return {'history': obs, 'weights': self.weights}
        return obs

    def _get_done(self):
        """Check if episode has ended."""
        return self.index == self.dates[-1]

    def step(self, action):
        """Step through environment one step at a time.

        Parameters
        ----------
        action: object
            Action provided by the agent.

        Returns
        -------
        observation: object
            Agent's observation of the current environment.
        reward: float
            Amount of reward returned after previous action.
        done: bool
            Whether the episode has ended, in which case further step() calls will return undefined results.
        info: dict
            Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        # timestep
        self._counter += 1
        self.step_count += 1

        self.action = action

        # action validity check
        if not self.action_space.contains(action):
            raise ValueError('invalid `action` attempted: %s' % (action))

        if self.action_space_type == 'continuous':
            # self.weights = np.exp(actions)/np.sum(np.exp(actions))  # softmax normalisation
            self.weights = action/action.sum()
        else:
            self.weights = self.action_set[self.action]

        self.record.actions.loc[self.index] = self.weights

        # ======================================================================

        # relative return value
        y = self.Y.loc[self.index]

        # calculate commision to change from dw to new weights - ignoring cash for transaction cost
        mu = self.trading_cost * np.abs(self.dw - self.weights)[:-1].sum()  # (eq 16)

        # rho = (self.returns.loc[self.index] * self.weights[:-1]).sum()  # (eq 3)
        # rho = np.dot(y, self.weights) - 1  # (eq 3)
        rho = (1 - mu) * np.dot(y, self.weights) - 1  # (eq 9)

        # log_returns = np.log(np.dot(y, self.weights))  # (eq 4)
        log_returns = np.log((1 - mu) * np.dot(y, self.weights))  # (eq 10)

        # self.portfolio_value = self.portfolio_value * (rho + 1)  # (eq 2)
        # self.portfolio_value = self.portfolio_value * np.dot(y, self.weights)  # (eq 2)
        self.portfolio_value = self.portfolio_value * (1 - mu) * np.dot(y, self.weights)  # (eq 11)

        # calculating changed weights over the course of trading day
        self.dw = (y * self.weights)/np.dot(y, self.weights)  # (eq 7)

        # ======================================================================

        # fetch observation values
        observation = self._get_observation()

        # fetch termination flag
        done = self._get_done()

        # select reward
        reward = rho
        # reward = log_returns
        # reward = log_returns*1000/len(self.prices)

        info = {
            "reward": reward,
            "portfolio_value": self.portfolio_value,  # value_memory
            "rate_of_return": rho,  # return_memory
            "log_return": log_returns,
            "return": y.values,
            # "return": self._returns.loc[self.index].values,
            # 'actions': self.weights,
            'date': self.index,
            'step': self._counter
        }
        self.infos.append(info)

        # reward = rho + sharpe(np.array([x['rate_of_return'] for x in self.infos])

        if done:
            self.df_info = pd.DataFrame(self.infos, index=self.dates[self.window_length-1:])
            # calculate return for buy and hold a bit of each asset
            # self.df_info['market_value'] = np.cumprod(pd.DataFrame([info['return'] for info in self.infos])).mean(axis=1)
            # self.df_info['market_value'] = np.cumprod(self.df_info['return'].apply(lambda x: x[:-1])).apply(np.mean)

            self.episode_count += 1
            self.record.episodes.append({'rewards': self.df_info['rate_of_return'].sum(), 'total_wealth': self.portfolio_value})

            if self.render_mode == 'train':
                print("EPISODE:", self.episode_count, 'Steps:', self._counter)

            if self.render_enabled:
                self.render(self.render_mode)

        return observation, reward, done, info

    def reset(self) -> pd.DataFrame:
        """Reset the environment to an initial state and returns an observation.

        Returns
        -------
        observation: object
            The initial observation.
        """
        if self.action_space_type == 'continuous':
            self.action = np.append(np.zeros(self.n_instruments), 1)
        else:
            # action 0 corresponds to starting portfolio weights in discretised action space
            self.action = 0
        # setting initial weights to 0 for non-cash assets and 1 for cash
        self.weights = np.append(np.zeros(self.n_instruments), 1)
        self.dw = np.append(np.zeros(self.n_instruments), 1)

        self.portfolio_value = 1.0
        self.infos = [{'portfolio_value': 1.0, 'rate_of_return': 0.0, 'return': np.ones(self.n_instruments+1)}]

        # reset episode step counter
        self._counter = 0
        # get initial observation
        ob = self._get_observation()
        return ob

    def render(self, mode='train') -> None:
        """Render the environment."""
        # initialize figure and axes
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12), sharex=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        ax0, ax1 = axes[0], axes[1]

        # plot asset closing prices
        close_prices = self.prices[self.close].iloc[self.window_length-1:]
        close_prices['CASH'] = 1
        close_prices.plot(ax=ax0)  # :self.index

        # plot asset weights as heatmap over prices
        for asset in close_prices:
            close_prices[asset].reset_index().plot.scatter(x='Date', y=asset, c=self.record.actions[asset].values, cmap=plt.cm.Reds,
                                                           marker='o', s=100, vmin=0, vmax=1, alpha=0.2, ax=ax0, colorbar=False)

        # agent
        self.result = AgentStrategy(self.record.actions).run(self.prices[self.close].iloc[self.window_length-1:])  # :self.index
        self.result.fee = self.trading_cost

        # portfolio
        self.result.plot(assets=False, weights=False, ucrp=True, bah=True, ax=ax1, linewidth=2)

        # metrics
        self.bcrp.plot(assets=False, weights=False, ucrp=False, bah=False, portfolio_label='BCRP', ax=ax1, color=(0.0, 0.8789398846597463, 1.0))
        asset_equity = self.result.asset_equity
        best_stock = asset_equity[asset_equity.iloc[-1].idxmax()]
        best_stock.rename('Best Stock').plot(ax=ax1).legend()

        # market
        self.market.plot(assets=False, weights=False, ucrp=False, bah=False, portfolio_label='DJIA', ax=ax1, color='black')

        print("============================================")
        print("Total wealth:", round(self.result.total_wealth, 4))
        if mode == 'test':
            print(self.result.summary()[8:])
            print("Final Holdings:", np.round(self.weights, 3))
        print("============================================")

        # axes settings
        ax0.set_ylabel('Asset Prices')
        ax0_ = ax0.twinx()
        ax0_.set_ylim(ax0.get_ylim())

        ax1.set_xlim(self.dates[self.window_length-1], self.dates.max())
        ax1.yaxis.tick_right()
        # ax1.yaxis.set_label_position("right")
        ax1_ = ax1.twinx()
        ax1_.set_ylim(ax1.get_ylim())

        # draw throttled
        # plt.pause(0.0001)
        # fig.canvas.draw()
        plt.show()
