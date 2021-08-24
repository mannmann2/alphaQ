"""Config file."""

from universal import algos
from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise


MODELS = {
    'dqn': DQN,
    'ddpg': DDPG
}

MODEL_PATH = {
    'ddpg': 'models/DDPG_best.zip',
    'dqn': 'models/DQN_best.zip'
}

ACTION_SPACE = {
    'dqn': 'discrete',
    'ddpg': 'continuous',
}

ACTION_NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

# DJIA as of April 14, 2021
DOW_TICKERS = [
  'UNH',   # United Health
  'GS',    # Goldman Sachs
  'HD',    # Home Depot
  'MSFT',  # Microsoft
  'BA',    # Boeing
  'AMGN',  # Amgen
  'CAT',   # Caterpillar
  'MCD',   # McDonald's
  'HON',   # Honeywell
  'CRM',   # Salesforce
  'V',     # Visa (2013-09-20/2008-03-19)
  'MMM',   # 3M
  'DIS',   # Disney
  'JNJ',   # Johnson & Johnson
  'TRV',   # Travelers Companies
  'JPM',   # JPMorgan Chase
  'AXP',   # American Express
  'WMT',   # Walmart
  'PG',    # Proctor & Gamble
  'IBM',   # IBM
  'NKE',   # Nike
  'AAPL',  # Apple (2015-03-19)
  'CVX',   # Chevron Corporation
  'MRK',   # Merck & Co
  'DOW',   # Dow Inc
  'INTC',  # Intel
  'VZ',    # Verizon
  'WBA',   # Walgreens Boots Alliance
  'KO',    # Coca-Cola
  'CSCO',  # Cisco
]

##################################################
# Environment Configurations
##################################################

# TICKERS = ['AAPL', 'GE', 'JPM', 'MSFT', 'VOD', 'GS', 'MMM']
# TICKERS = ['AAPL', 'GE', 'JPM', 'MSFT']
TICKERS = ['AAPL', 'MSFT', 'V', 'JPM']
# TICKERS = ['AAPL', 'MSFT', 'V', 'JPM', 'JNJ', 'WMT']

# start date from which to begin downloading ticker data
START = '2008-03-19'
# START = '2013-09-20'

# end date for data used in experiments
END = '2021-07-31'

# number of units to divide discreted weights into
SLICES = 8

# number of days' prices used for a single observation
WINDOW_LENGTH = 50

# fee charged as commission by the broker
COMMISSION_RATE = 0.0025

##################################################
# Training Hyperparameters
##################################################

DDPG_KWARGS = dict(
    policy='MultiInputPolicy',
    learning_rate=0.000002,
    learning_starts=10000,
    buffer_size=10000,
    batch_size=64,
    tau=0.005,
    gamma=1,
    train_freq=100,
    action_noise='ornstein_uhlenbeck'  # 'normal' | None
)

DQN_KWARGS = dict(
    policy='MultiInputPolicy',
    learning_rate=0.00005,
    learning_starts=10000,
    buffer_size=10000,
    batch_size=64,
    tau=0.005,
    gamma=0.5,
    train_freq=1,
    target_update_interval=1000
)

TRAIN_VERBOSE_LEVEL = 2
RANDOM_SEED = 42

##################################################
# Exploration configurations
##################################################

DDPG_EX_PARAMS = dict(
    noise_sigma=0.2,
)

DQN_EX_PARAMS = dict(
    exploration_fraction=0.4,
    exploration_initial_eps=0.8,
    exploration_final_eps=0.01,
)

##################################################
# Combined Model and NN Configurations
##################################################

MODEL_PARAMS = {
    'ddpg': dict(
        net_arch=[512, 256, 128],
        multiplier=2,
        hyperparams=DDPG_KWARGS,
        exploration=DDPG_EX_PARAMS,
    ),
    'dqn': dict(
        net_arch=[256, 128, 64],
        multiplier=1,
        hyperparams=DQN_KWARGS,
        exploration=DQN_EX_PARAMS,
    )
}

##################################################
# Callback configurations
##################################################

CALLBACK_ENABLED = True
CALLBACK_START = 20000
SAVE_PATH = 'models/'
LOG_PATH = 'logs/'
CALLBACK_VERBOSE_LEVEL = 2

##################################################

# metrics to be calculated when evaluating strategies
ATTRIBUTES = [
    'total_wealth',
    'cumulative_return',
    'annualised_return',
    'sharpe',
    'max_drawdown',
    'max_drawdown_period',
    'ulcer_index',
    'profit_factor',
    'winning_pct',
    'beta',
    'alpha',
    'appraisal_ratio',
    'information_ratio',
    'annualised_volatility',
    'annual_turnover',
]

# online portfolio selection strategies for
# benchmarking/comparing agent performance
OLPS_STRATEGIES = [
    # benchmarks
    algos.BAH(),
    algos.CRP(),
    algos.BCRP(),
    # algos.DCRP(),
    algos.MPT(window=50, min_history=1, mu_estimator='historical', cov_estimator='empirical', q=0),  # min-variance

    # follow the winner
    algos.UP(),
    algos.EG(),

    # follow the loser
    algos.Anticor(window=WINDOW_LENGTH),
    algos.PAMR(eps=1),
    algos.OLMAR(window=WINDOW_LENGTH),
    algos.RMR(window=WINDOW_LENGTH),
    algos.CWMR(),
    algos.WMAMR(window=WINDOW_LENGTH),

    # pattern matching
    algos.BNN(k=WINDOW_LENGTH),
    algos.CORN(),

    # meta-learning
    algos.ONS(),
]
