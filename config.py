"""Config file."""

from universal.algos import *
from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
# eps = np.finfo(float).eps

# MODEL = 'ddpg'
MODEL = 'dqn'

MODEL_PATH = {
    'ddpg': 'models/DDPG7_best_model.zip',
    'dqn': 'models/DQN7_best_model.zip'
}

MODELS = {
    'dqn': DQN,
    'ddpg': DDPG
}

ACTION_SPACE = {
    'dqn': 'discrete',
    'ddpg': 'continuous',
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

TRAIN_VERBOSE_LEVEL = 2
RANDOM_SEED = 42

if MODEL == 'ddpg':

    POLICY = 'MultiInputPolicy'
    ALPHA = 0.000002
    TRAIN_START = 10000
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    TAU = 0.005
    GAMMA = 1
    TRAIN_FREQ = 100
    ACTION_NOISE = 'ornstein_uhlenbeck'  # 'normal' | None

elif MODEL == 'dqn':
    POLICY = 'MultiInputPolicy'
    ALPHA = 0.00002
    TRAIN_START = 10000
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    TAU = 0.005
    GAMMA = 0.5
    TRAIN_FREQ = 1
    TARGET_UPDATE = 1000

##################################################
# Exploration configurations
##################################################

if MODEL == 'ddpg':
    NOISE = {
        "normal": NormalActionNoise,
        "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
    }
    NOISE_SIGMA = 0.2

elif MODEL == 'dqn':
    EPSILON_FRACTION = 0.4
    EPSILON_INITIAL = 0.8
    EPSILON_FINAL = 0.01

##################################################
# Callback configurations
##################################################

CALLBACK_ENABLED = True
CALLBACK_START = 20000
SAVE_PATH = 'models/'
LOG_PATH = 'logs/'
CALLBACK_VERBOSE_LEVEL = 2

##################################################

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
# TODO other window/params
OLPS_STRATEGIES = [
    # benchmarks
    algos.BAH(),
    algos.CRP(),
    algos.BCRP(),
    # algos.DCRP(),
    algos.MPT(window=50, min_history=1, mu_estimator='historical', cov_estimator='empirical', q=0), # min-variance

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
