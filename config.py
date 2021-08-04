"""Config file."""

from universal.algos import *
from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
# eps = np.finfo(float).eps

# MODEL = 'dqn'
MODEL = 'ddpg'

MODELS = {
    'dqn': DQN,
    'ddpg': DDPG
}

ACTION_SPACE = {
    'dqn': 'discrete',
    'ddpg': 'continuous',
}

MODEL_PATH = 'models/DDPG7_best_model.zip'
# MODEL_PATH = 'models/DQN7_best_model.zip'

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

# render agent performance on training data
RENDER_TRAINING = False

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
    GAMMA = 0.5
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
    TRAIN_FREQ = 1000
    TARGET_UPDATE = 1

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
SAVE_PATH = 'models/'
LOG_PATH = 'logs/'
CALLBACK_VERBOSE_LEVEL = 2

##################################################

# online portfolio selection strategies for
# benchmarking/comparing agent performance
# TODO other window/params
OLPS_STRATEGIES = [
    # benchmarks
    BAH(),
    CRP(),
    BCRP(),
    # DCRP()

    # follow the winner
    UP(),
    EG(),

    # follow the loser
    Anticor(window=50),
    PAMR(),
    OLMAR(window=50),
    RMR(window=50),
    CWMR(),
    WMAMR(window=50),

    # pattern matching
    # BNN(),
    CORN(window=50),

    # others
    # BestMarkowitz(),
    # Kelly(),
    # BestSoFar(),
    ONS(),
    # MPT()
]
