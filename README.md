# alphaQ

A Q-learing based portfolio trading system built using [OpenAI Gym](https://github.com/openai/gym) and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Requirements
- Python 3.7

## Installation
1. Clone this repository: `git clone https://github.com/mannmann2/alphaQ.git`
1. `cd alphaQ`
1. `pip install -r requirements.txt`

## Usage 
For learning how to train and evaluate the agents, follow the starter notebook provided with this repository.
- [main.ipynb](main.ipynb)

Or view it in Jupyter's online notebook viewer: 
- [Nbviewer](https://nbviewer.jupyter.org/github/mannmann2/alphaQ/blob/main/main.ipynb)

## Models
Below is a list of all the pre-trained models provided.

#### Best models for `[AAPL, JPM, MSFT, V]`  
[DQN_best](models/DQN_best.zip)  
[DDPG_best](models/DDPG_best.zip)

#### Alternate models for `[AAPL, JPM, MSFT, V]` 
**DQN:** [DQN 5](models/DQN5.zip), [DQN 6](models/DQN6.zip), [DQN 7](models/DQN7.zip), [DQN 8](models/DQN8.zip)  
**DDPG:** [DDPG 7](models/DDPG7.zip), [DDPG 8](models/DDPG8.zip)  

#### Variant Models
[DQN A](models/variants/DQN_A.zip)  `AXP, CVX, DIS, KO`  
[DQN B](models/variants/DQN_B.zip)  `JNJ, MCD, MMM, WMT`  
[DQN C](models/variants/DQN_B.zip)  `CAT, CSCO, HD, IBM`  

[DDPG A](models/variants/DDPG_A.zip)  `JNJ, MCD, MMM, WMT`  
[DDPG B](models/variants/DDPG_B.zip)  `AMGN, NKE, UNH, VZ`  
[DDPG C](models/variants/DDPG_C.zip)  `GS, NKE, PG, UNH`  

## Documentation
[Project Proposal](docs/ProjectProposal.pdf)  
[Progress Slides 1](docs/ProgressSlides1.pdf)  
[Progress Slides 2](docs/ProgressSlides2.pdf)  
[Progress Slides 3](docs/ProgressSlides3.pdf)  
[Progress Slides 4](docs/ProgressSlides4.pdf)  
[Preliminary Report](docs/PreliminaryProjectReport.pdf)  


* Free software: MIT license
