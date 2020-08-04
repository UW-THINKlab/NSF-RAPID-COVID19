import sys
sys.path.append("..")

from COVID19_env.SEIR_env import SEIR_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines.common import make_vec_env
from stable_baselines import ACKTR
from stable_baselines.common.evaluation import evaluate_policy

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

# Get environment inputs
hospitalCapacity = 10000 # maximum number of people in the ICU

# Define environment
env = SEIR_env(hospitalCapacity)

# Define and Train the agent
numTimesteps = 500000 # number of training steps
model = ACKTR(MlpPolicy, env, tensorboard_log="./SEIR_tensorboard/")
model.learn(total_timesteps=numTimesteps)
