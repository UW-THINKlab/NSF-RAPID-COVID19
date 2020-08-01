import sys
sys.path.append("..")

from COVID19_env.SEIR_env import SEIR_env

from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.ddpg.noise import NormalActionNoise

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

# Get environment inputs
hospitalCapacity = 10000 # maximum number of people in the ICU

# Define environment
env = SEIR_env(hospitalCapacity)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Define and Train the agent
numTimesteps = 50000 # number of training steps
model = TD3(MlpPolicy, env, action_noise=action_noise, tensorboard_log="./SEIR_tensorboard/")
model.learn(total_timesteps=numTimesteps)
