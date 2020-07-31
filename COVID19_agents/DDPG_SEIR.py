import sys
sys.path.append("..")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from COVID19_env.SEIR_env import SEIR_env

from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

# R <--> python conversions
numpy2ri.activate() # automatic conversion of numpy objects to rpy2 objects
robjects.r('''
       source("../COVID19_models/SEIR/read_input.R")
       source("../COVID19_models/SEIR/seir_r.R")
''') # source all R functions in the specified file
getData_r = robjects.globalenv['getData']
seir_r = robjects.globalenv['seirPredictions'] # get R model


# Get environment inputs
input_data = getData_r("../COVID19_models/SEIR/RL_input")
hospitalCapacity = 10000 # maximum number of people in the ICU
beta_data    = input_data.rx2("beta_data")
beta_sd_data = input_data.rx2("beta_sd_data")
latent       = input_data.rx2("latent")
gamma        = input_data.rx2("gamma")
St_data      = np.array(input_data.rx2("St_data"))
Et_data      = np.array(input_data.rx2("Et_data"))
It_data      = np.array(input_data.rx2("It_data"))
Rt_data      = np.array(input_data.rx2("Rt_data"))
ODs          = input_data.rx2("ODs")
pops         = input_data.rx2("pops")
current      = int(input_data.rx2("current")[0])
pred         = int(input_data.rx2("pred")[0])
city_names   = input_data.rx2("city_names")

# Define environment
env = SEIR_env(hospitalCapacity,
                     beta_data,
                     beta_sd_data,
                     latent,
                     gamma,
                     St_data,
                     Et_data,
                     It_data,
                     Rt_data,
                     ODs,
                     pops,
                     current,
                     pred,
                     city_names)

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# Define and Train the agent
numTimesteps = 50000 # number of training steps
model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=numTimesteps)
