import sys
sys.path.append("..")

from COVID19_env.SEIR_env import SEIR_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines import ACKTR
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

def make_env(hospitalCapacity, env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = SEIR_env(hospitalCapacity)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    # Get environment inputs
    hospitalCapacity = 10000 # maximum number of people in the ICU

    env_id = "SEIR_env"
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(hospitalCapacity, env_id, i) for i in range(num_cpu)])

    # Define and Train the agent
    numTimesteps = 25000000 # number of training steps
    model = ACKTR(MlpPolicy, env, tensorboard_log="./SEIR_tensorboard/")
    model.learn(total_timesteps=numTimesteps)
