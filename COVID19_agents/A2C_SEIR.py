import sys
sys.path.append("..")

from COVID19_env.SEIR_env import SEIR_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines import A2C
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

if __name__ == "__main__":
    # Get environment inputs
    hospitalCapacity = 10000 # maximum number of people in the ICU

    env_id = "SEIR_env"
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(hospitalCapacity, env_id, i) for i in range(num_cpu)])

    # Define and Train the agent
    numTimesteps = 25000000 # number of training steps
    model = A2C(MlpPolicy, env, tensorboard_log="./SEIR_tensorboard/", full_tensorboard_log=True, verbose=False)
    trained_model = model.learn(total_timesteps=numTimesteps)


#    #-- Test the trained agent on single environment
#    env = SEIR_env(hospitalCapacity)
#    num_cities = 14
#    obs = env.reset()
#    obs_mat = np.reshape(obs,(4,num_cities))
#
#    # initial conditions
#    S = np.sum(obs_mat[0])
#    E = np.sum(obs_mat[1])
#    I = np.sum(obs_mat[2])
#    R = np.sum(obs_mat[3])
#    actions = []
#
#    # max steps(days) for test
#    max_steps = 100
#
#    n_steps = 0 # for tracking number of steps taken
#    for step in range(max_steps):
#      # increment
#      n_steps += 1
#      action, _ = model.predict(obs, deterministic=True)    #obtain optimal actions based on current state (obs)
#      obs, reward, done, info = env.step(action)
#      obs_mat = np.reshape(obs,(4,num_cities))
#
#      # save data to be plotted
#      S.append(np.sum(obs_mat[0]))
#      E.append(np.sum(obs_mat[1]))
#      I.append(np.sum(obs_mat[2]))
#      R.append(np.sum(obs_mat[3]))
#      actions.append(np.sum((action+1)/2))
#
#      # print update
#      print("Step {}".format(step + 1))
#      print("Action: ", np.sum((action+1)/2))
#      print('obs=', (np.sum(obs_mat[0]),np.sum(obs_mat[1]),np.sum(obs_mat[2]),np.sum(obs_mat[3])),'reward=', reward,'done=', done)
#
#      if done:
#        print("Done.", "reward=", reward)



    actions.append('-') # no action for last time step

    #-- Plot results

    # initiate figure
    fig, ax = plt.subplots(constrained_layout=True)

    # define time vector for plot
    steps = np.linspace(0,n_steps,n_steps+1)

    # plot saved data from test
    plt.plot(steps, S, "-b", label="Susceptible")
    plt.plot(steps, E, "-b", label="Susceptible")
    plt.plot(steps, I, "-r", label="Infected")
    plt.plot(steps, R, "-y", label="Recovered")
    plt.plot([0,max(steps)], [hospitalCapacity,hospitalCapacity], "--k", label="Hospital Capacity")

    # Create 'Action' axis
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(steps)
    secax.set_xticklabels(actions)
    secax.set_xlabel("Action")

    # Create legends and labels
    textBoxStr = '\n'.join((
        '0: Open all',
        '1: Open half',
        '2: Stay home'))
    ax.text(0.7, 0.85, textBoxStr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top') # action text box / legend
    plt.legend(loc="best")
    plt.xlabel("Day")
    plt.ylabel("Number of People")
    titleStr= ("A2C agent after %d training steps" % (numTimesteps))
    plt.title(titleStr)

    # save figure to results folder
    saveStr= ("../Results/A2C_SEIR_results%d.png" % (numTimesteps))
    fig.savefig(saveStr)
