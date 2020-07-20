import sys
sys.path.append("..")

from COVID19_env.simple_SIR_env import simple_SIR_env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines import A2C
from stable_baselines.common.evaluation import evaluate_policy

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

# Initiate the env
S0 = 999  # number of susceptibles at time = 0
I0 =   1  # number of infected at time = 0
R0 =   0  # number of recovered (and immune) at time = 0
hospitalCapacity = 300 # maximum number of people in the ICU
env = simple_SIR_env(S0, I0, R0, hospitalCapacity)

# Define and Train the agent
numTimesteps = 50000 # number of training steps
model = A2C(MlpPolicy,env, tensorboard_log="./DQN_SIR_tensorboard/")
model.learn(total_timesteps=numTimesteps)

#-- Test the trained agent

obs = env.reset()

# initial conditions
S = [obs[0]]
I = [obs[1]]
R = [obs[2]]
actions = []

# max steps(days) for test
max_steps = 100

n_steps = 0 # for tracking number of steps taken
for step in range(max_steps):
  # increment
  n_steps += 1
  action, _ = model.predict(obs, deterministic=True)
  obs, reward, done, info = env.step(action)

  # save data to be plotted
  S.append(obs[0])
  I.append(obs[1])
  R.append(obs[2])
  actions.append(action)

  # print update
  print("Step {}".format(step + 1))
  print("Action: ", action)
  print('obs=', obs, 'reward=', reward, 'done=', done)

  if done:
    print("Done.", "reward=", reward)
    break

actions.append('-') # no action for last time step

#-- Plot results

# initiate figure
fig, ax = plt.subplots(constrained_layout=True)

# define time vector for plot
steps = np.linspace(0,n_steps,n_steps+1)

# plot saved data from test
plt.plot(steps, S, "-b", label="Susceptible")
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
saveStr= ("../Results/A2C_simple_SIR_results%d.png" % (numTimesteps))
fig.savefig(saveStr)

'''
run the following in a separate terminal to monitor results:
    tensorboard --logdir ./DQN_SIR_tensorboard/
'''
