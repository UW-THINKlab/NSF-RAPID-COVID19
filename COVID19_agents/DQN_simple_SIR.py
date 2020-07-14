import sys
sys.path.append("..")

from COVID19_env.simple_SIR_env import simple_SIR_env

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

# Initiate the env
env = simple_SIR_env()

# Define and Train the agent
numTimesteps = 50000
model = DQN(MlpPolicy,env, tensorboard_log="./DQN_SIR_tensorboard/")
model.learn(total_timesteps=numTimesteps)


# Test the trained agent
obs = env.reset()
max_steps = 100
n_steps = 0

S = [obs[0]]
I = [obs[1]]
R = [obs[2]]

actions = []

for step in range(max_steps):
  n_steps += 1
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  S.append(obs[0])
  I.append(obs[1])
  R.append(obs[2])
  actions.append(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  #env.render()
  if done:
    print("Done.", "reward=", reward)
    break

actions.append('-')

# Plot
steps = np.linspace(0,n_steps,n_steps+1)

fig = plt.figure()

fig, ax = plt.subplots(constrained_layout=True)

plt.plot(steps, S, "-b", label="Susceptible")
plt.plot(steps, I, "-r", label="Infected")
plt.plot(steps, R, "-y", label="Recovered")
plt.plot([0,max(steps)], [300,300], "--k", label="Hospital Capacity")

# Create 'Action' axis
secax = ax.secondary_xaxis('top')
secax.set_xticks(steps)
secax.set_xticklabels(actions)
secax.set_xlabel("Action")

# place a text box in upper left in axes coords
textBoxStr = '\n'.join((
    '0: Open all',
    '1: Open half',
    '2: Stay home'))

ax.text(0.7, 0.85, textBoxStr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top')

plt.legend(loc="best")
plt.xlabel("Day")
plt.ylabel("Number of People")
titleStr= ("DQN agent after %d training steps" % (numTimesteps))
plt.title(titleStr)
saveStr= ("DQN_simple_SIR_results%d.png" % (numTimesteps))
fig.savefig(saveStr)

'''
run the following in separate terminal to monitor results:
    tensorboard --logdir ./DQN_SIR_tensorboard/
'''
