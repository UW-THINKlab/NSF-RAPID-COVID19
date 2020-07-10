"""
custom environment must follow the gym interface
skeleton copied from:
    https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
"""
import gym
from gym import spaces

class CustomEnv(gym.Env):
  """
  Description:
        A population is broken down into three compartments -- Susceptible,
        Infected, and Recovered -- to model the spread of COVID-19.

  Source:
        Original SIR model from https://rpubs.com/choisy/sir

  Observation:
        Type: Box(3,)
        Num     Observation     Min     Max
        0       Susceptible     0       Total Population
        1       Infected        0       Total Population
        2       Recovered       0       Total Population

  Actions:

  Reward:

  Starting State:

  Episode Termination:

  """
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2, ...):
    super(CustomEnv, self).__init__()

    # SIR model parameters
    self.beta = 0.004   # infectious contact rate (/person/day)
    self.gamma = 0.5    # recovery rate (/day)
    
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    ...
    return observation, reward, done, info
  def reset(self):
    ...
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    ...
  def close (self):
    ...
