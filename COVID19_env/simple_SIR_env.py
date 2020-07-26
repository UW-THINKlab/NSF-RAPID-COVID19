"""
custom environment must follow the gym interface
skeleton copied from:
    https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
"""

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

class simple_SIR_env(gym.Env):
  """
  Description:
        A population is broken down into three compartments -- Susceptible,
        Infected, and Recovered -- to model the spread of COVID-19.

  Source:
        Original SIR model from https://rpubs.com/choisy/sir
        Code modeled after cartpole.py from
         github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

  Observation:
        Type: Box(3,)
        Num     Observation       Min     Max
        0       Susceptible       0       Total Population
        1       Infected          0       Total Population
        2       Recovered         0       Total Population

  Actions:
        Type: Discrete(3)
        Num     Action                      Change in model
        0       open everything             beta = 0.004
        1       open at half capacity       beta = 0.002
        2       stay at home order          beta = 0.001

  Reward:
        reward = health cost + economic cost

        Health cost:
            -1  for every infected case
            -10 for every infected case that brings total infected count above a
                specified maximum (hospital capacity)
        Economic cost:
             0 for every day using 'open everything'
           -10 for every day using 'open at half capacity'
          -100 for every day using 'stay at home order'

  Starting State:
        Susceptible:    999
        Infected:       001
        Recovered:      000

  Episode Termination:
        Number of infections is zero
        Episode length (time) reaches specified maximum (end time)
  """


  metadata = {'render.modes': ['human']}

  def __init__(self, S0, I0, R0, hospitalCapacity):
    super(simple_SIR_env, self).__init__()

    # SIR model parameters
    self.beta  = 0.004     # infectious contact rate (/person/day)
    self.gamma = 0.5       # recovery rate (/day)
    self.hospitalCap = hospitalCapacity  # maximum number of people in the ICU
    self.dt = 1            # time step

    # beta variation table, each corresponding to actions 0,1,2 respectively
    self.betaTable = (0.004,0.002,0.001)

    # Economic cost table, each corresponding to actions 0,1,2 respectively
    self.economicCost = (0,-10,-100)

    # SIR model initial conditions
    self.S0 = S0   # number of susceptibles at time = 0
    self.I0 = I0   # number of infectious at time = 0
    self.R0 = R0   # number of recovered (and immune) at time = 0

    # Define action and observation space
    # They must be gym.spaces objects
    totalPop = self.S0 + self.I0 + self.R0
    low  = np.array([0,0,0],dtype=np.float64)
    high = np.array([totalPop,totalPop,totalPop],dtype=np.float64)
    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(low, high,dtype=np.float64)

    # random seed
    self.seed()

    # initialize state
    self.state  = None

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):

    # Check for valid action
    err_msg = "%r (%s) invalid" % (action, type(action))
    assert self.action_space.contains(action), err_msg

    # R <--> python conversions
    numpy2ri.activate() # automatic conversion of numpy objects to rpy2 objects
    robjects.r('''
           source('../COVID19_models/SIR/SIR_example.R')
    ''') # source all R functions in the specified file
    sir_r = robjects.globalenv['sir_func'] # get R model

    # Update model based on actions
    self.beta = self.betaTable[action]

    # Unpack state
    S0 = self.state[0]
    I0 = self.state[1]
    R0 = self.state[2]

    # Plug in SIR model
    times = np.array([0,self.dt])
    modelOut = sir_r(self.beta, self.gamma, S0, I0, R0, times)
    S  = modelOut[1][1]
    I  = modelOut[1][2]
    R  = modelOut[1][3]

    # Update state
    self.state = (S,I,R)

    # Reward
    healthCost   = -1*I + -10*max(0,I - self.hospitalCap)
    economicCost = self.economicCost[action]
    reward = healthCost + economicCost

    # Observation
    observation = np.array(self.state)

    # Check if episode is over
    done = bool(
        I < 0.5
    )

    return observation, reward, done, {}

  def reset(self):
    # reset to initial conditions
    S = self.S0
    I = self.I0
    R = self.R0
    self.beta  = 0.004
    self.state = (S,I,R)
    observation = np.array(self.state)
    return observation  # reward, done, info can't be included
