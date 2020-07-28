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

class SEIR_env(gym.Env):
  """
  Description:
        Each city's population is broken down into four compartments --
        Susceptible, Exposed, Infectious, and Removed -- to model the spread of
        COVID-19.

  Source:
        SEIR model from https://github.com/UW-THINKlab/SEIR/
        Code modeled after cartpole.py from
         github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

  Observation:
        Type: Box(4,)
        Num     Observation       Min     Max
        0       Susceptible       0       Total Population
        1       Exposed           0       Total Population
        2       Infected          0       Total Population
        3       Recovered         0       Total Population

  Actions:
        Type: Box(14,14), min=0 max=1
        Num     Action                          Change in model
        1       No travel restriction           reduction_factor = 1
        0       Complete travel restriction     reduction_factor = 0

  Reward:
        reward = health cost + economic cost

        Health cost:
            -1 for every infected case
           -10 for every infected case that brings total infected count above a
                specified maximum (hospital capacity)
        Economic cost:
             0 for every day using 'open everything'
           -10 for every day using 'open at half capacity'
          -100 for every day using 'stay at home order'

  Starting State:
        Susceptible:
        Infected:
        Recovered:

  Episode Termination:
        Number of infections is zero
        Episode length (time) reaches specified maximum (end time)
  """


  metadata = {'render.modes': ['human']}

  def __init__(self, S0, I0, R0, hospitalCapacity):
    super(SEIR_env, self).__init__()

    # SIR model parameters
    self.beta  = 0.004     # infectious contact rate (/person/day)
    self.gamma = 0.5       # recovery rate (/day)
    self.hospitalCap = hospitalCapacity  # maximum number of people in the ICU
    self.dt = 1            # time step

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
           source("../COVID19_models/SEIR/read_input.R")
           source("../COVID19_models/SEIR/seir_r.R")
    ''') # source all R functions in the specified file
    seir_r = robjects.globalenv['seirPredictions'] # get R model

    # Update model based on actions
    self.beta = self.betaTable[action]

    # Unpack state
    S0 = self.state[0]
    I0 = self.state[1]
    R0 = self.state[2]

    # Plug in SEIR model
    times = np.array([0,self.dt])

    modelOut = seir_r(reduction_factor,
                      beta_data    = input_data.rx2("beta_data"),
                      beta_sd_data = input_data.rx2("beta_sd_data"),
                      latent       = input_data.rx2("latent"),
                      gamma        = input_data.rx2("gamma"),
                      St_data      = input_data.rx2("St_data"),
                      Et_data      = input_data.rx2("Et_data"),
                      It_data      = input_data.rx2("It_data"),
                      Rt_data      = input_data.rx2("Rt_data"),
                      ODs          = input_data.rx2("ODs"),
                      pops         = input_data.rx2("pops"),
                      current      = input_data.rx2("current"),
                      pred         = input_data.rx2("pred"),
                      city_names   = input_data.rx2("city_names"))

    S  = np.array(modelOut.rx2("S"))
    E  = np.array(modelOut.rx2("E"))
    I  = np.array(modelOut.rx2("I"))
    R  = np.array(modelOut.rx2("R"))

    # Update state
    self.state = (S,E,I,R)

    # Reward
    overflowI = I - self.hospitalCap
    healthCost   = -1*I + -10*sum(overflowI>0)
    economicCost = -(10*(1-action))**2
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
