"""
custom environment must follow the gym interface
skeleton copied from:
    https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
"""
import sys
sys.path.append("..")

import os

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
        Type: Box(4,14)
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

  def __init__(self, hospitalCapacity):
    super(SEIR_env, self).__init__()

    # R <--> python conversions
    numpy2ri.activate() # automatic conversion of numpy objects to rpy2 objects
    robjects.r('''
           source("../COVID19_models/SEIR/read_input.R")
           source("../COVID19_models/SEIR/seir_r.R")
    ''') # source all R functions in the specified file
    getData_r = robjects.globalenv['getData']
    seir_r = robjects.globalenv['seirPredictions'] # get R model

    # Get input data
    input_data = getData_r("../COVID19_models/SEIR/RL_input")
    os.chdir("../../../COVID19_agents")

    # SEIR model inputs
    self.hospital_cap = hospitalCapacity
    self.beta_data    = input_data.rx2("beta_data")
    self.beta_sd_data = input_data.rx2("beta_sd_data")
    self.latent       = input_data.rx2("latent")
    self.gamma        = input_data.rx2("gamma")
    self.St_data      = np.array(input_data.rx2("St_data"))
    self.Et_data      = np.array(input_data.rx2("Et_data"))
    self.It_data      = np.array(input_data.rx2("It_data"))
    self.Rt_data      = np.array(input_data.rx2("Rt_data"))
    self.ODs          = input_data.rx2("ODs")
    self.pops         = input_data.rx2("pops")
    self.current      = int(input_data.rx2("current")[0])
    self.pred         = int(input_data.rx2("pred")[0])
    self.city_names   = input_data.rx2("city_names")

    # Save intial conditions for reset
    self.current0 = self.current
    self.St_data0 = self.St_data
    self.Et_data0 = self.Et_data
    self.It_data0 = self.It_data
    self.Rt_data0 = self.Rt_data

    # Define action and observation space
    # They must be gym.spaces objects
    num_cities = len(self.city_names)
    self.num_cities = num_cities
    #low_a = np.zeros((num_cities,num_cities),np.float16)
    #high_a = np.ones((num_cities,num_cities),np.float16)
    self.action_space = spaces.Box(low=-1, high=1,shape=(num_cities*num_cities,),dtype=np.float16)
    #low_o  = np.zeros((4,num_cities),np.float16)
    #high_o = np.inf*np.ones((4,num_cities),np.float16)
    self.observation_space = spaces.Box(0, np.inf,shape=(4*num_cities,),dtype=np.float64)

    # random seed
    self.seed()

    # initialize state
    self.state  = np.empty(shape=(4*num_cities,))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    # Check for valid action
    err_msg = "%r (%s) invalid" % (action, type(action))
    assert self.action_space.contains(action), err_msg

    # R <--> python conversions
    numpy2ri.activate() # automatic conversion of numpy objects to rpy2 objects

    # Update model based on actions
    action = (action+1)/2
    reduction_factor = np.reshape(action,(self.num_cities,self.num_cities))

    # Get model
    seir_r = robjects.globalenv['seirPredictions'] # get R model

    # Plug in SEIR model
    modelOut = seir_r(reduction_factor,
                      beta_data    = self.beta_data,
                      beta_sd_data = self.beta_sd_data,
                      latent       = self.latent,
                      gamma        = self.gamma,
                      St_data      = self.St_data,
                      Et_data      = self.Et_data,
                      It_data      = self.It_data,
                      Rt_data      = self.Rt_data,
                      ODs          = self.ODs,
                      pops         = self.pops,
                      current      = self.current,
                      pred         = self.pred,
                      city_names   = self.city_names)

    # Unpack output
    S  = np.array(modelOut.rx2("S"))
    E  = np.array(modelOut.rx2("E"))
    I  = np.array(modelOut.rx2("I"))
    R  = np.array(modelOut.rx2("R"))

    # Update state
    self.state = np.matrix((S,E,I,R))
    self.current += self.pred
    self.St_data[self.current] = S
    self.Et_data[self.current] = E
    self.It_data[self.current] = I
    self.Rt_data[self.current] = R

    # Reward
    overflowI    = I - self.hospital_cap
    healthCost   = -1*sum(I) + -10*sum(overflowI>0)
    economicCost = np.sum(-(10*(1-action))**2)
    reward = healthCost + economicCost

    # Observation
    observation = np.reshape(self.state,(4*self.num_cities,))

    # Check if episode is over
    done = bool(
        all(I < 0.5) or
        self.current >= 62 - self.pred
    )
    return observation, reward, done, {}

  def reset(self):

    # reset to initial conditions
    self.current = self.current0
    self.St_data = self.St_data0
    self.Et_data = self.Et_data0
    self.It_data = self.It_data0
    self.Rt_data = self.Rt_data0

    self.state = np.matrix((self.St_data[self.current],
                            self.Et_data[self.current],
                            self.It_data[self.current],
                            self.Rt_data[self.current]),dtype=np.float64)
    observation = np.reshape(self.state,(4*self.num_cities,))

    return observation  # reward, done, info can't be included
