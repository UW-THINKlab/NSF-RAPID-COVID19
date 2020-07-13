# call the SIR model (writting in R) in python

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

# automatic conversion of numpy objects into rpy2 objects
numpy2ri.activate()

# source all R functions in the specified file
robjects.r('''
       source('simple_SIR_model.R')
''')

# get functions
sir_r = robjects.globalenv['sir_func']

# specify parameters
beta  = 0.004 # infectious contact rate (/person/day)
gamma = 0.5    # recovery rate (/day)

# specify initial initial_values
S0 = 999  # number of susceptibles at time = 0
I0 =   1  # number of infectious at time = 0
R0 =   0   # number of recovered (and immune) at time = 0

# time
dt = np.array([0,1])

# call functions, print result
print("    S               I             R")
for t  in range(10):
  out = sir_r(beta, gamma, S0, I0, R0, dt)
  S = out[1][1]
  I = out[1][2]
  R  = out[1][3]
  print(S,I,R)
  S0 = S
  I0 = I
  R0 = R
