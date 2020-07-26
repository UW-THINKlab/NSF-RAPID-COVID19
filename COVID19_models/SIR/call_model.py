# call the SIR model (writting in R) in python

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

# automatic conversion of numpy objects into rpy2 objects
numpy2ri.activate()

# source all R functions in the specified file
robjects.r('''
       source('SIR_example.R')
''')

# get functions
sir_1_r = robjects.globalenv['sir_func']

# specify parameters
beta  = 0.004 # infectious contact rate (/person/day)
gamma = 0.5    # recovery rate (/day)

# specify initial initial_values
S0 = 999  # number of susceptibles at time = 0
I0 =   1  # number of infectious at time = 0
R0 =   0   # number of recovered (and immune) at time = 0

# time
times = np.linspace(0, 10, 11)

# call functions, print result
SIR_df = sir_1_r(beta, gamma, S0, I0, R0, times)
print(" time        S               I             R")
print(SIR_df)
