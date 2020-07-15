from stable_baselines.common.env_checker import check_env
from simple_SIR_env import simple_SIR_env

# Initiate the env
S0 = 999  # number of susceptibles at time = 0
I0 =   1  # number of infected at time = 0
R0 =   0  # number of recovered (and immune) at time = 0
hospitalCapacity = 300 # maximum number of people in the ICU
env = simple_SIR_env(S0, I0, R0, hospitalCapacity)

# Check the environment
check_env(env)

'''
Note: you may get many warnings about future versions of some python packages,
but from my experience, you can ignore these warnings. Will work on resolving
this.
'''
