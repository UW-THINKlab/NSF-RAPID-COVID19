from stable_baselines.common.env_checker import check_env
from simple_SIR_env import simple_SIR_env

env = simple_SIR_env()
check_env(env)
