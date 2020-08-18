import sys
sys.path.append("..")

from SEIR_env import SEIR_env
import rl_baselines_zoo
import gym



train.py --algo a2c --env SEIR-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler tpe --pruner median
