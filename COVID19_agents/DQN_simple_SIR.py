import sys
sys.path.append("..")
from COVID19_env.simple_SIR_env import simple_SIR_env
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

# Initiate the env
env = simple_SIR_env()

# Define and Train the agent
model = DQN(MlpPolicy,env, tensorboard_log="./DQN_SIR_tensorboard/")
model.learn(total_timesteps=50000)

# Load trained agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

'''
run the following in separate terminal to monitor results:
    tensorboard --logdir ./DQN_SIR_tensorboard/
'''
