# COVID19_RL

## Setup
Using python version 3.7.8   
Create virtual environment and install required packages :
```console
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```  

Make sure rpy2 is working and install the required R packages:  
```console
$ python installing_packages.py
```  
## Results
July 14 - first successful results on simple SIR model with DQN agent.   
See DQN_simple_SIR_results#.png, where the # is the number of  training episodes.

## COVID19_agents
This folder contains agents that train on COVID19_env. The first is a DQN agent
from Stable Baselines.

## COVID19_env
This folder will contain RL environments for COVID-19. The first environment is
called simple_SIR_env.py and is based on a minimal SIR model.

## COVID19_models
This folder contains COVID19 spread models in R to be used in the RL environments.   
A simple SIR model is implemented in SIR_example.R and can be called by running call_model.py.
```console
$ cd COVID19_models
$ python python call_model.py
```

## rpy2_examples
This folder is to test rpy2, such as calling custom R functions.     
The custom R functions are located in testFunc.R. They are called by running the Python file call_testFunc.py.
```console
$ cd rpy2_examples
$ python call_testFunc.py
```

## stable_baselines_examples
This folder contains relevant Stable Baselines RL examples for reference when
working on our COVID-19 RL implementation.
