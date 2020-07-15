# COVID19_RL

## Setup
The following setup was used in context of macOS   

Python version 3.7.8   
R version 4.0.2   

After cloning the GitHub repository, create virtual environment and install the
required packages:   
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
This folder contains images/plots of training results.
See DQN_simple_SIR_results#.png, where the # is the number of  training episodes.  

Most recent results:

![DQN SIR](/Results/DQN_simple_SIR_results50000.png)

## COVID19_agents
This folder contains agents that train on COVID19_env. The first is a DQN agent
from Stable Baselines (DQN_simple_SIR.py). To train this agent and test it on
the simple SIR environment, simply run the python script:
```console
$ cd COVID19_agents
$ python DQN_simple_SIR.py
```   

Stable Baselines agents use Tensorflow and support the use of Tensorboard to monitor results. To monitor training progress while training an agent, run the following in a separate terminal:
```console
$ tensorboard --logdir ./DQN_SIR_tensorboard/
```  
Replace './DQN_SIR_tensorboard/' with whatever name is specified for
tensorboard_log in the agent's .py file.

## COVID19_env
This folder contains RL environments for COVID-19, which are used by COVID19_agents. The first environment is called simple_SIR_env.py and is based
on a minimal SIR model.   

These environments are written in Python but utilize dynamics models written in
R (e.g. simple_SIR_model.R). The conversions between Python and R are handled by  rpy2. See the rpy2_examples folder for examples and the following website for more documentation: https://rpy2.github.io/doc/latest/html/index.html

## COVID19_models
This folder contains COVID19 spread models in R to be used in the RL environments. The purpose of this folder is to have a place to test the R files before they are integrated into the RL environments.  

A simple SIR model is implemented in SIR_example.R and can be tested by running call_model.py.
```console
$ cd COVID19_models
$ python call_model.py
```

## rpy2_examples
This folder is to test rpy2, such as calling custom R functions.     
The custom R functions are located in testFunc.R. They are called by running the Python file call_testFunc.py.
```console
$ cd rpy2_examples
$ python call_testFunc.py
```

rpy2 docuentation: https://rpy2.github.io/doc/latest/html/index.html

## stable_baselines_examples
This folder contains relevant Stable Baselines RL examples for reference when
working on our COVID-19 RL implementation.  


Stable Baselines documentation: https://stable-baselines.readthedocs.io/en/master/
