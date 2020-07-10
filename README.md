# COVID19-RL

## Setup
Using python version 3.7.8   
Create virtual environment and install required packages :
```console
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```  

Now, make sure rpy2 is working and install the required R packages:  
```console
$ python installing_packages.py
```  

## rpy2_examples
This folder is to test rpy2, such as calling custom R functions.     
The custom R functions are located in testFunc.R. They are called by running the Python file call_testFunc.py.
```console
$ cd rpy2_examples
$ python call_testFunc.py
```

## COVID19_models
This folder contains COVID19 spread models in R to be used in the RL environments.   
A simple SIR model is implemented in SIR_example.R and can be called by running call_model.py.
```console
$ cd COVID19_models
$ python python call_model.py
```

## COVID19_env
This folder will contain RL environments for COVID-19. The first environment is
based on COVID19_models/SIR_example.R and is not yet completed.

## stable_baselines_examples
This folder contains relevant Stable Baselines RL examples for reference when
working on our COVID-19 RL implementation.
