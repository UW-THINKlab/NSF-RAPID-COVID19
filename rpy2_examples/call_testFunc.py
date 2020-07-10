# test rpy2 ability to call custom R functions in python

import rpy2.robjects as robjects

# source all R functions in the specified file
robjects.r('''
       source('testFunc.r')
''')

# get functions
r_hello_world = robjects.globalenv['hello_world']
r_fahrenheit_to_celsius = robjects.globalenv['fahrenheit_to_celsius']

# call functions, print result
print('r_hello_world')
r_hello_world("Ben")

print('r_fahrenheit_to_celsius')
print(r_fahrenheit_to_celsius(32))
