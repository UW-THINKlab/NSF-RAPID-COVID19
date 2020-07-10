# SIR model tutorial from https://rpubs.com/choisy/sir

sir_1 <- function(beta, gamma, S0, I0, R0, times) {
  require(deSolve) # for the "ode" function

# differential equations
  sir_equations <- function(time, variables, parameters) {
    with(as.list(c(variables, parameters)), {
      dS <- -beta * I * S
      dI <-  beta * I * S - gamma * I
      dR <-  gamma * I
      return(list(c(dS, dI, dR)))
    })
  }

# parameters values:
  parameters_values <- c(beta  = beta, gamma = gamma)

# the initial values of variables:
  initial_values <- c(S = S0, I = I0, R = R0)

# solving
  out <- ode(initial_values, times, sir_equations, parameters_values)

# returning the output:
  as.data.frame(out)
}
