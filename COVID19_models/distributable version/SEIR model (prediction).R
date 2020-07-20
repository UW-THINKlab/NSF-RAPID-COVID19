library(BB)
library(truncnorm)
library(TruncatedNormal)

###The study period is from Jan. 21 to Mar. 22. In this code, Jan. 21 is viewed as day 1.
###The study region includes 14 metropolitan areas in the U.S.

###The directory where observed case data, population data and mobility data are stored
###All data has been pre-processed to metropolitan level
dataPath <- "C:/Users/benbernhard/Documents/GitHub/COVID19_RL/COVID19_models/distributable version/"

###Constants
set.seed(3)
start_date <- 1    #Starting date of model parameter estimation. Default is Jan. 21
end_date <- 62     #End date of model parameter estimation. Default is Mar. 22.
prediction_window <- 7    #How many days you want to predict forward after end_date? Default is 7 days.
number_of_particles <- 50000    #Number of particles in the particle filtering algorithm
initial_E_range <- c(0,50)      #The number of exposed people on Jan. 21 in each metro area is assumed to be a random number in the range 0~50.
initial_I <- 0    #No one was in the infectious compartment on Jan. 21
initial_R <- 0    #No one was in the recovered compartment on Jan. 21
##Parameters in SEIR model include transmission rate, latent period, and recovery rate. Values of latent period and recovery rate are taken from the literature, while transmission rate is to be estimated by the model.
##We assume a prior Gamma distribution for transmission rate (in each metro area) and update its distribution using particle filter.
prior_transmission_rate_mean <- 0.18    #Transmission rate assumed at the begining of particle filtering
prior_transmission_rate_var <- 0.01     #Variance in the transmission rate
latent_period <- 5.1       #Length of the latent period as in the literature
recovery_rate <- 0.06      #Recovery rate as in the literature
under_reporting <- 0.65    #The proportion of infectious people that will be reported
##The airplane_capacity and airplane_occupancy are used to convert number of flight into number of passengers traveling between two metro areas.
##airplane_capacity is assumed to decrease from 0.8 to 0.75 and to 0.5 as the pandemic goes on.
airplane_capacity <- 200
airplane_occupancy <- c(0.8,0.75,0.5)    #Airplane occupancy in Jan, Feb and Mar, respectively

###Helper function for calculating column products of a matrix
colSD <- function(m) return(apply(m, 2, sd))

###Importing the observed case data and metropolitan population data
dataset_cases <- read.csv(paste(dataPath,"cases_data.csv",sep=""))
dataset_population <- read.csv(paste(dataPath,"population.csv",sep=""))
##Processing the imported data
dataset_cases[is.na(dataset_cases)] <- 0    #If a metropolitan area has no case on a given day, set the value to 0.
city_names <- as.character(dataset_population$City)    #Names of the metropolitan areas
num_cities <- length(city_names)    #How many metro areas are included in this study
pop_data <- dataset_population$Population    #Populations in the metropolitan areas
case_data <- as.matrix(dataset_cases[,match(city_names,colnames(dataset_cases))])    #Reorder the case data of different metros to correspond to the order of metros in the variable "city_names"

###Importing mobility data
###This data includes the number of flights between each two metro areas on each day, and is converted to the number of passengers between two metro areas on each day.
###The data is first imported into an R list indexed by metro names, and then organized into an R list indexed by dates.
passenger_flow_data_temp <- list()
for (a_city in city_names) {
  flow_city_temp <- read.csv(paste(dataPath, "trip_data/" ,a_city, ".csv", sep=""))    #Outflows from the given a_city to other cities on different dates
  flow_city_temp[,a_city] <- 0    #Outflow from a_city to itself is assumed to be 0.
  flow_city_temp[1:11,-1] <- flow_city_temp[1:11,-1]*airplane_capacity*airplane_occupancy[1]    #number of passengers in Jan
  flow_city_temp[12:40,-1] <- flow_city_temp[12:40,-1]*airplane_capacity*airplane_occupancy[2]    #number of passengers in Feb
  flow_city_temp[41:62,-1] <- flow_city_temp[41:62,-1]*airplane_capacity*airplane_occupancy[3]    #number of passengers in Mar
  passenger_flow_data_temp[[a_city]] <- as.matrix(flow_city_temp[,match(city_names,colnames(flow_city_temp))])    #Reorder the matrix columns to ensure consistency with the metro order in city_names
}
##Reorganize the data list from being metro-indexed to date-indexed
##Final output is an R list. Every item in the list is an OD matrix among the 14 metro areas on a given date (list index).
passenger_flow_data <- list()
for (i in 1:62) {
  passenger_flow_data[[i]] <- matrix(0,num_cities,num_cities)
  colnames(passenger_flow_data[[i]]) <- city_names
  rownames(passenger_flow_data[[i]]) <- city_names
  for (ci in city_names) {
    passenger_flow_data[[i]][ci,] <- passenger_flow_data_temp[[ci]][i,]
  }
}

###Importing estimated parameters (particles) and states
###The states data needs to be converted to an R list.
particle_data <- read.csv(paste(dataPath,"particles/",end_date,".csv",sep=""))
states_data <- read.csv(paste(dataPath,"states/",end_date,".csv",sep=""))
particles <- as.matrix(particle_data)
states_cities <- list()
for (city_i in 1:num_cities) {
  states_cities[[city_i]] <- as.matrix(states_data[(4*city_i-3):(4*city_i)])
}

###Prediction-----------------------------------------------------------------------------------------------------------------------------------------------------
###Across the prediction window, the parameter still evolves.
###The prediction outputs the number of infected people in each metro area on each day during the prediction window.

##Extending mobility data to the prediction window, assuming repeating weekly patterns of mobility
for (a_day in (end_date+1):(end_date+prediction_window)) {
  passenger_flow_data[[a_day]] <- passenger_flow_data[[a_day-7]]
}

infection_history <- matrix(NA,prediction_window,num_cities)
for (a_day in (end_date+1):(end_date+prediction_window)) {
  #Evolving the parameters for each metro area by adding random noises
  for (city_i in 1:num_cities) {
    parameter_mean <- particles[,city_i]
    particles[,city_i] <- rtruncnorm(number_of_particles, a=0, b=Inf, mean=parameter_mean, sd=sqrt(prior_transmission_rate_var))
  }

  #Getting the inflow and outflow of people in each compartment to and from each metro area
  #It is assumed that the comaprtment composition in the passenger flows is the same as the compartment composition in the trip origin.
  mean_states <- sapply(1:num_cities,function(x) colMeans(states_cities[[x]]))
  trip_matrix <- passenger_flow_data[[a_day]]    #The entry at i's row and j' column is the number of passengers traveling from metro i to metro j.
  outflow_S <- rep(NA,num_cities)
  inflow_S <- rep(NA,num_cities)
  outflow_E <- rep(NA,num_cities)
  inflow_E <- rep(NA,num_cities)
  outflow_I <- rep(NA,num_cities)
  inflow_I <- rep(NA,num_cities)
  outflow_R <- rep(NA,num_cities)
  inflow_R <- rep(NA,num_cities)
  for (city_i in 1:num_cities) {
    outflow_S[city_i] <- sum(trip_matrix[city_i,])*mean_states[1,city_i]/pop_data[city_i]
    outflow_E[city_i] <- sum(trip_matrix[city_i,])*mean_states[2,city_i]/pop_data[city_i]
    outflow_I[city_i] <- sum(trip_matrix[city_i,])*mean_states[3,city_i]/pop_data[city_i]
    outflow_R[city_i] <- sum(trip_matrix[city_i,])*mean_states[4,city_i]/pop_data[city_i]
    inflow_S[city_i] <- sum(trip_matrix[,city_i]*mean_states[1,]/pop_data)
    inflow_E[city_i] <- sum(trip_matrix[,city_i]*mean_states[2,]/pop_data)
    inflow_I[city_i] <- sum(trip_matrix[,city_i]*mean_states[3,]/pop_data)
    inflow_R[city_i] <- sum(trip_matrix[,city_i]*mean_states[4,]/pop_data)
  }

  for (city_i in 1:num_cities) {
    #Updating the states for each metro area according to the SEIR model
    delta_S <- -particles[,city_i]*states_cities[[city_i]][,3]*states_cities[[city_i]][,1]/pop_data[city_i] + inflow_S[city_i] - outflow_S[city_i]
    delta_E <- particles[,city_i]*states_cities[[city_i]][,3]*states_cities[[city_i]][,1]/pop_data[city_i] - states_cities[[city_i]][,2]/latent_period + inflow_E[city_i] - outflow_E[city_i]
    delta_I <- states_cities[[city_i]][,2]/latent_period - states_cities[[city_i]][,3]*recovery_rate + inflow_I[city_i] - outflow_I[city_i]
    delta_R <- states_cities[[city_i]][,3]*recovery_rate + inflow_R[city_i] - outflow_R[city_i]

    states_cities[[city_i]][,1] <- states_cities[[city_i]][,1] + delta_S
    states_cities[[city_i]][,2] <- states_cities[[city_i]][,2] + delta_E
    states_cities[[city_i]][,3] <- states_cities[[city_i]][,3] + delta_I
    states_cities[[city_i]][,4] <- states_cities[[city_i]][,4] + delta_R

    infection_history[a_day-end_date,city_i] <- mean(states_cities[[city_i]][,2] + states_cities[[city_i]][,3])
  }
}

#Export output to file
colnames(infection_history) <- city_names
write.csv(infection_history,paste(dataPath,"predicted_infections.csv",sep=""))
