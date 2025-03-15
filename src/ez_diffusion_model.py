import numpy as np
import pandas as pd

# N = Numbers of Observations
#Summary Statistics
# R = Accuracy Rate
# M = Mean Response Time
# V = Variance Response Time
# T = Numbers of Correct Trials
#Parameters
# v = Drift Rate
# a = Boundary Separation
# t = Non-Decision Time

def simulate_Summary_Stat(a, v, t, N):
#Foward Equations, Simulate Summary Statistics using parameters based on the EZdiffusion model
    y = np.exp(-a*v)
    Rpred = 1 / (1+y) #---------------------------------1
    Mpred = t + (a/(2*v))*((1-y)/(1+y)) #---------------2
    Vpred = (a / (2*v**3))*((1-2*a*v*y-y**2)/(y+1)**2) #3
    return Rpred, Mpred, Vpred

def simulate_parameter(R, M, V):
#Inverse Equations, Simulate the estimated Drift Rate, Boundary Separation, and Nondecision time values
    L = np.log(R/(1-R))
    v_est = np.sign(R - 1/2)*np.power( L*(R**2*L-R*L+R-1/2)/V, 1/4) #----------------4
    a_est = L/v_est #----------------------------------------------------------------5
    t_est = M - (a_est/2*v_est)*((1-np.exp(-v_est*a_est))/(1+np.exp(-v_est*a_est))) #6
    return v_est, a_est, t_est

def simulate_observed_data(Rpred,Mpred,Vpred,N):
#Predict the observed Values of Numbers of Correct Trials, Mean Response Time, and Variance Response Time
    Tobs = np.random.binomial(Rpred,N) #------------7
    Mobs = np.random.normal(Mpred , Vpred/N) #------8
    Vobs = np.random.gamma((N-1)/2, 2*Vpred/(N-1)) #9
    return Tobs, Mobs, Vobs
    
def recover_ez_parameters(N):
#Recover test
    


# Print results
print("Bias for N=10:", bias_10)
print("MSE for N=10:", mse_10)
print("Bias for N=40:", bias_40)
print("MSE for N=40:", mse_40)
print("Bias for N=4000:", bias_4000)
print("MSE for N=4000:", mse_4000)
