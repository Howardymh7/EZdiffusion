import numpy as np

# N = Numbers of Observations

#Summary Statistics
# R = Accuracy Rate
# M = Mean Response Time
# V = Variance Response Time
# T = Numbers of Correct Trials
# b = Estimate bias
# b^2 = Squared Error

#Parameters
# v = Drift Rate
# a = Boundary Separation
# t = Non-Decision Time
class EZDiffusionModel:

    def simulate_Summary_Stat(self, a, v, t):
#Foward Equations, Simulate Summary Statistics using parameters based on the EZdiffusion model
        y = np.exp(-a*v)
        Rpred = 1 / (1+y) #---------------------------------1
        Mpred = t + (a/(2*v))*((1-y)/(1+y)) #---------------2
        Vpred = (a / (2*v**3))*((1-2*a*v*y-y**2)/(y+1)**2) #3
        return Rpred, Mpred, Vpred

    def simulate_parameter(self, Robs, Mobs, Vobs):
#Inverse Equations, Simulate the estimated Drift Rate, Boundary Separation, and Nondecision time values
        L = np.log(Robs/(1-Robs))
        v_est = np.sign(Robs - 1/2)*np.power( L*(Robs**2*L-Robs*L+Robs-1/2)/Vobs, 1/4) #----------------4
        a_est = L/v_est #----------------------------------------------------------------5
        t_est = Mobs - (a_est/2*v_est)*((1-np.exp(-v_est*a_est))/(1+np.exp(-v_est*a_est))) #6
        return v_est, a_est, t_est

    def simulate_observed_data(self, Rpred,Mpred,Vpred,N):
#Predict the observed Values of Numbers of Correct Trials, Mean Response Time, and Variance Response Time
        Tobs = np.random.binomial(Rpred,N) #------------7
        Mobs = np.random.normal(Mpred , Vpred/N) #------8
        Vobs = np.random.gamma((N-1)/2, 2*Vpred/(N-1)) #9
        return Tobs, Mobs, Vobs
    
    def recover_ez_parameters(self, a,v,t,N):
#Recover test, select any a, v, t values that fits the range
        Rpred, Mpred, Vpred = self.simulate_Summary_Stat(a, v, t)
        Tobs, Mobs, Vobs = self.simulate_observed_data(Rpred,Mpred,Vpred,N)
        v_est, a_est, t_est = self.simulate_parameter(Tobs, Mobs, Vobs)
        b = (v,a,t)-(v_est, a_est, t_est)
        b_sq = b^2
        return b,b_sq