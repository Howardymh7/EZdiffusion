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
    def __init__(self):
        return None

    def simulate_Summary_Stat(self,a, v, t):
#Foward Equations, Simulate Summary Statistics using parameters based on the EZdiffusion model
        y = np.exp(-a*v)
        Rpred = 1 / (1+y) #---------------------------------1
        Mpred = t + (a/(2*v))*((1-y)/(1+y)) #---------------2
        Vpred = (a / (2*v**3))*((1-2*a*v*y-y**2)/(y+1)**2) #3

        return Rpred, Mpred, Vpred

    def simulate_parameter(self, R_obs, Mobs, Vobs):
#Inverse Equations, Simulate the estimated Drift Rate, Boundary Separation, and Nondecision time values
        if self.R_obs == 1:
            self.R_obs = 0.999
            L = np.log(self.R_obs / (1 - self.R_obs))
        elif self.R_obs < 0.1:
            self.R_obs = 0.001
            L = np.log(self.R_obs / (1 - self.R_obs))
        else:
            L = np.log(self.R_obs / (1 - self.R_obs))
        v_est = np.sign(self.R_obs - 1/2)*np.power( L*(self.R_obs**2*L-self.R_obs*L+self.R_obs-1/2)/Vobs, 1/4) #----------------4
        if v_est == 0:
            v_est = 0.001
        a_est = L/v_est #----------------------------------------------------------------5
        t_est = Mobs - (a_est/2*v_est)*((1-np.exp(-v_est*a_est))/(1+np.exp(-v_est*a_est))) #6
        return v_est, a_est, t_est

    def simulate_observed_data(self,Rpred,Mpred,Vpred,N):
#Predict the observed Values of Numbers of Correct Trials, Mean Response Time, and Variance Response Time
        Tobs = np.random.binomial(N,Rpred) #------------7
        R_obs = Tobs/N
        self.R_obs = R_obs
        Mobs = np.random.normal(Mpred, Vpred/N) #------8
        Vobs = np.random.gamma((N-1)/2, 2*Vpred/(N-1)) #9
        return Tobs, Mobs, Vobs, R_obs
    
    def recover_ez_parameters(self,a,v,t,N):
        b =[]
        for i in range(1000):
            Rpred, Mpred, Vpred = self.simulate_Summary_Stat(a, v, t)
            Tobs, Mobs, Vobs, Robs= self.simulate_observed_data(Rpred,Mpred,Vpred,N)
            v_est, a_est, t_est = self.simulate_parameter(Tobs, Mobs, Vobs)
            interString = round(float(v-v_est),6), round(float(a-a_est),6), round(float(t-t_est),6)
            b.append(sum(interString))
        
        EstimatedBias = sum(b)/1000
        b_Sq = EstimatedBias**2
        return EstimatedBias,b_Sq
    
# Run the command
ez = EZDiffusionModel()
a = np.random.uniform(0.5, 2)
v = np.random.uniform(0.5, 2)
t = np.random.uniform(0.1, 0.5)
b=[]
bsq=[]
Nvalues = [10,40,4000]
for i in range(len(Nvalues)):
    N = Nvalues[i]
    b_int,bsq_int = ez.recover_ez_parameters(a,v,t,N)
    b.append(b_int)
    bsq.append(bsq_int)
    print("when N is", Nvalues[i] , "b = ",b_int,"b^2 = ", bsq_int)

