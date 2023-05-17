import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})
import pdb 
def solve_mean_lagrange(x, mu, lam=0, max_iter = 20, tol = 1e-15):
    #Implements the Newton Raphson method:
    i = 0
    old_lam = lam

    def fx_mean(lam1, x, mu):
        return mu/np.exp(-1) - np.dot( x , np.exp(-lam1 * x) )

    def dxfx_mean(lam1, x):
        return np.dot( x**2 , np.exp(-lam1 * x)) + np.exp(-lam1 * x).sum()

    while abs( fx_mean(lam, x, mu) ) > tol: #run the helper function and check

        lam = old_lam - fx_mean(lam, x, mu)/dxfx_mean(lam,x)  # Newton-Raphson equation
        #print("Iteration" + str(i) + ": x = " + str(lam) + ", f(x) = " +  str( fx(lam, x, mu) ) )  
          
        old_lam = lam
        i += 1
        
        if i > max_iter:
          break

    return torch.tensor(lam)

def solve_var_lagrange(x, var, lam=0, max_iter = 20, tol = 1e-15):
    #Implements the Newton Raphson method:
    i = 0
    old_lam = lam

    def fx_var(lam2, x, var):
        return var/np.exp(-1) - np.dot( x**2 , np.exp(-lam2 * x**2 ) )

    def dxfx_var(lam2, x):
        return np.dot( x**4, np.exp(-lam2 * x**2 ) ) + np.exp(-lam2 * x**2 ).sum()

    while abs( fx_var(lam, x, var) ) > tol: #run the helper function and check
        lam = old_lam - fx_var(lam, x, var)/dxfx_var(lam, x)  # Newton-Raphson equation
        #print("Iteration" + str(i) + ": x = " + str(lam) + ", f(x) = " +  str( fx_var(lam, x, var) ) )  
          
        old_lam = lam
        i += 1
        
        if i > max_iter:
          break

    return torch.tensor(lam)

### Solve for multiple Lagrange multipliers (System of non-linear Equations) using Jacobian ###
def solve_multiple_lagrange(x, mu, var, lam1=0, lam2=0, max_iter = 1000, tol = 1e-15):

    def fx_1(lam1, lam2, x, mu):
        C = x - mu
        return mu/np.exp(-1) - np.dot( x , np.exp(-lam1 * x - lam2 * C**2) )

    def fx_2(lam1, lam2, x, mu, var):
        C = x - mu
        return var/np.exp(-1) - np.dot( C**2 , np.exp(-lam1 * x -lam2 * C**2 ) )

    def dxfx1_lam1(lam1, lam2, x, mu, var):
        C = (x - mu)
        return np.dot( x**2 , np.exp(-lam1 * x - lam2 * C**2)) + np.exp(-lam1 * x - lam2 * C**2).sum()

    def dxfx1_lam2(lam1, lam2, x, mu, var):
        C = (x - mu)
        return np.dot( x*C**2 , np.exp(-lam1 * x - lam2 * C**2)) + np.exp(-lam1 * x - lam2 * C**2).sum()

    def dxfx2_lam2(lam1, lam2, x, mu, var):
        C = (x - mu)
        return np.dot( C**4 , np.exp(-lam1 * x - lam2 * C**2)) + np.exp(-lam1 * x - lam2 * C**2).sum()
    
    def compute_fx(lam1, lam2, x, mu, var):
        b = np.empty([2, 1])
        #print(b, b.shape)
        b[0,0] = fx_1(lam1, lam2, x, mu)
        b[1,0] = fx_2(lam1, lam2, x, mu, var)
        
        return -b #this is negative, important!! easy to miss

    def compute_jacobian(lam1, lam2, x, mu):
        J = np.zeros([2,2])
        J[0,0] = dxfx1_lam1(lam1, lam2, x, mu, var)
        J[0,1] = dxfx1_lam2(lam1, lam2, x, mu, var)
        J[1,0] = dxfx1_lam2(lam1, lam2, x, mu, var)
        J[1,1] = dxfx2_lam2(lam1, lam2, x, mu, var)
        return J
    
    lam = torch.tensor([lam1, lam2]).reshape(2,1)
    i = 0
    old_lam = lam

    while abs( compute_fx(lam1, lam2, x, mu, var).mean() ) > tol: #run the helper function and check

        lam1, lam2 = old_lam
        #Put initial guess
        b = compute_fx(lam1, lam2, x, mu, var)
        J = compute_jacobian(lam1, lam2, x, mu)

        # Newton-Raphson equation
        d = np.linalg.solve(J, b) #Jx = b -> x = J^-1b
        lam = torch.tensor(d) + old_lam
        #print("Iteration" + str(i) + ": x = " + str(lam) + ", f(x) = " +  str( compute_fx(lam1, lam2, x, mu, var) ) )

        old_lam = lam
        i += 1
        
        if i > max_iter:
          break

    return torch.tensor(lam[0]), torch.tensor(lam[1])

class MeanMaxEnt_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(MeanMaxEnt_Model, self).__init__()
        #self.lam = lam

    def predict_distribution(self, x, lam1):
        return np.exp( -1 -lam1 * x)/np.exp( -1 -lam1 * x).sum()

    def fx_mean(self, lam1, x, mean):
        return mean/np.exp(-1) - np.dot(x, np.exp(-lam1 * x))/np.exp( -1 -lam1 * x).sum()

    def dxfx_mean(self, lam1, x):
        return np.sum( x**2 * np.exp(-lam1 * x)/np.exp( -1 -lam1 * x).sum() ) + np.sum( np.exp(-lam1 * x)/np.exp( -1 -lam1 * x).sum())

    def forward(self, x, mu, lam=0, max_iter = 100, tol = 1e-15):
        #Implements the Newton Raphson method:
        i = 0
        old_lam = lam

        while abs( self.fx_mean(lam, x, mu) ) > tol: #run the helper function and check

          lam = old_lam - self.fx_mean(lam, x, mu)/self.dxfx_mean(lam,x)  # Newton-Raphson equation
          
          old_lam = lam
          i += 1
        
          if i > max_iter:
            break

        self.lam = lam

class VarianceMaxEnt_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(VarianceMaxEnt_Model, self).__init__()
        #self.lam = lam

    def predict_distribution(self, x, mu, lam2):
        C = (x - mu)
        return np.exp( -1 -lam2 * C**2)/np.exp( -1 -lam2 * C**2).sum()

    def fx_var(self, lam2, x, mu, var):
        C = (x - mu)
        return var/np.exp(-1) - np.dot(C**2 , np.exp(-lam2 * C**2 ) )/ np.exp(-1 -lam2 * C**2 ).sum()

    def dxfx_var(self, lam2, x, mu):
        C = (x - mu)
        return np.sum( C**4 * np.exp(-lam2 * C**2 )/np.exp( -1 -lam2 * C**2).sum() ) + np.sum( np.exp(-lam2 * C**2 )/np.exp( -1 -lam2 * C**2).sum())

    def forward(self, x, mu, var, lam=0, max_iter = 100, tol = 1e-15):
        #Implements the Newton Raphson method:
        i = 0
        old_lam = lam

        while abs( self.fx_var(lam, x, mu, var) ) > tol: #run the helper function and check

          lam = old_lam - self.fx_var(lam, x, mu, var)/self.dxfx_var(lam, x, mu)  # Newton-Raphson equation
          #print("Iteration" + str(i) + ": x = " + str(lam) + ", f(x) = " +  str( self.fx_var(lam, x, mu, var) ) )  
          
          old_lam = lam
          i += 1
        
          if i > max_iter:
            break

        self.lam = lam

class VarianceMaxEnt_Model_1(nn.Module):
    def __init__(self, num_classes=10):
        super(VarianceMaxEnt_Model_1, self).__init__()
        #self.lam = lam

    def predict_distribution(self, x, lam2):
        C = (x )
        return np.exp( -1 -lam2 * C**2)/np.exp( -1 -lam2 * C**2).sum()

    def fx_var(self, lam2, x, var):
        C = (x )
        return var/np.exp(-1) - np.dot(C**2 , np.exp(-lam2 * C**2 ) )/ np.exp(-1 -lam2 * C**2 ).sum()

    def dxfx_var(self, lam2, x):
        C = (x )
        return np.sum( C**4 * np.exp(-lam2 * C**2 )/np.exp( -1 -lam2 * C**2).sum() ) + np.sum( np.exp(-lam2 * C**2 )/np.exp( -1 -lam2 * C**2).sum())

    def forward(self, x, var, lam=0, max_iter = 100, tol = 1e-15):
        #Implements the Newton Raphson method:
        i = 0
        old_lam = lam

        while abs( self.fx_var(lam, x, var) ) > tol: #run the helper function and check

          lam = old_lam - self.fx_var(lam, x, var)/self.dxfx_var(lam, x)  # Newton-Raphson equation
          #print("Iteration" + str(i) + ": x = " + str(lam) + ", f(x) = " +  str( self.fx_var(lam, x, var) ) )  
          
          old_lam = lam
          i += 1
        
          if i > max_iter:
            break

        self.lam = lam

class MultipleMaxEnt_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(MultipleMaxEnt_Model, self).__init__()
        #self.lam = lam

    def predict_distribution(self, x,  mu, lam1, lam2):
        C = (x-mu)
        return np.exp( -1 -lam1 * x -lam2 * C**2)/np.exp( -1 -lam1 * x -lam2 * C**2).sum()

    def fx_1(self, lam1, lam2, x, mu):
        C = (x-mu)
        return mu/np.exp(-1) - np.sum( x * np.exp(-lam1 * x - lam2 * C**2) ) /np.exp( -1 -lam1 * x - lam2 * C**2).sum() 

    def fx_2(self, lam1, lam2, x, mu, var):
        C = (x-mu)
        return var/np.exp(-1) - np.sum( C**2 * np.exp(-lam1 * x -lam2 * C**2 ) ) / np.exp( -1 -lam1 * x -lam2 * C**2 ).sum()

    def dxfx1_lam1(self, lam1, lam2, x, mu):
        C = (x-mu)
        return np.sum( x**2 * np.exp(-lam1 * x - lam2 * C**2)/np.exp(-1 -lam1 * x -lam2 * C**2 ).sum() ) + np.sum( np.exp(-lam1 * x -lam2 * C**2 )/np.exp( -1 -lam1 * x -lam2 * C**2).sum())

    def dxfx1_lam2(self, lam1, lam2, x, mu):
        C = (x-mu)
        return np.sum( x*C**2 * np.exp(-lam1 * x - lam2 * C**2)/np.exp(-1 -lam1 * x -lam2 * C**2).sum() ) + np.sum( np.exp(-lam1 * x -lam2 * C**2 )/np.exp(-1 -lam1 * x -lam2 * C**2).sum())

    def dxfx2_lam2(self, lam1, lam2, x, mu):
        C = (x-mu)
        return np.sum( C**4 * np.exp(-lam1 * x - lam2 * C**2))/np.exp(-1 -lam1 * x -lam2 * C**2).sum()  + np.sum( np.exp(-lam1 * x -lam2 * C**2 )/np.exp(-1 -lam1 * x -lam2 * C**2).sum())

    def compute_fx(self, lam1, lam2, x, mu, var):
        b = np.empty([2, 1])
        #print(b, b.shape)
        b[0,0] = self.fx_1(lam1, lam2, x, mu)
        b[1,0] = self.fx_2(lam1, lam2, x, mu, var)
        
        return -b #this is negative, important!! easy to miss

    def compute_jacobian(self, lam1, lam2, x, mu):
        J = np.zeros([2,2])
        J[0,0] = self.dxfx1_lam1(lam1, lam2, x, mu,)
        J[0,1] = self.dxfx1_lam2(lam1, lam2, x, mu)
        J[1,0] = self.dxfx1_lam2(lam1, lam2, x, mu)
        J[1,1] = self.dxfx2_lam2(lam1, lam2, x, mu)
        return J

    def forward(self, x, mu, var, lam1=0, lam2=0, max_iter = 1000, tol = 1e-15):
        #Implements the Newton Raphson method:
        
        i = 0
        lam = np.array([lam1, lam2]).reshape(2,1)
        old_lam = lam

        while abs( self.compute_fx(lam1, lam2, x, mu, var).mean() ) > tol: #run the helper function and check
          #print(old_lam)
          lam1, lam2 = old_lam
          #Put initial guess
          b = self.compute_fx(lam1, lam2, x, mu, var)
          J = self.compute_jacobian(lam1, lam2, x, mu)

          #Jx = b -> x = J^-1b
          d = np.linalg.solve(J, b) # Newton-Raphson equation
          lam = d + old_lam
          #print("Iteration" + str(i) + ": x = " + str(lam) + ", f(x) = " +  str( self.compute_fx(lam1, lam2, x, mu, var) ) )
          #print(": x = " + str(lam) )
          #print(J)
          old_lam = lam
          i += 1

          if i > max_iter:
            break

        self.lam = lam

'''
mean_model = MeanMaxEnt_Model()
variance_model = VarianceMaxEnt_Model_1()
multiple_model = MultipleMaxEnt_Model()

prior = np.array([0.0938, 0.0967, 0.0908, 0.1123, 0.1045, 0.0938, 0.0996, 0.1016, 0.0977,0.1094])
x = np.array(range(len(prior)))

mean = 4.5
variance = 28.5

mean_model(x, mean, lam=0)
print(mean_model.lam)
mean_probs = mean_model.predict_distribution(x, mean_model.lam)

variance_model(x, variance, lam=0)
print(variance_model.lam)
var_probs = variance_model.predict_distribution(x, variance_model.lam)

mult_variance = 8.25
multiple_model(x, mean, mult_variance, lam1=0, lam2=0)
print(multiple_model.lam[0], multiple_model.lam[1])
mult_probs = multiple_model.predict_distribution(x, mean, multiple_model.lam[0], multiple_model.lam[1])

plt.figure(1)
plt.figure(figsize=(10, 8)) 
plt.bar(x, mean_probs, align='center', color='blue', edgecolor='black', linewidth=2, alpha=1)
plt.ylim(0,1)
plt.xticks(np.arange(0, 10, step=1))
#plt.title("Mean Constraint")
plt.ylabel('Probability')
#plt.xlabel('Classes')
#plt.legend(["μ =" + str(mean)])
plt.text(0, 0.85, s="μ=" + str(mean))

#plt.subplot(3,3,2)
plt.figure(2)
plt.figure(figsize=(10, 8)) 
plt.bar(x, var_probs, align='center', color='red', edgecolor='black', linewidth=2, alpha=1)
plt.ylim(0,1)
plt.xticks(np.arange(0, 10, step=1))
#plt.title("Variance Constraint")
plt.ylabel('Probability')
#plt.xlabel('Classes')
#plt.legend(["σ^2 ="+ str(variance)])
plt.text(0, 0.85, s="σ^2="+ str(variance))

#plt.subplot(3,3,3)
plt.figure(3)
plt.figure(figsize=(10, 8)) 
plt.bar(x, mult_probs, align="center", color='magenta', edgecolor='black', linewidth=2, alpha=1)
plt.ylim(0,1)
plt.xticks(np.arange(0, 10, step=1))
#plt.title("Multiple Constraints")
plt.ylabel('Probability')
plt.xlabel('Classes')
#plt.legend(["μ =" + str(mean) + ", σ^2 ="+ str(variance)])
plt.text(0, 0.85, s="μ=" + str(mean) + ", σ^2="+ str(variance))
#print(sum(var_probs * (x - mean)**2))


plt.figure(4, figsize=(10*4, 8*4))
plt.subplot(4,4,1)
plt.bar(x, mean_probs, align='center', color='blue', edgecolor='black', linewidth=2, alpha=1)
plt.ylim(0,1)
plt.xticks(np.arange(0, 10, step=1))
plt.ylabel('Probability')

plt.subplot(4,4,2)
plt.bar(x, mean_probs, align='center', color='blue', edgecolor='black', linewidth=2, alpha=1)
plt.ylim(0,1)
plt.xticks(np.arange(0, 10, step=1))

plt.subplot(4,4,3)
plt.bar(x, mean_probs, align='center', color='blue', edgecolor='black', linewidth=2, alpha=1)
plt.ylim(0,1)
plt.xticks(np.arange(0, 10, step=1))

plt.subplot(4,4,4)
plt.bar(x, mean_probs, align='center', color='blue', edgecolor='black', linewidth=2, alpha=1)
plt.ylim(0,1)
plt.xticks(np.arange(0, 10, step=1))
#plt.title("Mean Constraint")
#plt.xlabel('Classes')
#plt.legend(["μ =" + str(mean)])
plt.text(0, 0.85, s="μ=" + str(mean))
plt.title("Global Gibbs Distributions")

'''