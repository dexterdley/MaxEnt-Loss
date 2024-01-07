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
          lam1, lam2 = old_lam
          #Put initial guess
          b = self.compute_fx(lam1, lam2, x, mu, var)
          J = self.compute_jacobian(lam1, lam2, x, mu)

          #Jx = b -> x = J^-1b
          d = np.linalg.solve(J, b) # Newton-Raphson equation
          lam = d + old_lam

          old_lam = lam
          i += 1

          if i > max_iter:
            break

        self.lam = lam
