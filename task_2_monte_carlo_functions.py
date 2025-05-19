import numpy as np
from numpy import exp,tanh
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import timeit
import pandas as pd
from IPython.display import display
import scipy
from random_seed import rng
from scipy.stats.qmc import Halton
from scipy.stats import norm


# Market parameters
r = 0.01
theta = 6998.53
alpha = 0.02
beta = 0.05
gamma = 0.99
# Volatility
sigma = 0.43
# Stock and option values
S_0 = 6986.49
T = 1.25
K = 75
X = 7000


# Write with t as a variable, even though it's not needed for enhanced readability.
f = lambda S,t: (alpha * theta - beta* S)
v = lambda S,t,sigma: (sigma * (np.abs(S))**gamma)

t_k = np.linspace(0,T,K+1)
delta_t = T/K

def Si(S_0,t_k,n,sigma,phi):
    # Initialise the size of the numpy array for each path in t_k
    S = np.zeros((n,K+1)) 

    S[:,0] = S_0 # Assert S(t=0) as it is given
    
    for k in range (1,K+1): # Again, start from 1 as t=0 asserted
        # Note the : is because in 1 for loop we are doing all n paths, iterating over only time
        S[:,k] = S[:,k-1] + f(S[:,k-1],t_k[k-1]) * delta_t + v(S[:,k-1],t_k[k-1],sigma)* np.sqrt(delta_t)* phi[:,k-1]
    return S

def g(Si):
    A = np.max(Si,axis = 1) # Max_k S(t_k)
    return np.maximum(A-X,0)

V = lambda Si,n : exp(-r*T) * (1/n) * np.sum(g(Si))


#-----------------------------
# Alternate Monte Carlo Methods
#-----------------------------
sampler = Halton(d=K, scramble=True,seed = 8032003)  

def halton_sequence(N,K):
    sampled = sampler.random(N)
    return  norm.ppf(sampled)

def moment_match(phi):
    phi = np.concatenate([phi, -phi], axis=0)  # Ensures shape (2N, K)
    nu_sq = np.var(phi) # Var(phi)
    return phi / np.sqrt(nu_sq)

def antithetic(phi):
    return np.concatenate([phi, -phi], axis=0)  # Ensures shape (2N, K)

def diffsigma(delta_sigma, phi,central=False):
    n = phi.shape[0]
    if central:
        V_plus = V(Si(S_0, t_k, n, sigma + delta_sigma, phi), n)
        V_minus = V(Si(S_0, t_k, n, sigma - delta_sigma, phi), n)
        return (V_plus - V_minus) / (2 * delta_sigma)
    else:
        V1 = V(Si(S_0, np.zeros_like(phi), n, (sigma + delta_sigma), phi), n)
        V2 = V(Si(S_0, np.zeros_like(phi), n, sigma, phi), n)
        return (V1 - V2) / delta_sigma

def diffsigma_monte_carlo(delta_sigma, N,M,central=False):
    del_sigma = np.zeros(M)
    for i in range(M):
        phi = rng.standard_normal((N, K))  
        del_sigma[i] = (diffsigma(delta_sigma, phi,central))
    a = 1/M * np.sum(del_sigma) 
    b_sq = 1/(M-1)*np.sum((a-del_sigma)**2) 
    b=np.sqrt(b_sq)
    return a,b,del_sigma

def diffsigma_monte_carlo_antithetic(delta_sigma, N,M,central=False):
    del_sigma = np.zeros(M)
    for i in range(M):
        phi = antithetic(rng.standard_normal((N//2, K)))
        del_sigma[i] = (diffsigma(delta_sigma, phi,central))
    a = 1/M * np.sum(del_sigma) 
    b_sq = 1/(M-1)*np.sum((a-del_sigma)**2) 
    b=np.sqrt(b_sq)
    return a,b,del_sigma

def diffsigma_monte_carlo_moment_match(delta_sigma, N,M,central=False):
    del_sigma = np.zeros(M)
    for i in range(M):
        phi = moment_match(rng.standard_normal((N//2, K)))
        del_sigma[i] = (diffsigma(delta_sigma, phi,central))
    a = 1/M * np.sum(del_sigma) 
    b_sq = 1/(M-1)*np.sum((a-del_sigma)**2) 
    b=np.sqrt(b_sq)
    return a,b,del_sigma

def diffsigma_monte_carlo_halton(delta_sigma, N,M,central=False):
    del_sigma = np.zeros(M)
    for i in range(M):
        phi = halton_sequence(N,K)
        del_sigma[i] = (diffsigma(delta_sigma, phi,central))
    a = 1/M * np.sum(del_sigma) 
    b_sq = 1/(M-1)*np.sum((a-del_sigma)**2) 
    b=np.sqrt(b_sq)
    return a,b,del_sigma
    
def diffsigma_central(delta_sigma, phi):
    n = phi.shape[0]
    V_plus = V(Si(S_0, t_k, n, sigma + delta_sigma, phi), n)
    V_minus = V(Si(S_0, t_k, n, sigma - delta_sigma, phi), n)
    return (V_plus - V_minus) / (2 * delta_sigma)