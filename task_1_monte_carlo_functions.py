import numpy as np
from numpy import exp,tanh
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import timeit
import pandas as pd
import scipy
from scipy.stats.qmc import Halton
from scipy.stats import norm
from random_seed import rng

S_0 = 601.14 # Initial stock price
r = 0.03 # Interest ratee
T=1 # Maturity
# Strike prices
X_1 = 600
X_2 = 650
# Fitted parameters
theta = 600.954
alpha = 0.01
beta = 0.02
gamma = 1.1
# Option volatility
sigma = 0.18
save_method = True
# Sampler is global such that it is not reset
sampler = Halton(d=1,scramble=True,seed = 8032003) # 1D Halton sequence

# The payoff function
g = lambda S_T: np.where(
    S_T < X_1, X_2, # If S_T < X_1 then the payoff is X_2
    np.where(S_T >= X_2, S_T, # If S_T >= X_2 then the payoff is S_T
                S_T - X_1) # Otherwise the payoff is S_T - X_1
)

# Monte carlo functions
f = lambda S_0,T: S_0* (exp(alpha*T)-exp(beta*T)) + theta*(1+tanh(alpha*T)-(beta*T))
v = lambda S_0,T: sigma * exp(beta*T)*S_0**2 * theta **(-gamma)


def Si_T(S_0,T,phi):
    return (f(S_0,T)+v(S_0,T)*phi*np.sqrt(T))

def run_monte_carlo(N):
    return 1/N * np.sum(g(Si_T(S_0,T,rng.standard_normal(N)))) * exp(-r*T)


integrand = lambda z,f,v,T: g(z)* exp(-(z-f)**2/(2* v**2 *T))

def analytical_integral(f,v,T):
    # Integrate the integrand from -inf to inf via quadrature
    return exp(-r*T)/np.sqrt(2 * np.pi * v**2 * T) * integrate.quad(integrand, -np.inf, np.inf, args=(f,v,T))[0]

def run_monte_carlo_halton(N):
    halton_samples = sampler.random(N).flatten()
    halton_samples = norm.ppf(halton_samples) #Find random normal values
    S_T = Si_T(S_0,T,halton_samples)
    return 1/N * np.sum(g(S_T)) * exp(-r*T)

def run_monte_carlo_moment_matching(N):
    phi = rng.standard_normal(N//2)
    phi = np.append(phi, -phi)
    nu_sq = np.var(phi) # Var(phi)
    phi = phi / np.sqrt(nu_sq)
    return 1/N * np.sum(g(Si_T(S_0,T,phi))) * exp(-r*T)

def run_monte_carlo_antithetic(N):
    phi = rng.standard_normal(N//2)
    phi = np.append(phi,-phi)
    return 1/N * np.sum(g(Si_T(S_0,T,phi))) * exp(-r*T)


def run_monte_carlo_antithetic_with_var(N):
    phi = rng.standard_normal(N//2)
    phi = np.append(phi,-phi)
    return 1/N * np.sum(g(Si_T(S_0,T,phi))) * exp(-r*T), np.var(g(Si_T(S_0,T,phi)) * exp(-r*T),ddof=1)

def run_monte_carlo_moment_matching_with_var(N):
    phi = rng.standard_normal(N//2)
    phi = np.append(phi, -phi)
    nu_sq = np.var(phi) # Var(phi)
    phi = phi / np.sqrt(nu_sq)
    return 1/N * np.sum(g(Si_T(S_0,T,phi))) * exp(-r*T), np.var(g(Si_T(S_0,T,phi)) * exp(-r*T),ddof=1)

def run_monte_carlo_halton_with_var(N):
    halton_samples = sampler.random(N).flatten()
    halton_samples = norm.ppf(halton_samples) #Find random normal values
    S_T = Si_T(S_0,T,halton_samples)
    return 1/N * np.sum(g(S_T)) * exp(-r*T), np.var(g(S_T) * exp(-r*T),ddof=1)

def run_monte_carlo_with_var(N):
    return 1/N * np.sum(g(Si_T(S_0,T,rng.standard_normal(N)))) * exp(-r*T), np.var(g(Si_T(S_0,T,rng.standard_normal(N))) * exp(-r*T),ddof=1)
