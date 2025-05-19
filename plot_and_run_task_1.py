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
from plotting_task_1 import create_values_to_plot
from task_1_monte_carlo_functions import *

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


N_values,MC_values = create_values_to_plot(run_monte_carlo)

plt.scatter(N_values, MC_values,label="Monte Carlo")
plt.xlabel('Number of simulations')
plt.axhline(analytical_integral(f(S_0,T),v(S_0,T),T), color='red',label='Analytical integral')
plt.xscale('log')
plt.legend()
plt.title('Monte Carlo vs analytical integration')
plt.ylabel('Option price')
plt.show()