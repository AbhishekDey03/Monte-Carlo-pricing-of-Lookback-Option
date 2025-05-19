"""
contains code for tasks 1.1 & 1.2
"""

import numpy as np
from numpy import exp,tanh
import matplotlib.pyplot as plt

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

print(f'Monte Carlo for N = 100,000 samples (6dp): {run_monte_carlo(int(1e5)):.6f}')
print(f'Analytic Integral (6dp): {analytical_integral(f(S_0,T),v(S_0,T),T):.6f}')
n_repeated = np.unique(N_values)
means = np.array([np.mean(MC_values[N==N_values]) for N in n_repeated])
stds = np.array([np.std(MC_values[N==N_values]) for N in n_repeated])

# Scatter plot
plt.figure(figsize=(10,8))
plt.scatter(N_values, MC_values, s=10, label='Individual Samples')

# Analytic line
plt.axhline(analytical_integral(f(S_0,T),v(S_0,T),T),color = 'r',linestyle='-',linewidth=1,label = r'Analytic solution (quadrature)')

# Log scale for better visualization
plt.xscale('log')
plt.xlabel(r'Number of paths $N$')
plt.ylabel(r'Option Value $V$')
plt.title(r'Monte Carlo approximation of Simple European call')
plt.legend()
plt.grid(True, which="both", linestyle=":", linewidth=0.5)
plt.savefig('convergence to analytic solution.png')
plt.show()