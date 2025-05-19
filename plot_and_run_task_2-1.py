import numpy as np
from numpy import exp,tanh
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import timeit
import pandas as pd
from IPython.display import display
import scipy

from task_2_monte_carlo_functions import *
from random_seed import rng



S = Si(S_0,t_k,int(1e5),sigma,rng.standard_normal((int(1e5),K)))

V_100k = V(S,int(1e5))

print(f'{V_100k:.6f}')


N_list = np.logspace(1, 6, 6, dtype=int)
N_list = np.repeat(N_list, 10) # Repeat each 10 times to show convergencee
V_list = np.array([V(Si(S_0,t_k,n,sigma,rng.standard_normal((n,K))),n) for n in N_list]) # Find values
N_unique = np.unique(N_list)
logspace = np.logspace(np.log10(np.min(N_unique)),np.log10(np.max(N_unique)),1000)
variances = np.array([np.var(V_list[N == N_list ]) for N in N_unique])
means = np.array([np.mean(V_list[N == N_list ]) for N in N_unique])

plt.figure(figsize=(10, 6))
plt.scatter(N_list,V_list,s=10, label='Individual Samples')
plt.axhline(V_100k,color='r', linestyle='-',linewidth=1,label = r'Value at $N=100,000$ paths')
plt.xscale('log')
plt.xlabel(r'Number of paths $N$')
plt.ylabel(r'Option Value $V$')
plt.title(r'Monte Carlo approximation of Fixed Strike Lookback Call Option')
plt.legend()
plt.grid(True, which="both", linestyle=":", linewidth=0.5)
plt.savefig('path dependant call option convergence.png')
plt.show()

N_times_variance =(N_unique * variances)[0]

plt.figure(figsize=(10, 6))
plt.plot(N_unique,variances,'o-',label='Variance')
plt.plot(logspace,N_times_variance/logspace,ls='--',label=r'O($N^{-1}$)')
plt.xscale('log')
plt.xlabel(r'Number of paths $N$')
plt.ylabel(r'Option Value $V$')
plt.title(r'Monte Carlo approximation of Fixed Strike Lookback Call Option')
plt.legend()
plt.grid(True, which="both", linestyle=":", linewidth=0.5)
plt.savefig('path dep variance.png')
plt.show()