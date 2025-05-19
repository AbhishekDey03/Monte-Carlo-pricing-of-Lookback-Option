import numpy as np
import matplotlib.pyplot as plt
import timeit
import pandas as pd
from scipy.stats.qmc import Halton
from random_seed import rng

from plotting_task_1 import create_values_to_plot
from task_1_monte_carlo_functions import *
# True to record time
time = True

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

N_values, vals_basic_mc= create_values_to_plot(run_monte_carlo,True,num_values = 12,num_repeats = 10)
vals_moment_matching = create_values_to_plot(run_monte_carlo_moment_matching,True,num_values = 12,num_repeats = 10)[1]
vals_antithetic = create_values_to_plot(run_monte_carlo_antithetic,True,num_values = 12,num_repeats = 10)[1]
vals_halton = create_values_to_plot(run_monte_carlo_halton,True,num_values = 12,num_repeats = 10)[1]

unique_N_values = np.unique(N_values)
variance_basic = np.array([np.var(vals_basic_mc[N_values == N]) for N in unique_N_values])
variance_moment_matching = np.array([np.var(vals_moment_matching[N_values == N]) for N in unique_N_values])
variance_antithetic = np.array([np.var(vals_antithetic[N_values == N]) for N in unique_N_values])
variance_halton = np.array([np.var(vals_halton[N_values == N]) for N in unique_N_values])

mean_basic = np.array([np.mean(vals_basic_mc[N_values == N]) for N in unique_N_values])
mean_moment_matching = np.array([np.mean(vals_moment_matching[N_values == N]) for N in unique_N_values])
mean_antithetic = np.array([np.mean(vals_antithetic[N_values == N]) for N in unique_N_values])
mean_halton = np.array([np.mean(vals_halton[N_values == N]) for N in unique_N_values])

plt.figure(figsize=(10, 6))
plt.plot(unique_N_values,mean_basic,'o-', label="Standard")
plt.plot(unique_N_values,mean_moment_matching,'s-', label="Moment Matching")
plt.plot(unique_N_values,mean_antithetic,'^-', label="Antithetic")
plt.plot(unique_N_values,mean_halton,'d-', label="Halton")
plt.xlabel('Number of simulations')
plt.axhline(analytical_integral(f(S_0,T),v(S_0,T),T),color = 'r',linestyle='-',linewidth=1,label = r'Analytic solution (quadrature)')
plt.xscale('log')
plt.legend()
plt.xlabel(r'Number of paths $N$')
plt.ylabel(r'Option Value $V$')
plt.title(r'Monte Carlo methods for Simple European call')
plt.grid(True, linestyle=":",linewidth=0.5)
plt.legend()
plt.savefig('variance reduction techniques convergence to analytic solution.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(unique_N_values,variance_basic,'o-', label="Standard")
plt.plot(unique_N_values,variance_moment_matching,'s-', label="Moment Matching")
plt.plot(unique_N_values,variance_antithetic,'^-', label="Antithetic")
plt.plot(unique_N_values,variance_halton,'d-', label="Halton")


# Scaled plots of 1/n and 1/(n^2)
N_values_logspace = np.logspace(np.log10(np.min(unique_N_values)),np.log10(np.max(unique_N_values)),1000)
mean_init_var = np.mean([variance_basic[0], variance_antithetic[0], variance_moment_matching[0]]) # All of these should scale 1/N
c1 = mean_init_var * unique_N_values[0]
c2 = variance_halton[0] * unique_N_values[0]**2

plt.plot(N_values_logspace, c1 / (N_values_logspace), '--', color='blue', label=r'$O(N^{-1})$')
plt.plot(N_values_logspace, c2 / (N_values_logspace**2), '--', color='red', label=r'$O(N^{-2})$')

plt.xscale('log')
plt.legend()
plt.xlabel(r'Number of paths $N$')
plt.ylabel(r'Variance')
plt.title(r'Variance of Monte Carlo methods for Simple European call')
plt.legend()
plt.grid(True, linestyle=":",linewidth=0.5)
plt.savefig('variance reduction techniques.png')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(unique_N_values,variance_halton,'d-', label="Halton",color = 'r')
# Scaled plots of 1/n and 1/(n^2)
N_values_logspace = np.logspace(np.log10(np.min(unique_N_values)),np.log10(np.max(unique_N_values)),1000)
mean_init_var = np.mean([variance_basic[0], variance_antithetic[0], variance_moment_matching[0]]) # All of these should scale 1/N
c2 = variance_halton[0] * unique_N_values[0]**2
plt.plot(N_values_logspace, c2 / (N_values_logspace**2), '--', color='red', label=r'$O(N^{-2})$')
plt.xscale('log')
plt.legend()
plt.xlabel(r'Number of paths $N$')
plt.ylabel(r'Variance')
plt.title(r'Variance of Monte Carlo methods for Simple European call')
plt.legend()
plt.grid(True, linestyle=":",linewidth=0.5)
plt.savefig('variance reduction Halton.png')
plt.show()

# Function to automate timing for a given method and N
def time_mc_method(mc_function, N, repetitions=10):
    avg_time = timeit.timeit(lambda: mc_function(N), number=repetitions) / repetitions
    return avg_time

timing_results = { "N": [] }

# List of MC methods to compare
mc_methods = {
    "Standard MC": run_monte_carlo,
    "Antithetic MC": run_monte_carlo_antithetic,
    "Moment Matching MC": run_monte_carlo_moment_matching,
    "Halton Sequence MC": run_monte_carlo_halton
}
if time:
    for N in unique_N_values:
        timing_results["N"].append(N)
        for method_name, mc_function in mc_methods.items():
            if method_name not in timing_results:
                timing_results[method_name] = []
            result = time_mc_method(mc_function, N)
            timing_results[method_name].append(float(f"{result:.6f}")) # Round to 4sf

    df_timing = pd.DataFrame(timing_results)

    # Compute Mean Squared Error (MSE) var + bias
    mse_basic = variance_basic + (mean_basic - analytical_integral(f(S_0,T),v(S_0,T),T))**2
    mse_antithetic = variance_antithetic + (mean_antithetic - analytical_integral(f(S_0,T),v(S_0,T),T))**2
    mse_moment_matching = variance_moment_matching + (mean_moment_matching - analytical_integral(f(S_0,T),v(S_0,T),T))**2
    mse_halton = variance_halton + (mean_halton - analytical_integral(f(S_0,T),v(S_0,T),T))**2

    # Compute efficiency: MSE/time
    efficiency_basic = mse_basic / np.array(df_timing["Standard MC"])
    efficiency_antithetic = mse_antithetic / np.array(df_timing["Antithetic MC"])
    efficiency_moment_matching = mse_moment_matching / np.array(df_timing["Moment Matching MC"])
    efficiency_halton = mse_halton / np.array(df_timing["Halton Sequence MC"])

    df_efficiency = pd.DataFrame({
        "N": unique_N_values,
        "Standard MC": efficiency_basic,
        "Antithetic MC": efficiency_antithetic,
        "Moment Matching MC": efficiency_moment_matching,
        "Halton MC": efficiency_halton
    })

    # Save to CSV
    df_efficiency.to_csv('efficiency_comparison.csv', index=False,float_format="%.6e")

