import numpy as np
import matplotlib.pyplot as plt
import timeit
import pandas as pd
import seaborn as sns
    
from task_2_monte_carlo_functions import *
from random_seed import rng


# The plots take a while to run, setting booleans to ensure only the necessary plots are run.
plot_convergence_in_n = False
delta_sigma_plot = True
compare_methods_variance = False 
halton_vs_standard = False

# Value used for MSE, high N, good delta sigma, use central differences for minimal bias.
del_sigma_true =  diffsigma(1e-12, rng.standard_normal((int(1e5), K))) 
print(f'derivative value, N=100,000 paths, single finite difference: {del_sigma_true:.6f}')

mean_halton,std_halton,_ = diffsigma_monte_carlo_halton(1e-12, int(1e5),int(1e2),central=True)

print(f'Mean and standard deviation average for derivative, 100,000 paths averaged 100 times: {mean_halton:.6f} +/- {std_halton:.6f}')

if halton_vs_standard:
    N_values = np.logspace(2, 5, 10, dtype=int) 
    std_standard_array = []
    std_halton_array = []

    for N in N_values:
        _, std_standard, _ = diffsigma_monte_carlo(1e-12, N, int(1e2))  # Standard MC
        _, std_halton, _ = diffsigma_monte_carlo_halton(1e-12, N, int(1e2), central=True)  # Halton with central differences

        std_standard_array.append(std_standard)
        std_halton_array.append(std_halton)

    std_standard_array = np.array(std_standard_array) ** 2  # Convert to variance
    std_halton_array = np.array(std_halton_array) ** 2  # Convert to variance

    logspace = np.logspace(np.log10(np.min(N_values)), np.log10(np.max(N_values)), 1000)
    scaling_standard = std_standard_array[0] * N_values[0]
    scaling_halton = std_halton_array[0] * (N_values[0]**2)

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, std_standard_array, 'o-', label="Standard MC")
    plt.plot(N_values, std_halton_array, 'd-', label="Halton + Central Difference")
    plt.plot(logspace, scaling_standard / logspace, ls='--', color='r', label=r'O($N^{-1}$)')
    # Formatting
    plt.xscale("log")
    plt.xlabel(r'Number of Paths ($N$)')
    plt.ylabel('Variance')
    plt.title(r"Variance Convergence: Halton + Central Differences vs Standard MC")
    plt.legend()
    plt.grid(True, linestyle=":")

    # Save and show
    plt.savefig('Halton_vs_Standard_Variance.png')
    plt.show()


if plot_convergence_in_n:
    N_values = np.logspace(2,6,5)
    results = [diffsigma_monte_carlo(1e-12, int(N),int(1e2)) for N in N_values]
    means, stds, del_sigmas = zip(*results)  # Unpacking
    n_repeated = np.repeat(N_values, int(1e2))  # Each N has 1e2 samples

    means = np.array(means)
    stds = np.array(stds)

    # Scatter plot
    plt.figure(figsize=(10,8))
    plt.scatter(n_repeated, del_sigmas, alpha=0.5, s=10, label='Individual Samples')

    # Mean line
    plt.plot(N_values, means, color='r', linestyle='-',linewidth=0.7, label=r'Mean')
    plt.plot(N_values, means+stds, color='r', linestyle='--',linewidth=0.7, label=r'95% confidence')
    plt.plot(N_values, means-stds, color='r',linewidth=0.7, linestyle='--')

    # Log scale for better visualization
    plt.xscale('log')
    plt.xlabel(r'Number of paths $N$')
    plt.ylabel(r'$\partial V / \partial\sigma$ ')
    plt.title(r'$\partial V/\partial\sigma$ dependence on path number')
    plt.legend()
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.savefig('Convergence Vega Path Dependant')
    plt.show()

if delta_sigma_plot:
    dense_low = np.logspace(-17, -15, 10)
    mid_range = np.logspace(-15, -3, 15)
    dense_high = np.logspace(-3, 0, 10)

    delta_sigma_unique = np.unique(np.concatenate([dense_low, mid_range, dense_high]))
    delta_sigma_vals = np.tile(delta_sigma_unique, 5)  # Repeat for 5 trials

    del_sigma_central = np.array([
        [diffsigma_central(ds, rng.standard_normal((int(1e5), K))) for ds in delta_sigma_unique]
        for _ in range(5)
    ])
    del_sigma_forward = np.array([
        [diffsigma(ds, rng.standard_normal((int(1e5), K))) for ds in delta_sigma_unique]
        for _ in range(5)
    ])

    # Compute statistics
    central_means = np.mean(del_sigma_central, axis=0)
    forward_means = np.mean(del_sigma_forward, axis=0)
    central_vars = np.var(del_sigma_central, axis=0)
    forward_vars = np.var(del_sigma_forward, axis=0)

    # Plot with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(delta_sigma_unique, central_means, yerr=np.sqrt(central_vars), fmt='s-', label="Central Difference",alpha=0.5)
    plt.errorbar(delta_sigma_unique, forward_means, yerr=np.sqrt(forward_vars), fmt='o-', label="Forward Difference",alpha=0.5)
    plt.xscale("log")
    plt.xlabel(r'$\Delta \sigma$')
    plt.ylabel(r'$\partial V / \partial\sigma$ Estimate')
    plt.title(r"$\partial V/\partial\sigma$ Approximation dependence on $\Delta \sigma$ (Mean ± Std)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig('Vega_Approximation_vs_DeltaSigma_MeanStd.png')
    plt.show()


if compare_methods_variance:
    methods = ["Standard", "Moment Matching", "Antithetic", "Halton"]
    N_values = np.logspace(2,4,10,dtype=int)
    std_standard_array = []
    std_moment_match_array = []
    std_antithetic_array = []
    std_halton_array = []
    for method in methods:
        for N in N_values:
            if method == "Standard":
                std_standard_array.append(diffsigma_monte_carlo(1e-12, N,int(1e2))[1])
            elif method == "Moment Matching":
                std_moment_match_array.append(diffsigma_monte_carlo_moment_match(1e-12, N,int(1e2))[1])
            elif method == "Antithetic":
                std_antithetic_array.append(diffsigma_monte_carlo_antithetic(1e-12, N,int(1e2))[1])
            elif method == "Halton":
                std_halton_array.append(diffsigma_monte_carlo_halton(1e-12, N,int(1e2))[1])
    # Plot the variances
    logspace = np.logspace(np.log10(np.min(N_values)),np.log10(np.max(N_values)),1000)
    scaling_standard = np.mean([(np.array(std_standard_array)**2)[0],(np.array(std_moment_match_array)**2)[0],(np.array(std_antithetic_array)**2)[0]]) * N_values[0]
    scaling_halton = (np.array(std_halton_array)**2* N_values)[0]
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, np.array(std_standard_array)**2, 'o-', label="Standard")
    plt.plot(logspace,scaling_standard/logspace,ls='--',color='r',label=r'O($N^{-1}$)')
    plt.plot(logspace,scaling_halton/(logspace**2),ls='--',color='r',label=r'O($N^{-2}$)')
    plt.plot(N_values, np.array(std_moment_match_array)**2, 's-', label="Moment Matching")
    plt.plot(N_values, np.array(std_antithetic_array)**2, '^-', label="Antithetic")
    plt.plot(N_values, np.array(std_halton_array)**2, 'd-', label="Halton")
    plt.xscale("log")
    plt.xlabel(r'Number of Paths ($N$)')
    plt.ylabel('Variance')
    plt.title(r"Variance of Monte Carlo methods for $\partial V /\partial\sigma$")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig('Path Dependence Variance.png')
    plt.show()

