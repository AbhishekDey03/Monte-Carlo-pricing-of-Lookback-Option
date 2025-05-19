import numpy as np
from random_seed import rng

def create_values_to_plot(monte_carlo_function,repeat=True,num_repeats=5,num_values = 15):
    rng = np.random.default_rng(8032003) # Reset the seed
    N_unique = np.logspace(3, 7, num_values, dtype=int)  # 15 unique log-spaced values

    # Create N_values with repeats
    if repeat:
        N_values = np.repeat(N_unique, num_repeats)
    else:
        N_values = N_unique

    # Run Monte Carlo simulations
    MC_values = np.array([monte_carlo_function(N) for N in N_values])
    return N_values,MC_values

