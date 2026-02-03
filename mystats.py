
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def normalized_autocorrelation(Y, max_lag=None, estimate_tau=False):
    """
    Computes ACF and optionally fits an exponential decay: f(t) = exp(-t/tau)
    """
    Y = np.asarray(Y, dtype=float)
    N = len(Y)
    if max_lag is None:
        max_lag = N // 2 # Standard practice: don't trust lags > N/2
        
    # --- 1. Compute ACF (FFT Method) ---
    Y_centered = Y - np.mean(Y)
    n_fft = 2**int(np.ceil(np.log2(2*N - 1)))
    psd = np.abs(np.fft.fft(Y_centered, n=n_fft))**2
    autocov = np.real(np.fft.ifft(psd))
    acf = autocov[:max_lag + 1] / autocov[0]
    
    if not estimate_tau:
        return acf

    # --- 2. Exponential Fitting ---
    # Define the model: f(t) = exp(-t / tau)
    def model_exp(t, tau):
        return np.exp(-t / tau)

    lags = np.arange(max_lag + 1)
    
    try:
        # We start the search at tau=10 as a heuristic
        popt, _ = curve_fit(model_exp, lags, acf, p0=[10.0])
        tau_opt = popt[0]
        fit_curve = model_exp(lags, tau_opt)
        return acf, fit_curve, tau_opt
    
    except Exception as e:
        print(f"Fit failed: {e}")
        return acf, None, None
#

# Normalized autocorrelation function
def normalized_autocorrelation_fft(Y, max_lag=None):
    """
    Optimized ACF for MD trajectories using FFT.
    Standard 'biased' statistical estimator.
    """
    Y = np.asarray(Y, dtype=float)
    N = len(Y)
    if max_lag is None:
        max_lag = N - 1
        
    # 1. Center the data
    Y_centered = Y - np.mean(Y)
    
    # 2. FFT-based convolution
    # We pad to the next power of 2 for optimal FFT performance
    n_fft = 2**int(np.ceil(np.log2(2*N - 1)))
    Y_fft = np.fft.fft(Y_centered, n=n_fft)
    
    # The Power Spectral Density is the FFT of the autocovariance
    psd = Y_fft * np.conj(Y_fft)
    autocov = np.real(np.fft.ifft(psd))
    
    # 3. Slice to max_lag and normalize
    # We divide by N (biased) and then by autocov[0] (which is the variance * N)
    acf = autocov[:max_lag + 1] / autocov[0]
    
    return acf
#

# Cumulative mean and standard deviation
def cum_scum(X):
    """
    Cumulative mean and standard deviation of the mean (SOM).
    Parameters
    ----------
    X : 1D np.ndarray

    Returns
    -------
    cum_mean : np.ndarray
        Cumulative mean
    som : np.ndarray
        Running standard deviation of the mean
    """
    X = np.asarray(X, dtype=float)
    n = np.arange(1, len(X) + 1)
    
    # Cumulative mean: E[X]
    cum_mean = np.cumsum(X) / n
    
    # Cumulative mean of squares: E[X^2]
    cum_mean_sq = np.cumsum(X**2) / n
    
    # Population variance: E[X^2] - (E[X])^2
    # We use clip(0) to avoid tiny negative numbers due to float precision
    cum_var = np.clip(cum_mean_sq - cum_mean**2, 0, None)
    
    # Standard Deviation (running)
    cum_std = np.sqrt(cum_var)
    
    # Standard Error of the Mean (SEM): std / sqrt(n)
    #cum_std  /= np.sqrt(n) # uncomment for standard error of the mean
    
    return cum_mean, cum_std
#

# Calculate ensemble mean and std across multiple trajectories
def ensemble_mean_and_std(traj_obs_list):
    """
    Ensemble means and std of the means
    :param listOfArrays: arrays of some observable
    """

    min_len = min(len(X) for X in traj_obs_list)
    truncated_matrix = np.array([X[:min_len] for X in traj_obs_list])
    
    ensemble_mean = np.mean(truncated_matrix, axis=0)
    ensemble_std = np.std(truncated_matrix, axis=0) # Spread between trajectories
    
    return ensemble_mean, ensemble_std
#

# Calculate ensemble histogram and std across multiple trajectories
def ensemble_histogram(traj_obs_list, density=True, bins=50, obs_range=None):
    """
    Calculates the ensemble average probability distribution and 
    the standard deviation across multiple trajectories.
    """
    # 1. Determine common range if not provided
    if obs_range is None:
        all_data = np.concatenate(traj_obs_list)
        obs_range = (np.min(all_data), np.max(all_data))
    
    hist_list = []
    
    # 2. Calculate histogram for each trajectory
    for traj_obs in traj_obs_list:
        # Use density=True to get a probability distribution (area = 1)
        counts, bin_edges = np.histogram(traj_obs, bins=bins, range=obs_range, density=density)
        hist_list.append(counts)
    
    # Convert to 2D array: (number_of_trajectories, number_of_bins)
    hist_matrix = np.array(hist_list)
    
    # 3. Calculate mean and std for each bin
    ensemble_mean = np.mean(hist_matrix, axis=0)
    ensemble_std = np.std(hist_matrix, axis=0)
    
    # Center of bins for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, ensemble_mean, ensemble_std
#

def ensemble_histogram_plus(traj_obs_list, density=True, bins=50, obs_range=None):
    """
    Calculates ensemble average distribution, standard deviation, 
    Shannon entropy, and the number of modes.
    """
    # 1. Determine common range if not provided
    if obs_range is None:
        all_data = np.concatenate(traj_obs_list)
        obs_range = (np.min(all_data), np.max(all_data))
    
    hist_list = []
    for traj_obs in traj_obs_list:
        counts, bin_edges = np.histogram(traj_obs, bins=bins, range=obs_range, density=density)
        hist_list.append(counts)
    
    hist_matrix = np.array(hist_list)
    ensemble_mean = np.mean(hist_matrix, axis=0)
    ensemble_std = np.std(hist_matrix, axis=0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- 4. Ensemble Entropy (Shannon Entropy) ---
    # Entropy calculation requires probabilities (P) that sum to 1.
    # If density=True, we must multiply by bin width to get probabilities.
    bin_width = bin_edges[1] - bin_edges[0]
    probs = ensemble_mean * bin_width if density else ensemble_mean / np.sum(ensemble_mean)
    
    # Avoid log(0) by masking or adding a tiny epsilon
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))

    # --- 5. Number of Modes ---
    # We use find_peaks on the ensemble mean. 
    # 'prominence' helps ignore tiny noise-related fluctuations.
    peaks, _ = find_peaks(ensemble_mean, prominence=np.max(ensemble_mean)*0.05)
    num_modes = len(peaks)
    
    return {
        "bin_centers": bin_centers,
        "mean": ensemble_mean,
        "std": ensemble_std,
        "entropy": entropy,
        "num_modes": num_modes,
        "peak_indices": peaks
    }


def calculate_pt_diagnostics(out_df):
    """
    Calculates HMC-PT efficiency using the dataframe index as the time-tracker.
    """
    # 1. Sort by replica, then by the DataFrame's actual index (time)
    # We use out_df.index instead of the string 'index'
    df_sorted = out_df.sort_index().sort_values(by='replicaIx', kind='mergesort')
    
    # 2. Identify Swaps
    # A swap occurs if the thermoIx of a replica changes from the previous timestep
    # We use groupby to ensure we don't compare the end of Replica 0 to the start of Replica 1
    df_sorted['swapped'] = df_sorted.groupby('replicaIx')['thermoIx'].shift(1) != df_sorted['thermoIx']
    
    # Remove the first row of each replica (which always shows as a 'swap' due to NaN shift)
    n_replicas = out_df['replicaIx'].nunique()
    total_swaps = df_sorted['swapped'].sum() - n_replicas
    
    # 3. Efficiency Calculations
    exchange_rate = total_swaps / (len(out_df) - n_replicas)
    hmc_acc = out_df['acc'].mean()
    
    # 4. Mixing Matrix
    mixing_matrix = pd.crosstab(
        out_df['replicaIx'], 
        out_df['thermoIx'], 
        normalize='index'
    )
    
    return {
        "hmc_acceptance": hmc_acc,
        "replica_exchange_rate": exchange_rate,
        "mixing_matrix": mixing_matrix,
        "total_swaps": total_swaps
    }
#