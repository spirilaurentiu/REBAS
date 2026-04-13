
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

class LS_Statistics:
    """ Class for general statistics """
    def __init__(self):
        pass
    #

    # Cumulative mean and standard deviation
    def cum_scum(self, X):
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
    def ensemble_mean_and_std(self, traj_obs_list):
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
    def ensemble_histogram(self, traj_obs_list, density=True, bins=50, obs_range=None):
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

    def ensemble_histogram_plus(self, traj_obs_list, density=True, bins=50, obs_range=None):
        """
        Calculates ensemble average distribution, standard deviation, 
        Shannon entropy, and the number of modes.
        """
        # 1. Determine common range if not provided
        if obs_range is None:
            all_data = np.concatenate(traj_obs_list)
            obs_range = (np.min(all_data), np.max(all_data))
        
        hist_list = []
        bin_edges = None
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

    # Exchange matrix and HMC acceptance rate
    def exchange_matrix(self, out_df):
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

    # -----------------------------------------------------------------------------
    #                            Probability Distributions
    #region Probability Distributions ---------------------------------------------

    # Kolmogorov-Smirnov test
    def Kolmogorov_Smirnov_Test(self, sample1, sample2):
        """
        Performs the Kolmogorov-Smirnov test to compare two samples.
        Returns the KS statistic and p-value.
        """
        from scipy.stats import ks_2samp
        ks_statistic, p_value = ks_2samp(sample1, sample2, nan_policy='omit')
        return ks_statistic, p_value
    #

    #endregion # probability distributions ----------------------------------------

    # -----------------------------------------------------------------------------
    #                                 Autocorrelation
    #region Autocorrelation -------------------------------------------------------

    # Helper for autocorrelation functions
    def get_num_lags(self, N, lag_fraction, max_lag):
        """ Helper to determine the actual number of lags to compute.
            Arguments:
                N   :   int
                lag_fraction : float
                max_lag : int
            Returns:
                int : number of lags to compute
        """
        limit = int(N * lag_fraction)
        lag_limit = min(limit, max_lag)
        return lag_limit
    #

    # Calculates Integrated Autocorrelation Time (Sokal's method)
    def getTau(self, ACF_rho, window_factor = 5):
        """ Calculates Integrated Autocorrelation Time (Sokal's method).
        Sum until the window is ~5x the current estimate of tau.
            Arguments:
                ACF_rho : array-like
                window_factor : int : safety floor to prevent stopping too early (default=5)
            Returns:
                float : estimated integrated autocorrelation time
        """
        # 1 + 2 * sum(rho)
        # Using a running sum to find the self-consistent window
        tau_est = 1.0
        for aIx in range(1, len(ACF_rho)):
            tau_est += 2 * ACF_rho[aIx]

            #print(ACF_rho[aIx], tau_est, window_factor, window_factor * tau_est) # Debug: print the ACF values used in the sum

            if ACF_rho[aIx] <= 0.1:
                break

            if ACF_rho[aIx] < 0.2:
                if aIx > (window_factor * tau_est):
                    break

            # Warning if we hit the end of the array without 'breaking'
            if aIx == len(ACF_rho) - 1:
                print("Warning: Tau estimation did not converge within the provided lags.")

        return tau_est
    #

    # Autocorrelation manual loop
    def autocorr2_revised(self, data, lag_fraction=0.1, max_lag=5000):
        """ Manual: Loop-based (Slow for large max_lag)
        """
        N = len(data)
        xp = data - np.mean(data)
        var = np.var(data)
        max_lag = self.get_num_lags(N, lag_fraction, max_lag)
        
        # Calculate ACF up to num_lags
        ACF_rho = np.array([np.sum(xp[l:] * xp[:N-l]) / (N * var) for l in range(max_lag)])
        
        tau = self.getTau(ACF_rho)
        ess = N / tau

        return (ACF_rho, tau, ess)
    #

    # Autocorrelation using FFT (Wiener-Khinchin Theorem)
    def autocorr3_revised(self, data, lag_fraction=0.5, max_lag=5000):
        """FFT: Padded (Linear Correlation) - Best for max_lag=5000"""
        N = len(data)
        xp = data - np.mean(data)
        var = np.var(data)
        max_lag = self.get_num_lags(N, lag_fraction, max_lag)
        
        # Pad to power of 2 for FFT speed and to avoid circular wrap-around
        fsize = 2**np.ceil(np.log2(2*N-1)).astype(int)
        cf = np.fft.fft(xp, fsize)
        sf = cf.conjugate() * cf
        
        # Inverse FFT to get the correlation
        res = np.fft.ifft(sf).real
        # Normalize and slice to the requested lags
        full_corr = (res[:N] / N) / var
        ACF_rho = full_corr[:max_lag]
        
        tau = self.getTau(ACF_rho)
        ess = N / tau
        
        return (ACF_rho, tau, ess)
    #

    # Autocorrelation using np.correlate
    def autocorr5_revised(self, data, lag_fraction=0.5, max_lag=5000):
        """Numpy Correlate: Optimized C-loop"""
        N = len(data)
        xp = data - np.mean(data)
        var = np.var(data)
        num_lags = self.get_num_lags(N, lag_fraction, max_lag)
        
        # np.correlate provides the full linear correlation
        raw_corr = np.correlate(xp, xp, mode='full')[N-1:]
        corr = (raw_corr[:num_lags] / N) / var
        
        return corr, self.getTau(corr)
    #

    # Autocorrelation with FFT and exponential fitting 
    def normalized_autocorrelation(self, Y, max_lag=None, estimate_tau=False):
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

    #endregion # autocorrelation --------------------------------------------------