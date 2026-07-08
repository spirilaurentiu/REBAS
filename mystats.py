
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
        """ Cumulative mean and standard deviation of the mean (SOM).
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

    # Ensemble histogram with additional statistics: Shannon entropy and number of modes
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
    #

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
        for lIx in range(1, len(ACF_rho)):
            tau_est += 2 * ACF_rho[lIx]

            #print(ACF_rho[lIx], tau_est, window_factor, window_factor * tau_est) # Debug: print the ACF values used in the sum

            if ACF_rho[lIx] <= 0.1:
                break

            if ACF_rho[lIx] < 0.2:
                if lIx > (window_factor * tau_est):
                    break

            # Warning if we hit the end of the array without 'breaking'
            if lIx == len(ACF_rho) - 1:
                print("Warning: Tau estimation did not converge within the provided lags.")

        return tau_est
    #

    # Calculates Integrated Autocorrelation Time (Hai's paper method eq 19)
    def getTau_ac(self, ACF_rho):
        """ Calculates Integrated Autocorrelation Time (Hai's paper method eq 19).
            Arguments:
                ACF_rho : array-like
            Returns:
                float : estimated integrated autocorrelation time
        """
        tau_est = 0.0

        for lIx in range(1, len(ACF_rho)):
            
            if ACF_rho[lIx] <= 0.1:
                break

            lenRatio = float(lIx) / float(len(ACF_rho))

            tau_est += ACF_rho[lIx] - (lenRatio * ACF_rho[lIx])

        return tau_est
    #

    # Clean an array-like of infs and nans
    def _sanitize_timeseries(self, data):
        """Return finite samples and their original indices."""
        arr = np.asarray(data, dtype=float).reshape(-1)
        finite_mask = np.isfinite(arr)
        clean = arr[finite_mask]
        clean_indices = np.nonzero(finite_mask)[0]
        return clean, clean_indices
    #

    # Detect equilibration using Chodera's method
    def _detect_equilibration_chodera(self, clean_data, lag_fraction=0.1, max_lag=5000):
        """Detect equilibration by maximizing effective uncorrelated samples over t0.

        This mirrors Chodera's criterion: choose t0 that maximizes N_eff(t0) = N_t / g_t,
        where g_t is the statistical inefficiency estimated from the post-t0 segment.
        """
        n = len(clean_data)
        if n < 5:
            return {
                "detected": False,
                "t0_clean": 0,
                "t0_original": 0,
                "g": np.nan,
                "Neff_max": np.nan,
                "n_clean": n,
                "scan_step": 1,
                "reason": "insufficient_clean_samples"
            }

        # Limit search density for long trajectories to avoid O(N^2) scans.
        scan_step = max(1, n // 200)
        t0_candidates = np.arange(0, n - 3, scan_step, dtype=int)
        if t0_candidates.size == 0 or t0_candidates[-1] != (n - 4):
            t0_candidates = np.append(t0_candidates, n - 4)

        best = {
            "t0_clean": 0,
            "g": np.nan,
            "Neff_max": -np.inf,
            "detected": False
        }

        for t0 in t0_candidates:
            seg = clean_data[t0:]
            seg_n = len(seg)
            if seg_n < 4:
                continue

            seg_var = np.var(seg)
            if not np.isfinite(seg_var) or seg_var <= 0:
                continue

            seg_max_lag = self.get_num_lags(seg_n, lag_fraction, max_lag)
            if seg_max_lag < 2:
                continue

            seg_x = seg - np.mean(seg)
            seg_acf = np.array([
                np.sum(seg_x[lag:] * seg_x[:seg_n-lag]) / (seg_n * seg_var)
                for lag in range(seg_max_lag)
            ])

            tau_seg = self.getTau(seg_acf)
            if not np.isfinite(tau_seg) or tau_seg <= 0:
                continue

            # Statistical inefficiency g is approximately tau for this estimator.
            g_seg = max(1.0, float(tau_seg))
            neff_seg = seg_n / g_seg

            if neff_seg > best["Neff_max"]:
                best = {
                    "t0_clean": int(t0),
                    "g": g_seg,
                    "Neff_max": float(neff_seg),
                    "detected": True
                }

        return {
            "detected": bool(best["detected"]),
            "t0_clean": int(best["t0_clean"]),
            "t0_original": int(best["t0_clean"]),
            "g": float(best["g"]) if np.isfinite(best["g"]) else np.nan,
            "Neff_max": float(best["Neff_max"]) if np.isfinite(best["Neff_max"]) else np.nan,
            "n_clean": n,
            "scan_step": int(scan_step),
            "reason": "ok" if best["detected"] else "fallback_to_t0_0"
        }
    #

    # Autocorrelation manual loop (UNREVISED)
    def autocorr2_revised(self, data, lag_fraction=0.1, max_lag=5000, detect_equilibration=False):
        """ Manual: Loop-based (Slow for large max_lag)
        """
        clean_data, clean_indices = self._sanitize_timeseries(data)
        N_clean = len(clean_data)

        # Backward-compatible metadata side-channel.
        self.last_autocorr2_meta = {
            "detect_equilibration": bool(detect_equilibration),
            "n_input": len(np.asarray(data).reshape(-1)),
            "n_clean": N_clean,
            "nan_or_inf_dropped": int(len(np.asarray(data).reshape(-1)) - N_clean),
            "equilibration": None
        }

        if N_clean == 0:
            print("No finite samples; autocorrelation undefined.")
            return (np.array([np.nan]), np.nan, np.nan)

        t0_clean = 0
        if detect_equilibration:
            eq_meta = self._detect_equilibration_chodera(clean_data, lag_fraction=lag_fraction, max_lag=max_lag)
            t0_clean = int(eq_meta["t0_clean"])
            if clean_indices.size > t0_clean:
                eq_meta["t0_original"] = int(clean_indices[t0_clean])
            self.last_autocorr2_meta["equilibration"] = eq_meta

        work_data = clean_data[t0_clean:]
        N = len(work_data)

        if N < 2:
            print("Not enough post-equilibration samples; autocorrelation undefined.")
            return (np.array([np.nan]), np.nan, np.nan)

        miu = np.mean(work_data)
        xp = work_data - miu
        var = np.var(work_data)

        if not np.isfinite(var) or var <= 0:
            print("Variance of data is zero; autocorrelation undefined.")
            return (np.full(N, np.nan), np.nan, np.nan)

        max_lag = self.get_num_lags(N, lag_fraction, max_lag)
        if max_lag < 1:
            return (np.array([1.0]), 1.0, float(N))

        # Calculate ACF up to num_lags for finite, optionally post-equilibration data.
        ACF_rho = np.array([
            np.sum(xp[lag:] * xp[:N-lag]) / (N * var)
            for lag in range(max_lag)
        ])

        tau = self.getTau(ACF_rho)
        ess = N / tau if np.isfinite(tau) and tau > 0 else np.nan

        if detect_equilibration and self.last_autocorr2_meta["equilibration"] is not None:
            self.last_autocorr2_meta["equilibration"]["n_production"] = int(N)
            self.last_autocorr2_meta["equilibration"]["tau_production"] = float(tau) if np.isfinite(tau) else np.nan
            self.last_autocorr2_meta["equilibration"]["ess_production"] = float(ess) if np.isfinite(ess) else np.nan

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





    