# rex_validator.py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------
#                      Robosample efficiency estimator
#region REXEfficiency ----------------------------------------------------------
class REXEfficiency:
    ''' REXEfficiency provides analysis routines for REX simulation data.
    Attributes:
        df: The DataFrame containing simulation data
    '''
    def __init__(self, df):
        self.df = df
    #

    def get_dataframe(self):
        return self.df
    #

    # Basic summary statistics
    def summary_stats(self):
        ''' Return basic summary statistics of numeric columns '''
        return self.df.describe(include='all')
    #

    # Calculate exchange rates for each replica
    def calc_exchange_rates(self, burnin=1024): # 
        ''' Calculate exchange rates for each replica.
        Exchange rate is defined as the fraction of times a replica changes its thermoIx.
        Returns:
            A DataFrame with columns:
                - replicaIx
                - sim_type
                - seed
                - n_transitions
                - n_total
                - exchange_rate (transitions / (n_total - 1))
        '''
        results = []
        grouped = self.df.groupby(['replicaIx', 'sim_type', 'seed'])

        for (replica, sim_type, seed), group in grouped:
            thermo_series = group['thermoIx'].values

            thermo_series = thermo_series[burnin:]  # Skip burn-in period

            n_total = len(thermo_series)
            if n_total <= 1:
                exchange_rate = 0.0
                n_transitions = 0
            else:
                n_transitions = (thermo_series[1:] != thermo_series[:-1]).sum()
                exchange_rate = n_transitions / (n_total - 1)
            results.append({
                'replicaIx': replica,
                'sim_type': sim_type,
                'seed': seed,
                'n_transitions': n_transitions,
                'n_total': n_total,
                'exchange_rate': exchange_rate
            })

        return pd.DataFrame(results)
    #

    # Compute C_k(t) for each replica
    def compute_autocorrelation(self, max_lag):
        ''' Compute autocorrelation function C_k(t) for each replica up to max_lag.
        C_k(t) = (<M_k(s) M_k(s + t)> - <M_k(s)>^2) / (<M_k(s)^2> - <M_k(s)>^2)
        Arguments:
            max_lag: Maximum time lag (t) to compute autocorrelation for
        Returns:
            A DataFrame with columns:
                - replicaIx
                - sim_type
                - seed
                - lag
                - autocorrelation
        '''
        results = []
        grouped = self.df.groupby(['replicaIx', 'sim_type', 'seed'])
        for (replica, sim_type, seed), group in grouped:
            M = group['thermoIx'].values
            M = M - np.mean(M)
            denom = np.mean(M**2)
            if denom == 0:
                continue

            for lag in range(1, max_lag + 1):
                if lag >= len(M):
                    break
                autocov = np.mean(M[:-lag] * M[lag:])
                autocorr = autocov / denom
                results.append({
                    'replicaIx': replica,
                    'sim_type': sim_type,
                    'seed': seed,
                    'lag': lag,
                    'autocorrelation': autocorr
                })

        return pd.DataFrame(results)
    #

    # region myverification
    # def compute_autocorrelation_me(self, max_lag):
    #     grouped = self.df.groupby(['replicaIx', 'sim_type', 'seed'])
    #     for (replica, sim_type, seed), group in grouped:
    #         print(group)
    #         print(group["thermoIx"])
    #         M_k_s = group["thermoIx"].values
    #         miu = np.mean(M_k_s) # mean
    #         miumiu = miu * miu # square mean
    #         miu_2 = np.mean(M_k_s**2) # second moment
    #         q_term = np.empty(max_lag) * np.nan
    #         C_k_t = np.empty(max_lag) * np.nan
    #         for lag in range(1, max_lag + 1):
    #             q_term[lag-1] = np.mean(M_k_s[:-lag] * M_k_s[lag:])
    #             C_k_t[lag-1] = (q_term[lag-1] - miumiu) / (miu_2 - miumiu)
    #         return C_k_t
    # endregion myverification

    # Compute the average autocorrelation C(t) for each 
    def compute_mean_autocorrelation(self, max_lag):
        ''' Compute the average autocorrelation C(t) over all replicas:
        C(t) = (1/K) * sum_k C_k(t)
        Arguments:
            max_lag: Maximum lag to compute C(t)
        Returns:
            A DataFrame with columns:
                - seed
                - lag
                - mean_autocorrelation
        '''
        per_replica = self.compute_autocorrelation(max_lag)

        # region study
        # print("per_replica\n", per_replica)
        # grouped_obj = per_replica.groupby(['seed', 'lag'])
        # for group in grouped_obj:
        #     print("grouped_obj group\n", group)
        # autocorr_groups = grouped_obj['autocorrelation']
        # for group in autocorr_groups:
        #     print("autocorr_groups group\n", group)
        # endregion study

        grouped = per_replica.groupby(['seed', 'lag'])['autocorrelation'].mean().reset_index()
        grouped.rename(columns={'autocorrelation': 'mean_autocorrelation'}, inplace=True)
        return grouped
    #

    def exp_func(self, t, tau, C0=1.0):
        return C0 * np.exp(-t / tau)
        
    # Compute the integrated autocorrelation time τ_ac per seed
    def compute_autocorrelation_time(self, max_lag):
        ''' Compute the integrated autocorrelation time τ_ac per seed:
        τ_ac = sum_{t=1}^{T} [1 - (t / T)] * C(t)
        Arguments:
            max_lag: Maximum lag (T) to use in the summation
        Returns:
            A DataFrame with columns:
                - seed
                - autocorrelation_time
        '''
        mean_corr = self.compute_mean_autocorrelation(max_lag)

        # Fit on an exponential?
        lags = mean_corr['lag'].values          # t
        C_t = mean_corr['mean_autocorrelation'].values  # C(t)
        max_fit_lag = int((100.0/100.0) * max_lag)
        lags_fit = lags[:max_fit_lag]
        C_t_fit = C_t[:max_fit_lag]
        tau_guess = 10  
        popt, pcov = curve_fit(self.exp_func, lags_fit, C_t_fit, p0=[tau_guess])
        tau_fit = popt[0]
        print(f"Fitted autocorrelation time tau = {tau_fit:.2f}")

        T = max_lag
        grouped = []

        for seed, group in mean_corr.groupby('seed'):
            tau_ac = sum((1 - (row['lag'] / T)) * row['mean_autocorrelation'] for _, row in group.iterrows())
            grouped.append({'seed': seed, 'autocorrelation_time': tau_ac})

        return pd.DataFrame(grouped)
    # 

    def compute_tau2(self):
        """Estimate relaxation time tau_2 from the empirical transition matrix.

        Returns:
            DataFrame with columns:
                - seed
                - tau_2
                - mu2 (second-largest eigenvalue)
        """
        results = []

        grouped = self.df.groupby("seed")
        for seed, group in grouped:
            states = group["thermoIx"].values
            K = states.max() + 1  # assuming thermoIx runs from 0..K-1

            # Build count matrix Nij
            Nij = np.zeros((K, K), dtype=int)
            for i in range(len(states) - 1):
                a, b = states[i], states[i+1]
                Nij[a, b] += 1

            # Symmetrize
            T = np.zeros((K, K), dtype=float)
            for i in range(K):
                denom = sum(Nij[i, k] + Nij[k, i] for k in range(K))
                if denom > 0:
                    for j in range(K):
                        T[i, j] = (Nij[i, j] + Nij[j, i]) / denom

            # Print the transition matrix
            print("\nEmpirical transition matrix for seed ", seed)
            for i in range(K):
                for j in range(K):
                    print(T[i, j], end = ' ')
                print()
            print()

            # Diagonalize
            eigvals = np.linalg.eigvals(T)
            eigvals = np.real(eigvals)  # should be real for stochastic T
            eigvals = np.sort(eigvals)[::-1]  # descending order

            mu2 = eigvals[1] if len(eigvals) > 1 else np.nan
            tau2 = 1.0 / (1.0 - mu2) if mu2 < 1 else np.inf

            results.append({
                "seed": seed,
                "mu2": mu2,
                "tau_2": tau2
            })

        return pd.DataFrame(results)
    
    def compute_tau_p(self):
        """Estimate mean first-passage time τ_p from thermoIx trajectories.

        τ_p = mean number of exchanges required for a replica to go from state 1 to K.

        Returns:
            DataFrame with columns:
                - seed
                - tau_p
        """
        results = []

        grouped = self.df.groupby("seed")
        for seed, group in grouped:
            states = group["thermoIx"].values
            unique_states = np.unique(states)
            K = len(unique_states)

            # Map states to 0..K-1 in case thermoIx is not contiguous
            state_map = {s: i for i, s in enumerate(sorted(unique_states))}
            states = np.array([state_map[s] for s in states])

            # Build Nij
            Nij = np.zeros((K, K), dtype=int)
            for i in range(len(states) - 1):
                a, b = states[i], states[i+1]
                Nij[a, b] += 1

            # Symmetrize
            T = np.zeros((K, K), dtype=float)
            for i in range(K):
                denom = sum(Nij[i, k] + Nij[k, i] for k in range(K))
                if denom > 0:
                    for j in range(K):
                        T[i, j] = (Nij[i, j] + Nij[j, i]) / denom

            # Build absorbing Markov chain T'
            Tprime = T.copy()
            Tprime[K-1, :] = 0.0
            Tprime[K-1, K-1] = 1.0

            # Q = top-left (K-1)x(K-1)
            Q = Tprime[:K-1, :K-1]

            # Fundamental matrix
            I = np.eye(K-1)
            try:
                U = np.linalg.inv(I - Q)
            except np.linalg.LinAlgError:
                results.append({"seed": seed, "tau_p": np.inf})
                continue

            # MFPT from state 1 → K (state index 0-based)
            tau_p = U[0, :].sum()

            results.append({"seed": seed, "tau_p": tau_p})

        return pd.DataFrame(results)    
         
#endregion --------------------------------------------------------------------


# -----------------------------------------------------------------------------
#                      Robosample per replica analysis
#region Robosample ------------------------------------------------------------
class RoboAnalysis:
    """ RoboAnalysis provides validation and analysis routines for simulation data.
    Attributes:
        df: The DataFrame containing simulation data
    """
    def __init__(self, df):
        self.df = df
    #

    def get_dataframe(self):
        return self.df
    #

    # Calculate acceptance rates for one simulation
    def compute_acceptance(self, cumulative=False):
        ''' Calculate acceptance rates for one simulation.

        Returns:
            A DataFrame with columns:
                - thermoIx
                - sim_type
                - seed
                - n_acc
                - n_total
                - acc_rate (n_acc / (n_total - 1))
        '''
        results = []
        grouped = self.df.groupby(['thermoIx', 'sim_type', 'seed'])

        for (thermoIx, sim_type, seed), group in grouped:
            acc_series = group['acc'].values
            n_total = len(acc_series)
            if n_total <= 1:
                acc_rate = 0.0
                n_acc = 0
            else:
                n_acc = (acc_series).sum()
                acc_rate = n_acc / (n_total)
            results.append({
                'thermoIx': thermoIx,
                'sim_type': sim_type,
                'seed': seed,
                'n_acc': n_acc,
                'n_total': n_total,
                'acc_rate': acc_rate
            })
            
        return pd.DataFrame(results)
    #

    # Compute histograms of values from any given column per (seed, thermoIx)
    def column_histograms(self, column, bins=50, lower_percentile=0.005, upper_percentile=99.995):
        ''' Compute histograms of values from any given column per (seed, thermoIx).
        Arguments:
            column : str
                Column name in self.df to histogram
            bins   : int or sequence
                Number of bins or bin edges to use for histogramming
            lower_percentile : float
                Lower cutoff percentile for bin range
            upper_percentile : float
                Upper cutoff percentile for bin range
        Returns:
            Dictionary mapping (seed, thermoIx) -> (hist, bin_edges)
        '''
        if column not in self.df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame")
            return {}

        histograms = {}
        grouped = self.df.groupby(['seed', 'thermoIx'])

        col_clean = self.df[column].dropna()
        vmin = np.percentile(col_clean, lower_percentile)
        vmax = np.percentile(col_clean, upper_percentile)
        bin_edges = np.linspace(vmin, vmax, bins + 1)

        for (seed, thermoIx), group in grouped:
            values = group[column].dropna().values
            hist, _ = np.histogram(values, bins=bin_edges, density=True)
            histograms[(seed, thermoIx)] = (hist, bin_edges)

        return histograms


    # Compute histograms of ΔE = pe_n - pe_o per (seed, thermoIx).
    def delta_pe_histograms(self, bins=50, lower_percentile=0.005, upper_percentile=99.995):
        ''' Compute histograms of ΔE = pe_n - pe_o per (seed, thermoIx).
        Arguments:
            bins : Number of bins or bin edges to use for histogramming
            lower_percentile : Lower bound percentile for histogram range (default 0.005)
            upper_percentile : Upper bound percentile for histogram range (default 99.995)
        Returns:
            Dictionary mapping (seed, thermoIx) -> (hist, bin_edges)
        '''
        histograms = {}
        grouped = self.df.groupby(['seed', 'thermoIx'])

        # Compute ΔE and clean NaNs
        delta_pe = (self.df['pe_n'] - self.df['pe_o']).dropna()
        vmin = np.percentile(delta_pe, lower_percentile)
        vmax = np.percentile(delta_pe, upper_percentile)
        bin_edges = np.linspace(vmin, vmax, bins + 1)

        for (seed, thermoIx), group in grouped:
            values = (group['pe_n'] - group['pe_o']).dropna().values
            hist, _ = np.histogram(values, bins=bin_edges, density=True)
            histograms[(seed, thermoIx)] = (hist, bin_edges)

        return histograms
    #
#endregion --------------------------------------------------------------------

