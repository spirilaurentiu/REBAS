# rex_validator.py
import pandas as pd
import numpy as np

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

    def summary_stats(self):
        ''' Return basic summary statistics of numeric columns '''
        return self.df.describe(include='all')
    #

    def calc_exchange_rates(self):
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


        burnin = 1000

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
        grouped = per_replica.groupby(['seed', 'lag'])['autocorrelation'].mean().reset_index()
        grouped.rename(columns={'autocorrelation': 'mean_autocorrelation'}, inplace=True)
        return grouped
    #

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
        T = max_lag
        grouped = []

        for seed, group in mean_corr.groupby('seed'):
            tau_ac = sum((1 - (row['lag'] / T)) * row['mean_autocorrelation'] for _, row in group.iterrows())
            grouped.append({'seed': seed, 'autocorrelation_time': tau_ac})

        return pd.DataFrame(grouped)
    #     
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

