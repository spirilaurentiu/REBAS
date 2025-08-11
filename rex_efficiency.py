# rex_validator.py
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
#                      Robosample efficiency estimator
#region REXEfficiency ----------------------------------------------------------
class REXEfficiency:
    '''
    REXValidator provides validation and analysis routines for REX simulation data.

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
        '''
        Calculate exchange rates for each replica.
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
        '''
        Compute autocorrelation function C_k(t) for each replica up to max_lag.

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
        '''
        Compute the average autocorrelation C(t) over all replicas:

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
        '''
        Compute the integrated autocorrelation time τ_ac per seed:

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
    '''
    RoboAnalysis provides validation and analysis routines for simulation data.

    Attributes:
        df: The DataFrame containing simulation data
    '''
    def __init__(self, df):
        self.df = df
    #

    def compute_acceptance(self, cumulative=False):
        '''
        Calculate acceptance rates for one simulation.

        Returns:
            A DataFrame with columns:
                - replicaIx
                - sim_type
                - seed
                - n_acc
                - n_total
                - acc_rate (n_acc / (n_total - 1))
        '''
        results = []
        grouped = self.df.groupby(['replicaIx', 'sim_type', 'seed'])

        for (replica, sim_type, seed), group in grouped:
            acc_series = group['acc'].values
            n_total = len(acc_series)
            if n_total <= 1:
                acc_rate = 0.0
                n_acc = 0
            else:
                n_acc = (acc_series).sum()
                acc_rate = n_acc / (n_total)
            results.append({
                'replicaIx': replica,
                'sim_type': sim_type,
                'seed': seed,
                'n_acc': n_acc,
                'n_total': n_total,
                'exchange_rate': acc_rate
            })
            
        return pd.DataFrame(results)
    #
#endregion --------------------------------------------------------------------




