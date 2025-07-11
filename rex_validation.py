# rex_validation.py
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import pandas as pd
import seaborn as sns

class ExperimentsValidations:
    def __init__(self, df):
        self.df = df

    def summary_stats(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not in data.")
        return self.df[column].describe()

    def get_unique_replicas(self):
        return self.df['replicaIx'].unique()

    def count_exchanges_per_replica(self):
        exchange_counts = {}
        for replica, group in self.df[self.df['wIx'] == 0].groupby('replicaIx'):
            prev_thermoIx = None
            count = 0
            for thermoIx in group['thermoIx']:
                if prev_thermoIx is not None and thermoIx != prev_thermoIx:
                    count += 1
                prev_thermoIx = thermoIx
            exchange_counts[replica] = count / group['nofSamples'][-1]
        return exchange_counts

    def plot_replica_trajectories(self, save_path=None):
        plt.figure(figsize=(12, 6))
        for replica, group in self.df[self.df['wIx'] == 0].groupby('replicaIx'):
            plt.plot(group['nofSamples'], group['thermoIx'], label=f"Replica {replica}")
        plt.xlabel('nofSamples')
        plt.ylabel('thermoIx')
        plt.title('Replica Trajectories Across Thermodynamic States')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def pairwise_exchange_counts(self):
        states = sorted(self.df['thermoIx'].unique())
        matrix = pd.DataFrame(0, index=states, columns=states)
        for _, group in self.df[self.df['wIx'] == 0].groupby('replicaIx'):
            prev = None
            for thermoIx in group['thermoIx']:
                if prev is not None and thermoIx != prev:
                    matrix.at[prev, thermoIx] += 1
                    matrix.at[thermoIx, prev] += 1
                prev = thermoIx
        return matrix

    def plot_pairwise_exchange_matrix(self, save_path=None):
        matrix = self.pairwise_exchange_counts()
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='viridis')
        plt.title("Pairwise Exchange Counts Between ThermoIx States")
        plt.xlabel("To")
        plt.ylabel("From")
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def trace_replica_path(self, replica_id):
        group = self.df[self.df['replicaIx'] == replica_id].sort_values('nofSamples')
        return list(zip(group['nofSamples'], group['thermoIx']))
