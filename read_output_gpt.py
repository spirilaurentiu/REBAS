import glob, os, sys
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

class REXData:
    def __init__(self, inFN):

        self.filepath = inFN
        self.df = self._load_data()
    
    def _load_data(self):
        # Load only lines starting with "REX," (skip header and initial shell prompt)
        with open(self.filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip().startswith('REX,')]

        if not lines:
            raise ValueError("No REX data found in the file.")

        # Parse header
        header_line = lines[0] + ", unif, HARDMOLNAME"
        for errIx in range(10):
            print("ADDED FIELDS TO REX OUTPUT UNTIL ROBOSAMPLE UPDATED", file=sys.stderr)

        headers = [h.strip() for h in header_line.split(',')][1:-1]  # remove first and last columns

        # Parse data lines
        data_lines = lines[1:]
        cleaned_data = []
        for line in data_lines:
            fields = [f.strip() for f in line.split(',')]
            if len(fields) < len(headers) + 2:
                continue  # Skip malformed lines
            cleaned_data.append(fields[1:-1])  # remove first and last fields

        # Create DataFrame
        df = pd.DataFrame(cleaned_data, columns=headers)
        
        # Convert numeric columns
        df = df.apply(pd.to_numeric, errors='ignore')
        return df

    def get_dataframe(self):
        return self.df

    def summary_stats(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not in data.")
        return self.df[column].describe()

    def get_unique_replicas(self):
        return self.df['replicaIx'].unique()

    def count_exchanges_per_replica(self):
        """
        Returns a dictionary mapping each replicaIx to its number of exchanges
        (thermoIx changes over increasing nofSamples).
        """
        exchange_counts = {}
    
        for replica, group in self.df[self.df['wIx'] == 0].groupby('replicaIx'):
            prev_thermoIx = None
            count = 0
            for thermoIx in group['thermoIx']:
                if prev_thermoIx is not None and thermoIx != prev_thermoIx:
                    count += 1
                prev_thermoIx = thermoIx
            exchange_counts[replica] = count
    
        return exchange_counts


    def plot_replica_trajectories(self,save_path=None):
        """
        Plots thermoIx (temperature index) over nofSamples for each replicaIx.
        """
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
        """
        Returns a DataFrame where entry (i, j) is the number of times a replica moved
        between thermoIx i and j (i ≠ j).
        """
        states = sorted(self.df['thermoIx'].unique())
        matrix = pd.DataFrame(0, index=states, columns=states)
    
        for _, group in self.df[self.df['wIx'] == 0].groupby('replicaIx'):
            prev = None
            for thermoIx in group['thermoIx']:
                if prev is not None and thermoIx != prev:
                    matrix.at[prev, thermoIx] += 1
                    matrix.at[thermoIx, prev] += 1  # symmetric
                prev = thermoIx
    
        return matrix
    
    def trace_replica_path(self, replica_id):
        """
        Returns a list of (nofSamples, thermoIx) for the specified replica.
        """
        group = self.df[self.df['replicaIx'] == replica_id].sort_values('nofSamples')
        return list(zip(group['nofSamples'], group['thermoIx']))



# -----------------------------------------------------------------------------
#                            Parse arguments
#region Parse arguments -------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None,
  help='Directory with data files')
parser.add_argument('--extension', default='',
  help='Data files extension.')
parser.add_argument('--inFNRoots', default=[], nargs='+',
    help='Robosample processed output file root names')
args = parser.parse_args()

# -----------------------------------------------------------------------------
#                            Main function
#region -----------------------------------------------------------------------
def main(dir, inFNRoots):

    filepath = dir + inFNRoots[0]
    rex = REXData(filepath)
    df = rex.get_dataframe()

    # replicaIx, thermoIx, wIx, T, boostT, ts, mdsteps, DISTORT_OPTION, NU, nofSamples, pe_o, pe_n, pe_set, ke_prop, ke_n, fix_o, fix_n, logSineSqrGamma2_o, logSineSqrGamma2_n , etot_n, etot_proposed, JDetLog , acc , MDorMC, unif
    #print(df.head())  # View first few parsed rows
    #print(df["wIx"])
    # print(df.to_numpy().size)
    # print(df[df['wIx'] == 0].to_numpy().size)
    #print( df[df['wIx'] == 0] )
    # df_grouped_by_thermoIx = df.groupby('thermoIx')
    # for thermoIx, thermoGroup in df_grouped_by_thermoIx:
    #     print(thermoGroup)
    # exit(0)

    # Get all rows where T = 300
    # filtered = rex.filter_by_temperature(300)
    # print(filtered)

    # Summary stats for 'ke_prop'
    #print(rex.summary_stats('ke_prop'))

    # Unique replica indexes
    print(rex.count_exchanges_per_replica())


    # Plot all replica trajectories
    rex.plot_replica_trajectories("x.pdf")

    # Get and display the pairwise exchange matrix
    print(rex.pairwise_exchange_counts())

    # Trace the path of replica 0
    # path = rex.trace_replica_path(0)
    # print(path[0])

# -----------------------------------------------------------------------------
#                            Main call
#region -----------------------------------------------------------------------
if __name__=="__main__":

    if not args.inFNRoots:
        parser.error("Output filenames roots are required.")

    main(args.dir, args.inFNRoots)
#endregion --------------------------------------------------------------------

