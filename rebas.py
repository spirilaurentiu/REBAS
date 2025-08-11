# main.py
import argparse
import os
import re
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rex_data import REXData
from rex_efficiency import REXEfficiency
from rex_efficiency import REXEnergy



# -----------------------------------------------------------------------------
#                      Robosample file manager
#region REXFileManager --------------------------------------------------------
class REXFNManager:
    ''' File manager
    Attributes:
        dir: Directory containing the files
        FNRoots: File prefixes
    '''
    def __init__(self, dir, FNRoots, SELECTED_COLUMNS):
        self.dir = dir
        self.FNRoots = FNRoots
        self.SELECTED_COLUMNS = SELECTED_COLUMNS
    #

    def getSeedAndTypeFromFN(self, FN):
        """ Parse filename of type out.<7-digit seed>
        """
        match = re.match(r"out\.(\d{7})", FN)
        if not match: raise ValueError(f"Filename '{FN}' does not match expected format 'out.<7-digit-seed>'")

        seed = match.group(1)
        sim_type = seed[2]  # third digit (0-based index)
        return seed, sim_type
    #

    def getDataFromFile(self, FN, seed, sim_type):
        """ Read data from file
        """
        rex = REXData(FN, self.SELECTED_COLUMNS)
        df = rex.get_dataframe().copy()
        df['seed'] = seed
        df['sim_type'] = sim_type
        return df
    #

    def getDataFromAllFiles(self):
        """ Grab all files and read data from them
        """
        all_data = []
        for FNRoot in self.FNRoots:
            matches = glob.glob(os.path.join(self.dir, FNRoot + "*"))
            if not matches:
                print(f"Warning: No files found matching {FNRoot}*", file=sys.stderr)
                continue

            for filepath in matches:

                print("Reading", filepath, "...", end = ' ', flush=True)

                try:
                    seed, sim_type = self.getSeedAndTypeFromFN(os.path.basename(filepath))
                except ValueError as e:
                    print(f"Skipping file due to naming error: {filepath}\n{e}", file=sys.stderr)
                    continue

                df = self.getDataFromFile(filepath, seed, sim_type)
                all_data.append(df)

                print("done.", flush=True)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    #
#endregion --------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                                MAIN
#region Main ------------------------------------------------------------------

def parse_filters(filters):
    """Convert list of 'col=value' strings into a dict"""
    filter_dict = {}
    for f in filters:
        if '=' not in f:
            raise ValueError(f"Invalid filter format: '{f}' (expected col=value)")
        key, val = f.split('=', 1)
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass  # Leave as string
        filter_dict[key] = val
    return filter_dict

def main(args):

    # Get all the data
    if args.useCache and os.path.exists(args.cacheFile):
        print(f"Loading data from cache: {args.cacheFile}")
        df = pd.read_pickle(args.cacheFile)
    else:
        FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols)
        df = FNManager.getDataFromAllFiles()

        if args.writeCache:
            if os.path.exists(args.cacheFile):
                raise FileExistsError(f"Cache file '{args.cacheFile}' already exists. Use a different name or delete it.")
            print(f"Writing data to cache: {args.cacheFile}")
            df.to_pickle(args.cacheFile)

    # Apply filters if specified
    if args.filterBy:
        filters = parse_filters(args.filterBy)
        for col, val in filters.items():
            if col not in df.columns:
                raise ValueError(f"Filter column '{col}' not found in DataFrame columns.")
            df = df[df[col] == val]

    print(df)

    potentialEnergyFigure = False
    efficiencyFigure = True

    if potentialEnergyFigure:
        E_analyzer = REXEnergy(df)
        pe_histograms = E_analyzer.pe_o_histograms(bins=50)
        print(pe_histograms)

    # Plot each seed's histogram
        plt.figure()
        for (seed, thermoIx), (hist, bin_edges) in pe_histograms.items():
            #plt.xlim(-2000, 0)

            plt.plot(bin_edges[:-1], hist)

        plt.title(f"Histogram of pe_o for seed {seed}")
        plt.xlabel("pe_o")
        plt.ylabel("Density")
        plt.legend(pe_histograms.keys(), title="Seeds")
        plt.grid(True)
        plt.tight_layout()
        plt.show()        

    if efficiencyFigure:

        # Evaluate efficiency
        rexEff = REXEfficiency(df)

        # # Calculate exchange rate
        eff_df = rexEff.calc_exchange_rates() 
        print(eff_df)

        # Calculate autocorrelation function
        max_lag = 50
        acorCk_df = rexEff.compute_autocorrelation(max_lag) # per replica
        print(acorCk_df)

        acorC_df = rexEff.compute_mean_autocorrelation(max_lag) # total
        print(acorC_df)

        tau_df = rexEff.compute_autocorrelation_time(max_lag) # total autocorrelation time
        print(tau_df)    

#endregion --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True,
                   help='Directory with data files')
    parser.add_argument('--inFNRoots', nargs='+', required=True,
                   help='Robosample processed output file names')
    parser.add_argument('--cols', nargs='+', required=True,
                   help='Columns to be read')
    parser.add_argument('--filterBy', nargs='*', default=[],
                   help='Optional filters in the format col=value (e.g. wIx=0)')
    parser.add_argument('--useCache', action='store_true', help='Load data from cache file if it exists')
    parser.add_argument('--writeCache', action='store_true', help='Write new cache file (fails if file exists)')
    parser.add_argument('--cacheFile', default='rex_cache.pkl', help='Path to cache file')
    args = parser.parse_args()

    main(args)

# SELECTED_COLUMNS = [
#     "replicaIx", "thermoIx", "wIx", "T", 
#     "ts", "mdsteps",
#     "pe_o", "pe_n", "pe_set",
#     "ke_prop", "ke_n",
#     "fix_o", "fix_n",
#     "logSineSqrGamma2_o", "logSineSqrGamma2_n",
#     "etot_n", "etot_proposed",
#     "JDetLog",
#     "acc", "MDorMC"
# ]

# Header description:
# "replicaIx" means replica,
# "thermoIx" means temperature index,
# "wIx" means what part of the molecule was simulated as a Gibbs block,
# "ts" means the timestep used to integrate,
# "mdsteps" how many steps was integrated for a Hamiltonian Monte Carlo trial
# "pe_o" means
# 