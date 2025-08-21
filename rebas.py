# main.py
import argparse
import os
import re
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from rex_data import REXData
from rex_efficiency import RoboAnalysis, REXEfficiency



# -----------------------------------------------------------------------------
#                      Robosample file manager
#region REXFileManager --------------------------------------------------------
import MDAnalysis as mda
#from MDAnalysis.coordinates.TRJ import Restart
import mdtraj as md

try:
    import parmed as pmd
    _HAS_PARMED = True
except Exception:
    _HAS_PARMED = False

class REXFNManager:
    """ File manager
    Attributes:
        dir: Directory containing the files
        FNRoots: File prefixes
        SELECTED_COLUMNS: columns selected to be read from file
    """
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

    # Write restart files into self.dir/restDir/restDir.<seed>
    def write_restarts_from_trajectories(self, restDir, topology, out_ext='rst7', dry=False):
        """ Read trajectory files of the form <mol>_<seed>.<replica>.dcd from self.dir,
        extract the last frame, and write restart files into self.dir/restDir/restDir.<seed>.
        Arguments:
            restDir : Name of the subdirectory (inside self.dir) to store restart files
            topology: Path to a topology file (AMBER prmtop or PDB) required to read the DCDs
            out_ext : Output extension/format. 'rst7' (default) requires ParmEd; if
                      ParmEd is unavailable, a PDB will be written instead.
        Output:
            For each trajectory file <mol>_<seed>.<replica>.dcd, writes:
                self.dir/restDir/restDir.<seed>/<mol>_<seed>.<replica>.<out_ext>
        """
        
        traj_files = sorted(glob.glob(os.path.join(self.dir, '*.dcd')))
        if not traj_files:
            print(f"Warning: No trajectory files found in {self.dir}", file=sys.stderr)
            return

        for traj in traj_files:
            base = os.path.basename(traj)  # e.g. protein_1234567.0.dcd
            root, _ = os.path.splitext(base)  # protein_1234567.0

            # Extract seed from filename assuming <mol>_<seed>.<replica>.dcd
            try:
                parts = root.split('_') # protein 1234567.0
                seed_part = parts[-1].split('.')[0]  # after "_"
                seed = seed_part
                replica_part = parts[-1].split('.')[1]  # after "."
                replica = int(replica_part.replace('repl', ''))
            except Exception:
                print(f"Could not parse seed from {base}", file=sys.stderr)
                continue

            # Create per-seed subdirectory: restDir/restDir.<seed>
            seed_dir = os.path.join(self.dir, restDir, f"{restDir}.{seed}")
            if not dry:
                os.makedirs(seed_dir, exist_ok=True)

            try:
                # Load trajectory with topology
                t = md.load_dcd(traj, top=topology)
                last = t[-1]  # last frame

                # Decide extension
                ext = out_ext.lower()
                if ext == 'rst7' and not _HAS_PARMED:
                    print(f"ParmEd not available; writing PDB instead for {base}", file=sys.stderr)
                    ext = 'pdb'

                if ext == 'rst7':
                    # ParmEd path
                    struct = pmd.load_file(topology)
                    coords_ang = last.xyz[0] * 10.0
                    struct.coordinates = coords_ang

                    if (last.unitcell_lengths is not None) and (last.unitcell_angles is not None):
                        lengths = (last.unitcell_lengths[0] * 10.0).tolist()
                        angles = last.unitcell_angles[0].tolist()
                        struct.box = lengths + angles

                    out_path = os.path.join(seed_dir, f"ligand.s{replica}.rst7")
                    if not dry: struct.save(out_path, overwrite=True)

                elif ext == 'pdb':
                    out_path = os.path.join(seed_dir, f"ligand.s{replica}.pdb")
                    if not dry: last.save_pdb(out_path)

                else:
                    out_path = os.path.join(seed_dir, f"ligand.s{replica}.pdb")
                    if not dry: last.save_pdb(out_path)

                print(f"Wrote restart: {out_path}")

            except Exception as e:
                print(f"Error processing {traj}: {e}", file=sys.stderr)

#endregion --------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                                MAIN
#region Main ------------------------------------------------------------------

# Parse a set of filters given as arguments in the format: ---filterBy col=val
def parse_filters(filters):
    """ Convert list of 'col=value' strings into a dict.
        Supports multiple OR values separated by commas, e.g. 'wIx=0,1'
    """
    filter_dict = {}
    for currFilter in filters:
        if '=' not in currFilter:
            raise ValueError(f"Invalid filter format: '{currFilter}' (expected col=value or col=val1,val2,...)")

        key, val_str = currFilter.split('=', 1)

        # Split on commas to allow OR logic
        vals = val_str.split(',')

        parsed_vals = []
        for v in vals:
            v = v.strip()
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass  # leave as string
            parsed_vals.append(v)

        # Store single values as scalar, multi as list
        filter_dict[key] = parsed_vals if len(parsed_vals) > 1 else parsed_vals[0]

    return filter_dict
#

# Generic plotting function for timelines
def plot_column(df, col, save_path=None):
    """ Plot values of a single column in the order they appear in the DataFrame.
    Arguments:
        df : DataFrame
        col : column name to plot
        save_path : optional, if given save plot to file
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    grouped = df.groupby(['seed', 'thermoIx'])
    plt.figure()
    
    colors = ["blue", "red", "black", "grey", "green", "brown", "cyan", "purple", "yellow", "pink"]
    for i, ((seed, thermoIx), group) in enumerate(grouped):
        color = colors[i % len(colors)]
        plt.plot(group[col].to_numpy(), marker='.', linestyle='-', alpha=0.7, color=color, label=f"seed {seed}, thermo {thermoIx}")

    plt.title(f"Plot of {col} grouped by seed & thermoIx")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
#

# Generic plotting function for histogram dictionaries
def plot_histogram(hist_dict, title="title", xlabel="x", ylabel="Density", save_path=None):
    """ Generic plotting function for histogram dictionaries of the form:
        {(seed, thermoIx): (hist, bin_edges), ...}
    Arguments:
        hist_dict : dict
            Dictionary mapping keys (e.g. (seed, thermoIx)) -> (hist, bin_edges)
        title     : str
            Title of the plot
        xlabel    : str
            Label for the x-axis
        ylabel    : str
            Label for the y-axis
        save_path : str
            File path to save the figure
    """
    plt.figure(figsize=(8, 6))
    colors = [
        "blue", "red", "black", "grey", "green",
        "brown", "cyan", "purple", "yellow", "pink"
    ]

    for idx, (key, (hist, bin_edges)) in enumerate(hist_dict.items()):
        color = colors[idx % len(colors)]
        plt.plot(bin_edges[:-1], hist, color=color, label=str(key))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Group")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()
#

#region MAIN function for the REBAS paper. Has two types of calculations:
#   (I)   In-house checks
#   (II)  Figures for the paper
# Types of files:
#   1) OUTPUT
#   2) TRAJECTORY
# Types of figures:
#   1) Validation figures
#       1.1) Potential energy based validation
#       1.2) Free energy based validation
#   2) Efficiency figures
#       2.1) Exchange rates
#       2.2) Autocorrelation-based functions
#endregion
def main(args):

    OUTPUT_REQUIRED, TRAJECTORY_REQUIRED = True, False # flags
    FNManager = None # classes
    out_df, traj_df = None, None # pandas

    #region Read output from all files
    if args.useCache and os.path.exists(args.cacheFile):
        print(f"Loading data from cache: {args.cacheFile}")
        out_df = pd.read_pickle(args.cacheFile)
    else:
        FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols)
        out_df = FNManager.getDataFromAllFiles()

        if args.writeCache:
            if os.path.exists(args.cacheFile):
                raise FileExistsError(f"Cache file '{args.cacheFile}' already exists. Use a different name or delete it.")
            print(f"Writing data to cache: {args.cacheFile}")
            out_df.to_pickle(args.cacheFile)                

    # Apply filters if specified
    if args.filterBy:
        filters = parse_filters(args.filterBy)
        for col, val in filters.items():
            if col not in out_df.columns:
                raise ValueError(f"Filter column '{col}' not found in DataFrame columns.")
            if isinstance(val, list):  # multiple OR values
                out_df = out_df[out_df[col].isin(val)]
            else:  # single value
                out_df = out_df[out_df[col] == val]

    print(out_df)
    #endregion

    #region Restart: write restart files into self.dir/restDir/restDir.<seed>
    if (args.restDir):
        TRAJECTORY_REQUIRED = True
        FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols)
        FNManager.write_restarts_from_trajectories(args.restDir, args.topology, dry=args.dry)
    #endregion

    #region In-house basic checks
    # Column histograms checks
    if len(args.basicChecks) > 0:

        roboAna = RoboAnalysis(out_df)

        want_histogram = False
        for column in args.basicChecks:
            if want_histogram:
                hist_df = roboAna.column_histograms(column, bins=50)
                if not hist_df:
                    print(f"Warning: no histogram data found for column '{column}'")
                    continue  
                save_path = f"check_{column}_hist.png"
                plot_histogram(hist_df, save_path=save_path)
            else:
                save_path = f"check_{column}.png"
                plot_column(out_df, column, save_path=save_path)
    #endregion

    #region In-house other checks
    # Acceptance
    if 'acceptance' in args.checks:
        roboAna = RoboAnalysis(out_df)

        # Acceptance rate
        acc_df = roboAna.compute_acceptance(cumulative=True)
        print("Acceptance rates")
        print(acc_df)

    # Delta potential energy
    if 'dpe' in args.checks:
        roboAna = RoboAnalysis(out_df)
        dpeHist_df = roboAna.delta_pe_histograms(bins=50)
        plot_histogram(dpeHist_df, save_path=f"check_dpe_hist.png")
    #endregion



    #region Paper figures: Validation
    if "pe_o" in args.figures:
        roboAna = RoboAnalysis(out_df)

        hist_df = roboAna.column_histograms(bins=50)
        plot_histogram(hist_df, save_path="x.png")
    #endregion

    #region Paper figures: Efficiency
    if "tau_ac" in args.figures:

        rexEff = REXEfficiency(out_df)

        # Calculate exchange rate
        exch_df = rexEff.calc_exchange_rates()
        print("Exchange rates")
        print(exch_df)

        # Calculate autocorrelation function
        max_lag = 50
        acorCk_df = rexEff.compute_autocorrelation(max_lag) # per replica
        #print(acorCk_df)

        acorC_df = rexEff.compute_mean_autocorrelation(max_lag) # total
        #print(acorC_df)

        tau_df = rexEff.compute_autocorrelation_time(max_lag) # total autocorrelation time
        print(tau_df)
    #endregion  

#endregion --------------------------------------------------------------------

if __name__ == "__main__":

    #region Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Directory with data files')
    parser.add_argument('--inFNRoots', nargs='+', required=True, help='Robosample processed output file names')
    parser.add_argument('--topology', help='Topology file')
    parser.add_argument('--cols', nargs='+', help='Columns to be read')
    parser.add_argument('--filterBy', nargs='*', default=[], help='Optional filters in the format col=value (e.g. wIx=0)')
    parser.add_argument('--useCache', action='store_true', help='Load data from cache file if it exists')
    parser.add_argument('--writeCache', action='store_true', help='Write new cache file (fails if file exists)')
    parser.add_argument('--cacheFile', default='rex_cache.pkl', help='Path to cache file')

    parser.add_argument('--restDir', help='Directory where restart files are put')
    parser.add_argument('--dry', action='store_true', default=False, help="No actions, just print.")

    parser.add_argument('--basicChecks', nargs='+', default=[], help='Checks: <col>')
    parser.add_argument('--checks', nargs='+', default=[], help='Checks: acceptance, dpe')

    parser.add_argument('--figures', nargs='+', default=[], type=str, help='Figures: tau_ac, potentialEnergyDistrib')
    args = parser.parse_args()
    #endregion

    main(args)

#region temp docs
# Example usage:
# python ~/git6/REBAS/rebas.py --dir prod/trpch/mulReplSalieri/ --inFNRoots out.3030500 --cols replicaIx thermoIx wIx acc --filterBy wIx=7 --checks acceptance --figures tau_ac
#
# Header description:
# "replicaIx" means replica,
# "thermoIx" means temperature index,
# "wIx" means what part of the molecule was simulated as a Gibbs block,
# "ts" means the timestep used to integrate,
# "mdsteps" how many steps was integrated for a Hamiltonian Monte Carlo trial
# "pe_o" means
# 
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
#endregion
