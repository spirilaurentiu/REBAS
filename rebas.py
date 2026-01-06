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
#matplotlib.use('Agg')

from rex_data import REXData
from rex_efficiency import RoboAnalysis, REXEfficiency
from rex_trajdata import REXTrajData


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
    # def __init__(self, dir=None, FNRoots=None, SELECTED_COLUMNS=None):
    #     self.dir = dir
    #     self.FNRoots = FNRoots
    #     self.SELECTED_COLUMNS = SELECTED_COLUMNS
    #     self.OUTPUT_DATA = False
    #     self.TRAJECTORY_DATA = False
    # #
    def __init__(self, dir=None, FNRoots=None, SELECTED_COLUMNS=None, topology="trpch/ligand.prmtop"):
        self.dir = dir
        self.FNRoots = FNRoots
        self.SELECTED_COLUMNS = SELECTED_COLUMNS
        self.topology = topology
        self.OUTPUT_DATA = False
        self.TRAJECTORY_DATA = False

    # Get seed and simulation type from filename. Determine if OUT or DCD
    def getSeedAndTypeFromFN(self, FN):
        """ Parse filename of type out.<7-digit seed>
        """

        seed, sim_type = -1, -1

        if FN.startswith("out."):
            match = re.match(r"out\.(\d{7})", FN)
            if not match: raise ValueError(f"Filename '{FN}' does not match expected format 'out.<7-digit-seed>'")

            seed = match.group(1)
            sim_type = seed[2]  # third digit (0-based index)

            self.OUTPUT_DATA = True

        elif FN.endswith(".dcd"):
            pattern = r"([A-Za-z0-9]+)_(\d{7})\.repl\d+\.dcd$"
            match = re.match(pattern, FN)
            if not match:
                raise ValueError(f"Filename '{FN}' does not match expected format'<name_of_the_molecule>_<7-digit-seed>.repl<index>.dcd'")

            print("match", match.group(1), match.group(2))

            molName = match.group(1)
            seed = match.group(2)
            sim_type = seed[2]   # third digit

            self.TRAJECTORY_DATA = True

        return seed, sim_type
    #

    # Read data from a single file
    def getDataFromFile(self, FN, seed, sim_type, burnin=0):
        """ Read data from file
        """
        rex = REXData(FN, self.SELECTED_COLUMNS)
        df = rex.get_out_dataframe().copy()
        df['seed'] = seed
        df['sim_type'] = sim_type

        # Remove burn-in rows
        if burnin > 0:
            df = df.iloc[burnin:].reset_index(drop=True)

        return df
    #

    # Read data from all files
    def getDataFromAllFiles(self, burnin=0):
        """ Grab all out files and read data from them
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

                df = self.getDataFromFile(filepath, seed, sim_type, burnin=burnin)
                all_data.append(df)

                print("done.", flush=True)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    #

    #
    # # Get trajectory data from a single trajectory file
    # def getTrajDataFromFile(self, FN, seed, sim_type):
    #     """ Load trajectory from filepath 
    #     Returns:
    #         MDTraj object
    #     """
    #     trajData = REXTrajData(FN, topology="trpch/ligand.prmtop")
    #     traj, meta = trajData.get_traj_observable()
    #     trajData.clear()
    #     return (traj, meta)
    # #
    # # Get trajectory data from all files
    # def getTrajDataFromAllFiles(self):
    #     """ Grab all dcd files and read data from them
    #     Returns:
    #         info and MDTraj objects
    #     """        
    #     trajectories = []
    #     metadata_rows = []
    #     for FNRoot in self.FNRoots:
    #         pattern = os.path.join(self.dir, FNRoot + "*")
    #         matches = glob.glob(pattern)
    #         if not matches:
    #             print(f"Warning: No files found matching {FNRoot}*", file=sys.stderr)
    #             continue
    #         for filepath in matches:
    #             print(f"Reading {filepath} ...", end=" ", flush=True)
    #             try:
    #                 seed, sim_type = self.getSeedAndTypeFromFN(os.path.basename(filepath))
    #             except ValueError as e:
    #                 print(f"[SKIP] Bad filename: {filepath} -> {e}", file=sys.stderr)
    #                 continue
    #             try:
    #                 traj, meta = self.getTrajDataFromFile(filepath, seed, sim_type)
    #             except Exception as e:
    #                 print(f"[FAIL] Error reading {filepath}: {e}", file=sys.stderr)
    #                 continue
    #             trajectories.append(traj)
    #             metadata_rows.append(meta)
    #             print("done.")
    #     metadata_df = pd.DataFrame(metadata_rows)
    #     return trajectories, metadata_df
    # #

    def getTrajDataFromFile(self, FN, seed, sim_type, observable_fn, *, frames=None, **obs_kwargs):
        trajData = REXTrajData(FN, topology=self.topology)

        obs, meta = trajData.get_traj_observable(
            observable_fn,
            frames=frames,
            **obs_kwargs
        )

        trajData.clear()

        # enrich meta with filename-derived info
        meta["seed"] = seed
        meta["sim_type"] = sim_type

        return obs, meta

    def getTrajDataFromAllFiles(self, observable_fn, *, frames=None, **obs_kwargs):
        trajectories = []
        metadata_rows = []

        for FNRoot in self.FNRoots:
            pattern = os.path.join(self.dir, FNRoot + "*")
            matches = glob.glob(pattern)

            if not matches:
                print(f"Warning: No files found matching {FNRoot}*", file=sys.stderr)
                continue

            for filepath in matches:
                print(f"Reading {filepath} ...", end=" ", flush=True)

                try:
                    seed, sim_type = self.getSeedAndTypeFromFN(os.path.basename(filepath))
                except ValueError as e:
                    print(f"[SKIP] Bad filename: {filepath} -> {e}", file=sys.stderr)
                    continue

                try:
                    obs, meta = self.getTrajDataFromFile(
                        filepath, seed, sim_type,
                        observable_fn,
                        frames=frames,
                        **obs_kwargs
                    )
                except Exception as e:
                    print(f"[FAIL] Error reading {filepath}: {e}", file=sys.stderr)
                    continue

                trajectories.append(obs)
                metadata_rows.append(meta)
                print("done.")

        metadata_df = pd.DataFrame(metadata_rows)
        return trajectories, metadata_df
    
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
def plot1D(df, col, save_path=None):
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

# Generic plotting function for 2D scatters
def plot2D(df, xcol, ycol, save_path=None):
    """ Plot values of two columns against each other in a scatter plot.
    Arguments:
        df : DataFrame
        xcol : column name for x-axis
        ycol : column name for y-axis
        save_path : optional, if given save plot to file
    """
    if xcol not in df.columns or ycol not in df.columns:
        raise ValueError(f"Columns '{xcol}' or '{ycol}' not found in DataFrame.")

    plt.figure()
    plt.scatter(df[xcol], df[ycol], alpha=0.7)
    plt.title(f"Scatter plot of {ycol} vs {xcol}")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
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
        "brown", "cyan", "purple", "yellow"#, "pink"
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



def stack_pad_nan(traj_list):
    max_len = max(len(x) for x in traj_list)
    arr = np.full((len(traj_list), max_len), np.nan, dtype=float)
    for i, x in enumerate(traj_list):
        x = np.asarray(x, dtype=float)
        arr[i, :len(x)] = x
    return arr, max_len

def running_mean_and_som(x):
    """
    Running mean and standard deviation of the mean (SOM).

    Parameters
    ----------
    x : 1D np.ndarray

    Returns
    -------
    mean : np.ndarray
        Running mean
    som : np.ndarray
        Running standard deviation of the mean
    """
    x = np.asarray(x, dtype=float)
    n = np.arange(1, len(x) + 1)

    mean = np.cumsum(x) / n

    # Running variance of the mean
    var = np.zeros_like(mean)
    for i in range(1, len(x)):
        var[i] = np.sum((x[:i+1] - mean[i])**2) / i

    som = np.sqrt(var)
    return mean, som


# ============ PANDAS DOCUMENTATION ============
# dataframe: two-dimensional, size-mutable, potentially heterogeneous tabular data
# grouped: pandas.core.groupby.DataFrameGroupBy object = lazy grouping object - essentially a recipe for how the DataFrame should be grouped
# group: a tuple (name, dataframe)


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

    OUTPUT_REQUIRED, TRAJECTORY_REQUIRED = False, True # flags
    FNManager = None # classes
    out_df, traj_df = None, None # pandas

    if OUTPUT_REQUIRED:

        GLOBAL_BURNIN = 1000

        #region Read output from all files
        if args.useCache and os.path.exists(args.outCacheFile):
            print(f"Loading data from cache: {args.outCacheFile}")
            out_df = pd.read_pickle(args.outCacheFile)
        else:
            FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols)
            out_df = FNManager.getDataFromAllFiles(burnin = GLOBAL_BURNIN)

            if args.writeCache:
                if os.path.exists(args.outCacheFile):
                    raise FileExistsError(f"Cache file '{args.outCacheFile}' already exists. Use a different name or delete it.")
                print(f"Writing data to cache: {args.outCacheFile}")
                out_df.to_pickle(args.outCacheFile)                

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

        #region Panda_Study
        # print("out_df:\n", out_df)
        # grouped = out_df.groupby(['replicaIx'])
        # print("\n\nout_df.groupby(['replicaIx']):\n")
        # for name, df in grouped:
        #     print("Group:", name)
        #     print(df)
        # print("\n\nout_df.info:\n", out_df.info())
        # print("\n\nout_df.index:\n", out_df.index)
        # print("\nout_df.columns:\n", out_df.columns)
        # print("\nout_df.dtypes:\n", out_df.dtypes)
        # print("out_df.axes:\n", out_df.axes)
        # print("\n\nout_df.keys():\n", out_df.keys())
        # print('\n\nout_df.get("pe_o"):\n', out_df.get("pe_o"))
        # print('\n\nout_df.get("pe_o").to_numpy():\n', out_df.get("pe_o").to_numpy())
        #endregion Panda_Study

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
                    plot1D(out_df, column, save_path=save_path)
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

            grouped = roboAna.df.groupby(['thermoIx', 'sim_type', 'seed'])

            for (thermoIx, sim_type, seed), group in grouped:
                plt.figure(figsize=(8, 4))
                # compute delta PE
                for exchangeDirection in [0, 1]:
                    delta_pe = (group['pe_n'] - group['pe_o'])[exchangeDirection:1000][::2]

                    plt.plot(delta_pe.values, marker='.', linestyle='-', alpha=0.7)

                    plt.title(f"ΔE = pe_n - pe_o (thermoIx={thermoIx}, sim_type={sim_type}, seed={seed})")
                    plt.xlabel("Index")
                    plt.ylabel("ΔE")
                    plt.grid(True)

                    plt.tight_layout()
                plt.show()

            #dpeHist_df = roboAna.delta_pe_histograms(bins=50)
            ##plot_histogram(dpeHist_df, save_path=f"check_dpe_hist.png")
            #plot_histogram(dpeHist_df)
        #endregion

        #region Paper figures: Validation
        if "pe_o" in args.figures:

            # roboAna = RoboAnalysis(out_df)
            # hist_df = roboAna.column_histograms('pe_o', bins=5)
            # print("hist_df pandas", pd.DataFrame(hist_df))
            # print("hist_df.values", hist_df.values())
            # plot_histogram(hist_df, save_path="x.png")
            # plot_histogram(hist_df)

            # consistent bins across all groups
            col_clean = out_df['pe_o'].dropna()
            lower_percentile = 0.2 #0.005
            upper_percentile = 98.0 #99.995
            nbins = 300

            vmin = np.percentile(col_clean, lower_percentile)
            vmax = np.percentile(col_clean, upper_percentile)
            bin_edges = np.linspace(vmin, vmax, nbins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # dictionary to hold histograms per (sim_type, thermoIx)
            histograms_by_type_thermo = {}

            # loop over sim_type and thermoIx
            for (sim_type, thermoIx), subdf_group in out_df.groupby(["sim_type", "thermoIx"]):
                # list to store densities across seeds
                seed_densities = []
                
                # loop over seeds for this sim_type and thermoIx
                for seed, subdf in subdf_group.groupby("seed"):
                    counts, _ = np.histogram(subdf["pe_o"], bins=bin_edges)
                    densities = counts / counts.sum() if counts.sum() > 0 else counts
                    seed_densities.append(densities)
                
                density_array = np.array(seed_densities)
                avg_density = np.mean(density_array, axis=0)
                std_density = np.std(density_array, axis=0)
                
                histograms_by_type_thermo[(sim_type, thermoIx)] = {
                    "avg_density": avg_density,
                    "std_density": std_density,
                    "bin_edges": bin_edges
                }
                    

            # ---- Plot ----
            plt.figure(figsize=(10, 6))
            colors = [
                "black", "maroon", "red", "orange", "yellow", "green", "cyan", "blue", "violet", "pink",
            ]

            for (sim_type, thermoIx), data in histograms_by_type_thermo.items():
                plt.errorbar(
                    bin_centers, 
                    data["avg_density"], 
                    yerr=data["std_density"],
                    color=colors[thermoIx % len(colors)],
                    #fmt="-o", 
                    capsize=3,
                    label=f"type {sim_type}, thermo {thermoIx}"
                )
            plt.xlabel("pe_o")
            plt.ylabel("Density ± std (across seeds)")
            plt.title("Histogram densities of pe_o by sim_type and thermoIx")
            plt.legend()
            plt.tight_layout()
            plt.savefig("peo_density_by_type_thermo.png", dpi=300)
            #plt.show()
            plt.close()

            plt.figure(figsize=(10, 6))
            for seed, subdf in subdf_group.groupby("seed"):
                data = subdf["pe_o"].to_numpy()
                plt.plot(data, label=str(seed)+str(thermoIx))
            plt.ylabel("pe_o")
            plt.title("Timesesries of pe_o by seed")
            plt.legend()
            plt.tight_layout()
            plt.savefig("peo_tseries_by_type_thermo.png", dpi=300)
            #plt.show()
            plt.close()

        #endregion

        #region Paper figures: Efficiency
        if "tau_ac" in args.figures:

            rexEff = REXEfficiency(out_df)

            grouped = out_df.groupby('replicaIx')
            num_replicas = len(grouped)

            # Calculate exchange rate
            burnin = 1024

            # exch_df = rexEff.calc_exchange_rates(burnin=burnin)
            # print("Exchange rates at burnin", burnin)
            # print(exch_df)

            # Calculate autocorrelation function
            max_lag = 100

            # C_k_t_df = rexEff.compute_autocorrelation(max_lag) # per replica
            # colors = ["black", "maroon", "red", "orange", "yellow", "green", "cyan", "blue", "violet",]
            # plt.figure(figsize=(10, 6))
            # plt.ylabel("C_k_t")
            # plt.title("C_k_t")
            # plt.tight_layout()
            # subdfIx = -1
            # for (seed, replicaIx), subdf in C_k_t_df.groupby(["seed", "replicaIx"]):
            #     subdfIx += 1
            #     C_k_t = subdf[("autocorrelation")].values
            #     plt.plot(C_k_t, color=colors[subdfIx // num_replicas], label=seed)
            # plt.legend()
            # plt.savefig("C_k_t.png", dpi=300)
            # plt.close()

            C_t_df = rexEff.compute_mean_autocorrelation(max_lag) # mean among the replicas

            colors = ["black", "maroon", "red", "orange", "yellow", "green", "cyan", "blue", "violet",]
            plt.figure(figsize=(10, 6))
            plt.ylabel("C_t")
            plt.title("C_t")
            plt.tight_layout()
            subdfIx = -1
            for (seed), subdf in C_t_df.groupby([("seed")]):
                subdfIx += 1
                C_t = subdf[("mean_autocorrelation")].values
                plt.plot(C_t, color=colors[subdfIx], label=seed)
            plt.legend()
            plt.savefig("C_t.png", dpi=300)
            plt.show()
            plt.close()




            tau_df = rexEff.compute_autocorrelation_time(max_lag) # total autocorrelation time
            print(tau_df)

            # tau2_df = rexEff.compute_tau2() # relaxation time
            # print(tau2_df)
            # tau_p_df = rexEff.compute_tau_p() # MFPT
            # print(tau_p_df)
        #endregion






    if TRAJECTORY_REQUIRED:

        #region Paper figures: RMSD
        # if "rmsd" in args.figures:
        #     rmsd_records = []   # <-- needed
        #     for i, traj in enumerate(traj_observables):
        #         traj_sel = traj # No atom selection by default
        #         reference = traj_sel[0] # Reference = first frame
        #         rmsd_values = md.rmsd(traj_sel, reference) # Compute RMSD (nm)
        #         meta_row = traj_metadata_df.iloc[i].to_dict() # Store results
        #         for frame_idx, value in enumerate(rmsd_values):
        #             rmsd_records.append({
        #                 "traj_index": i,
        #                 "frame": frame_idx,
        #                 "RMSD": value,
        #                 **meta_row
        #             })
        #     # Convert to DataFrame
        #     rmsd_df = pd.DataFrame(rmsd_records)
        #     plt.figure(figsize=(10, 6))
        #     plt.xlabel("Frame")
        #     plt.ylabel("RMSD (nm)")
        #     plt.title("RMSD Over Trajectories")            
        #     for traj_index, group in rmsd_df.groupby("traj_index"):
        #     #for seed, group in rmsd_df.groupby("seed"):            
        #         plt.plot(group["frame"], group["RMSD"], label=f"Traj {traj_index}")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.tight_layout()
        #     #plt.show()
        #     plt.savefig("rmsd_plot.png")
        #     plt.close()
        #  if "obs" in args.figures:
        #     print("traj_observables")
        #     print(traj_observables)
        #     print("traj_metadata_df")            
        #     print(traj_metadata_df)
        #     # Plot
        #     plt.figure(figsize=(10, 6))
        #     plt.xlabel("(frames)")
        #     plt.ylabel("Observable")
        #     plt.title("Observable")
        #     utilObj = REXFNManager()
        #     for i, traj in enumerate(traj_observables):
        #         FN = (traj_metadata_df.iloc[i])["filepath"]
        #         seed, sim_type = utilObj.getSeedAndTypeFromFN(os.path.basename(FN))
        #         Color = "black"
        #         if int(seed) > 3019999:
        #             Color = "red"
        #         plt.plot(traj, color=Color, label=f"Traj {FN}")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.tight_layout()
        #     # plt.show()
        #     plt.savefig("traj_obs.png")
        #     plt.close() 
        # #endregion

        if "obs_mom" in args.figures:
        
            GLOBAL_BURNIN = 0

            # pick your frame slice once:
            frames = slice(GLOBAL_BURNIN, 80000)   # or slice(GLOBAL_BURNIN, None)

            # define any observable you want:
            def dist_8_298(traj, a1=8, a2=298):
                return md.compute_distances(traj, [[a1, a2]]).ravel()

            FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols)

            (traj_observables, traj_metadata_df) = FNManager.getTrajDataFromAllFiles(
                dist_8_298,
                frames=frames,
                a1=8, a2=298,   # optional; overrides defaults
            )

            print(traj_metadata_df)

            print("traj_observables")
            print(traj_observables)
            print("traj_metadata_df")            
            print(traj_metadata_df)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.xlabel("(frames)")
            plt.ylabel("Observable")
            plt.title("Observable")

            utilObj = REXFNManager()

            listOfType1Sims = []
            listOfType2Sims = []

            utilObj = REXFNManager()

            for i, traj in enumerate(traj_observables):
                FN = traj_metadata_df.iloc[i]["filepath"]
                seed, sim_type = utilObj.getSeedAndTypeFromFN(os.path.basename(FN))

                if int(sim_type) == 1:
                    listOfType1Sims.append(traj)
                else:
                    listOfType2Sims.append(traj)

            type1_arr, L1 = stack_pad_nan(listOfType1Sims)
            type2_arr, L2 = stack_pad_nan(listOfType2Sims)

            print("type1_arr, L1", type1_arr, L1)
            print("type2_arr, L2", type2_arr, L2)

            type1_mean = np.nanmean(type1_arr, axis=0)
            type2_mean = np.nanmean(type2_arr, axis=0)

            # type1_mom = np.cumsum(type1_mean) / np.arange(1, len(type1_mean) + 1)
            # type2_mom = np.cumsum(type2_mean) / np.arange(1, len(type2_mean) + 1)

            type1_mom, type1_som = running_mean_and_som(type1_mean)
            type2_mom, type2_som = running_mean_and_som(type2_mean)

            print("type1_mean", type1_mean)
            print("type2_mean", type2_mean)



            x = np.arange(len(type1_mom))
            somStride = 1000

            # plt.plot(type1_mom, label="Type 1 mean", color="black")
            # plt.plot(type2_mom, label="Type 2 mean", color="red")
            # plt.errorbar(
            #     x[::somStride],
            #     type1_mom[::somStride],
            #     yerr=type1_som[::somStride],
            #     fmt='-',
            #     color='black',
            #     label='Type 1 MOM'
            # )
            # plt.errorbar(
            #     x[::somStride] + (somStride/8),
            #     type2_mom[::somStride],
            #     yerr=type2_som[::somStride],
            #     fmt='-',
            #     color='red',
            #     label='Type 2 MOM'
            # )

            plt.plot(type1_mom, label="Type 1 MOM", color="black")
            plt.fill_between(
                x,
                (type1_mom - (type1_som/2)),
                (type1_mom + (type1_som/2)),
                color="black",
                alpha=0.3
            )
            plt.plot(type2_mom, label="Type 2 MOM", color="red")
            plt.fill_between(
                x,
                (type2_mom - (type2_som/2)),
                (type2_mom + (type2_som/2)),
                color="red",
                alpha=0.3
            )

            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.show()
            plt.savefig("traj_obs.png")
            plt.close() 

        #region Paper figures: trajectory autocorrelation
        if "traj_eeac" in args.figures:
            
            rex_eff = REXEfficiency(
                out_df=None,
                trajectories=traj_observables,
                traj_metadata_df=traj_metadata_df
            )

            print(rex_eff.trajectories)
            print(rex_eff.traj_metadata_df)

            # Compute per-trajectory autocorrelation functions
            acf_list, tau_list, meta_list = rex_eff.compute_end_to_end_autocorr(
                burnin=GLOBAL_BURNIN,
                max_lag=3000,          # or specify an int
                dt=1.0,                # frame time (adjust if needed)
                average_over_trajs=False
            )

            # Plot
            plt.figure(figsize=(10, 6))
            plt.xlabel("Lag (frames)")
            plt.ylabel("Autocorrelation")
            plt.title("End-to-End Distance ACF Over Trajectories")

            utilObj = REXFNManager()

            for i, acf in enumerate(acf_list):
                FN = (meta_list[i])["filepath"]
                seed, sim_type = utilObj.getSeedAndTypeFromFN(os.path.basename(FN))
                Color = "black"
                if int(seed) > 3019999:
                #if False:
                    Color = "red"
                plt.plot(acf, color=Color, label=f"Traj {FN} (τ={tau_list[i]:.2f})")

            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.show()
            plt.savefig("traj_acf_plot.png")
            plt.close()            

        #endregion

    #region Restart: write restart files into self.dir/restDir/restDir.<seed>
    if (args.restDir):
        TRAJECTORY_REQUIRED = True
        FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols)
        FNManager.write_restarts_from_trajectories(args.restDir, args.topology, dry=args.dry)
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
    parser.add_argument('--outCacheFile', default='rex_cache.pkl', help='Path to cache file')

    parser.add_argument('--trajCacheFile', default='rex_cache.pkl', help='Path to cache file')

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
