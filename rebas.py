# main.py
import argparse
import os
import re
import sys
import glob
from turtle import title
import pandas as pd
from batana import BATStats
import numpy as np

import scipy.stats
from scipy.signal import find_peaks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mystats import LS_Statistics

from rex_data import REXData
from rex_efficiency import RoboAnalysis, REXEfficiency
from rex_trajdata import REXTrajData

from rex_fn_manager import REXFNManager

import MDAnalysis as mda
#from MDAnalysis.coordinates.TRJ import Restart
import mdtraj as md

# -----------------------------------------------------------------------------
#                      Utility Functions
#region Utility Functions ----------------------------------------------------

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
        for value in vals:
            value = value.strip()
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # leave as string
            parsed_vals.append(value)

        # Store single values as scalar, multi as list
        filter_dict[key] = parsed_vals if len(parsed_vals) > 1 else parsed_vals[0]

    return filter_dict
#

# Generic plotting function for timelines
def plotChecks1D(df, col, save_path=None):
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


# Generic plotting function for 2D scatter plots
def plotScatter(
    X=None,
    Y=None,
    title="title",
    xlabel="x",
    ylabel="y",
    xlim=None,
    ylim=None,
    labels=None,
    legend=True,
    colors=None,
    save_path=None,
    s=1, linestyle="None", marker='.', alpha=0.7):
    """ Plot values of two columns against each other in a line plot.
    Arguments:
        X : array-like for x-axis
        Y : array-like for y-axis
    """

    #region Input validation and processing
    if X is None or Y is None:
        raise ValueError("Both X and Y must be provided for scatter plot.")
    if len(X) != len(Y):
        raise ValueError(f"X and Y must have the same number of series. Got {len(X)} and {len(Y)}.")

    # X should be a list of arrays
    if not isinstance(X, list):
        X = list(X)
    for ix, X_series in enumerate(X):
        if not isinstance(X_series, np.ndarray):
            X[ix] = np.asarray(X_series, dtype=float)

    # Y should be a list of arrays
    if not isinstance(Y, list):
        Y = list(Y)
    for ix, Y_series in enumerate(Y):
        if not isinstance(Y_series, np.ndarray):
            Y[ix] = np.asarray(Y_series, dtype=float)

    # Labels should be a list of strings
    if not isinstance(labels, list):
        labels = [None] * len(Y)
    elif len(labels) != len(Y):
        raise ValueError(f"Length of labels must match number of Y series. Got {len(labels)} and {len(Y)}.")

    if not isinstance(colors, list):
        colors = [None] * len(Y)
    elif len(colors) != len(Y):
        raise ValueError(f"Length of colors must match number of Y series. Got {len(colors)} and {len(Y)}.")
    #endregion
    
    # Plot generics
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plt plot or errorbar
    for ix, Y_series in enumerate(Y):
        plt.scatter(X[ix], Y_series,
            label=labels[ix], s=s, alpha=alpha,
            color=colors[ix] if colors else None)
    
    # Plot finishing touches
    if legend == True:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        pass
        #plt.show()
#

# Generic plotting function for 1D line plots
def plot1D(Y,
           X=None,
           Yerr=None,
           Yerr_every=1,
           title="title",
           xlabel="x",
           ylabel="y",
           ylim=None,
           labels=None,
           legend=True,
           colors=None,
           save_path=None,
           linestyle="None", marker='.', alpha=0.7):
    """ Plot values of two columns against each other in a line plot.
    Arguments:
        X : array-like for x-axis
        Y : array-like for y-axis
    """

    # Y should be a list of arrays
    if not isinstance(Y, list):
        Y = list(Y)
    for ix, Y_series in enumerate(Y):
        Y[ix] = np.asarray(Y_series, dtype=float)

    # Yerr should be a list of arrays
    if Yerr is not None:
        if not isinstance(Yerr, list):
            Yerr = list(Yerr)
        for ix, Yerr_series in enumerate(Yerr):
            Yerr[ix] = np.asarray(Yerr_series, dtype=float)

    # If X is None, create default X as range for each Y series
    if X is None:
        X = []
        for ix, Y_series in enumerate(Y):
            X.append(np.arange(len(Y_series)))

    # Labels should be a list of strings
    if not isinstance(labels, list):
        labels = [None] * len(Y)
    
    # Plot generics
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plt plot or errorbar
    for ix, Y_series in enumerate(Y):
        if Yerr is None:
            plt.plot(X[ix], Y_series, label=labels[ix],
                    color=colors[ix] if colors else None)
        else:
            plt.errorbar(X[ix], Y_series, yerr=Yerr[ix], errorevery=Yerr_every,
                         label=labels[ix], linestyle=linestyle, marker=marker, alpha=alpha,
                         color=colors[ix] if colors else None)
    
    # Plot finishing touches
    if legend == True:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if ylim is not None:
        plt.ylim(ylim)

    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        pass
        #plt.show()
#

# Generic plotting function for 2D scatters
def plot2DWithErr(df, xcol, ycol, save_path=None):
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

def stack_pad_nan(traj_obs_list):
    max_len = max(len(x) for x in traj_obs_list)
    arr = np.full((len(traj_obs_list), max_len), np.nan, dtype=float)
    for i, x in enumerate(traj_obs_list):
        x = np.asarray(x, dtype=float)
        arr[i, :len(x)] = x
    return arr, max_len

def colorByType(sim_type):
    if int(sim_type) == 1:
        return "black"
    elif int(sim_type) == 3:
        return "red"
    else:
        return "grey"
    
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

    OUTPUT_REQUIRED, TRAJECTORY_REQUIRED = False, False # default flags
    if args.inFNRoots[0][0:3] == 'out':
        OUTPUT_REQUIRED = True
    else:
        TRAJECTORY_REQUIRED = True
    
    FNManager = None # classes
    out_df, traj_df = None, None # pandas

    stats = LS_Statistics()

    if OUTPUT_REQUIRED:

        GLOBAL_BURNIN = 5000

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
        #endregion

        # out_df.info()
        # # #print("outf_df info:\n", out_df.info())

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
        # exit(2)
        #endregion Panda_Study

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
                    plotChecks1D(out_df, column, save_path=save_path)
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

            nofThermodynamicStates = roboAna.df['thermoIx'].nunique()

            direction_color = {0: 'blue', 1: 'red'}
            thermo_cmap = plt.get_cmap("tab20")
            #thermo_colors = [thermo_cmap(i) for i in np.linspace(0, 1, nofThermodynamicStates)]
            thermo_colors = ['blue', 'red']

            plt.figure(figsize=(8, 4))
            burnin_local = 0
            stop_at = 1000
            stride = 2
            for (thermoIx, sim_type, seed), group in grouped:
                # compute delta PE
                #for start_at in [0, 1]:
                #for start_at in [0]: # ala1
                #for start_at in [1]: # ethane
                for start_at in [0, 1]: # trpch
                    delta_pe = (group['pe_n'] - group['pe_o'])[(start_at + burnin_local) : stop_at][::stride]

                    plt.plot(delta_pe.values, marker='.', linestyle='-', alpha=0.7,
                             label=f"ΔPE thermoIx {thermoIx}",
                             color=thermo_colors[thermoIx%2])

                    #plt.title(f"ΔE = pe_n - pe_o (thermoIx={thermoIx}, sim_type={sim_type}, seed={seed})")
                    plt.title(f"ΔE")
                    plt.xlabel("Index")
                    plt.ylabel("ΔE")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
            #plt.show()

            #dpeHist_df = roboAna.delta_pe_histograms(bins=50)
            ##plot_histogram(dpeHist_df, save_path=f"check_dpe_hist.png")
            #plot_histogram(dpeHist_df)

            plt.figure(figsize=(8, 4))
            burnin_local = 0
            stop_at = 1000
            stride = 2
            for (thermoIx, sim_type, seed), group in grouped:
                
                for start_at in [0, 1]: # trpch

                    JDetLog = group['JDetLog'][(start_at + burnin_local) : stop_at][::stride]

                    plt.plot(JDetLog.values, marker='.', linestyle='-', alpha=0.7,
                             label=f"JDetLog thermoIx {thermoIx}",
                             color=thermo_colors[thermoIx%2])  
                    plt.title(f"JDetLog")
                    plt.xlabel("Index")
                    plt.ylabel("JDetLog")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
            plt.show()

        #endregion

        #region Paper figures: pe_o
        argStr = "pe_o"
        obsStr = "PE"
        if argStr in args.figures:

            observables = []
            observables_meta = []
            ix = -1
            for (sim_type, seed), subdf_group in out_df.groupby(["sim_type", "seed"]):
                ix += 1
                print(f"Processing sim_type={sim_type}, seed={seed} (group {ix})")
                #print(subdf_group)
                observables.append(subdf_group[argStr].to_numpy())
                observables_meta.append({
                    "sim_type": sim_type,
                    "seed": seed,})

            # Get a trimmed version cut at min length
            min_num_frames = min(len(Y) for Y in observables)
            max_num_frames = max(len(Y) for Y in observables)
            min_glob = min(Y.min() for Y in observables)
            max_glob = max(Y.max() for Y in observables)
            obs_list_trimmed = np.array([Y[:min_num_frames] for Y in observables])

            cumMean_list = []
            cumSom_list = []
            for ix, all_obss in enumerate(obs_list_trimmed):
                all_cummean, cumSom = cum_scum(all_obss)
                cumMean_list.append(all_cummean)
                cumSom_list.append(cumSom)

            # Get ensemble means and stds across trajectories
            type1_obs = []
            type3_obs = []
            for ix, all_obss in enumerate(obs_list_trimmed):
                sim_type = observables_meta[ix]["sim_type"]
                if int(sim_type) == 1:
                    type1_obs.append(all_obss)
                elif int(sim_type) == 3:
                    type3_obs.append(all_obss)
                else:
                    sys.exit(f"Unknown sim_type {sim_type} encountered.")

            type1_ens_hists = ensemble_histogram_plus(
                type1_obs,
                density=True,
                bins=50,
                obs_range=(min_glob, max_glob)
            )
            
            type3_ens_hists = ensemble_histogram_plus(
                type3_obs,
                density=True,
                bins=50,
                obs_range=(min_glob, max_glob)
            )
            #print("type1_bin_centers", type1_ens_results["bin_centers"])
            #print("type3_bin_centers", type3_ens_results["bin_centers"])
            assert np.allclose(type1_ens_hists["bin_centers"], type3_ens_hists["bin_centers"]), "Bin centers do not match between types."

            # Get autocorrelation functions
            acf_list = []
            for ix, all_obss in enumerate(obs_list_trimmed):
                acf = normalized_autocorrelation(all_obss, max_lag=min_num_frames - 1)
                acf_list.append(acf)

            PRINT__, PLOT__ = True, True

            if PRINT__:
                print("observables_meta", observables_meta)
                print("observables", observables)
                print("Careful: trimmed to min length:", min_num_frames, ". Max length:", max_num_frames)
            if PLOT__:
                plot1D(
                    Y=obs_list_trimmed,
                    title=obsStr,
                    xlabel="X",
                    ylabel=obsStr,
                    labels=[f"{observables_meta[ix]['seed']} type {observables_meta[ix]['sim_type']}" \
                            for ix in range(len(obs_list_trimmed))],
                    colors=[colorByType(observables_meta[ix]['sim_type']) \
                            for ix in range(len(obs_list_trimmed))]
                )

            if PRINT__:
                print("cumMean_list", cumMean_list)
            if PLOT__:            
                plot1D(
                    Y=cumMean_list,
                    title=obsStr + " cumulative mean",
                    xlabel="X",
                    ylabel=obsStr + " cumulative mean",
                    labels=[f"{observables_meta[ix]['seed']} type {observables_meta[ix]['sim_type']}" \
                            for ix in range(len(cumMean_list))],
                    colors=[colorByType(observables_meta[ix]['sim_type']) \
                            for ix in range(len(cumMean_list))]
                )

            if PRINT__:
                print("type1 stats (mean, std, entropy, num_modes):")
                print(type1_ens_hists["entropy"], type1_ens_hists["num_modes"])
                print("type3 stats (mean, std, entropy, num_modes):")
                print(type3_ens_hists["entropy"], type3_ens_hists["num_modes"])
            if PLOT__:
                plot1D(
                    Y=[type1_ens_hists["mean"], type3_ens_hists["mean"]],
                    Yerr=[type1_ens_hists["std"], type3_ens_hists["std"]],
                    X=[type1_ens_hists["bin_centers"], type3_ens_hists["bin_centers"]],
                    title=obsStr + " distribution",
                    xlabel="X",
                    ylabel=obsStr + " probability density",
                    labels=["type 1", "type 3"],
                    colors=[colorByType(1), colorByType(3)]
                )

            if PRINT__:
                pass
            if PLOT__:
                plot1D(
                    Y=acf_list,
                    title=obsStr + " ACF",
                    xlabel="Lag",
                    ylabel="ACF",
                    labels=[f"{observables_meta[ix]['seed']} type {observables_meta[ix]['sim_type']}" \
                            for ix in range(len(acf_list))],
                    colors=[colorByType(observables_meta[ix]['sim_type']) \
                            for ix in range(len(acf_list))]
                )

            plt.show()
            #plt.close()

        #endregion

        #region
        if "rex_eff" in args.figures:

            observables = []
            observables_meta = []
            nof_type1 = 0
            nof_type3 = 0

            repl_exxrs = []
            obs_stds = []
            obs_vars = []
            obs_statisticalEnergys = []
            obs_skews = []

            ACF_rhos = []
            ACF_rhos_mean_type1 = []
            ACF_rhos_mean_type3 = []
            tau_ac_type1, tau_ac_type3 = 0, 0
            obs_taus = []
            obs_ESSs = []

            PRINT__, PLOT__ = False, False

            ix = -1
            # We iterate through each group (replica) one by one
            for (sim_type, seed, replicaIx), subdf_group in out_df.groupby(["sim_type", "seed", "replicaIx"]):
                ix += 1

                if sim_type == "1":
                    nof_type1 += 1
                elif sim_type == "3":
                    nof_type3 += 1
                #print(f"Processing sim_type={sim_type}, seed={seed}, replicaIx={replicaIx} (group {ix})")
                
                # Extract the data for THIS specific replica
                obs_data = subdf_group["thermoIx"].to_numpy()

                #print(f"obs_data", obs_data.shape, obs_data)

                observables.append(obs_data)
                observables_meta.append({
                    "sim_type": sim_type,
                    "seed": seed,
                    "replicaIx": replicaIx
                })

                # --- Calculate Exchange Rate for this specific replica ---
                prevObsVal = -1
                exchanges = 0
                for obsVal in obs_data:
                    if prevObsVal != -1 and obsVal != prevObsVal:
                        exchanges += 1
                    prevObsVal = obsVal                
                exchange_rate = exchanges / len(obs_data)
                repl_exxrs.append(exchange_rate)
                
                # --- Standard deviation of the observable for this replica ---
                obs_std = np.std(obs_data)
                obs_var = np.var(obs_data)
                obs_skew = scipy.stats.skew(obs_data)
                obs_stds.append(obs_std)
                obs_vars.append(obs_var)
                obs_skews.append(obs_skew)

                # --- Calculate Autocorrelation Time (Integrated) ---
                #obs_tau = calculate_iat(obs_data)
                #print("obs_tau", obs_tau)
                (ACF_rho, obs_tau, ess) = stats.autocorr2_revised(obs_data, lag_fraction=0.5, max_lag=50000)
                #print("obs_tau", obs_tau)
                #(ACF_rho, obs_tau, ess) = stats.autocorr3_revised(obs_data, lag_fraction=0.5, max_lag=50000)
                #print("obs_tau", obs_tau)

                ACF_rhos.append(ACF_rho)
                obs_taus.append(obs_tau)
                
                obs_statisticalEnergy = obs_std / (obs_tau)
                obs_statisticalEnergy *= 1000  # scale to make more readable
                obs_statisticalEnergy = obs_statisticalEnergy**2
                obs_statisticalEnergys.append(obs_statisticalEnergy)

                obs_ESSs.append(ess)

                # --- Plotting ---
                PLOT__ = False
                if PLOT__:
                    plot1D(
                        Y=[obs_data], 
                        X=None,
                        ylim=(0, 14),
                        title=f"Replica Trajectory: {sim_type} | Seed {seed} | Rep {replicaIx}\n:exx {exchange_rate:.3f}",
                        xlabel="Frame",
                        ylabel="thermoIx (State)",
                        labels=[f"Rep {replicaIx}"],
                        colors=[colorByType(sim_type)],
                        #save_path=f"rex_eff_type_{sim_type}_seed_{seed}_rep_{replicaIx}.png"
                    )
                
                # Show/Close the specific figure for this replica before moving to the next
                plt.show() 
                plt.close()

                #matplotlib.use('Agg')        
                #plt.savefig("thermoIx_by_replica.png")

            # Get a trimmed version cut at min length
            min_num_frames = min(len(Y) for Y in observables)
            max_num_frames = max(len(Y) for Y in observables)
            min_glob = min(Y.min() for Y in observables)
            max_glob = max(Y.max() for Y in observables)
            obs_list_trimmed = np.array([Y[:min_num_frames] for Y in observables])
            #print(obs_list_trimmed.shape)

            cumMean_list = []
            cumSom_list = []
            for ix, all_obss in enumerate(obs_list_trimmed):
                all_cummean, cumSom = stats.cum_scum(all_obss)
                cumMean_list.append(all_cummean)
                cumSom_list.append(cumSom)

            ACF_min = min(len(acf) for acf in ACF_rhos)
            ACF_rhos = np.array([acf[:ACF_min] for acf in ACF_rhos])

            ACF_rhos_mean_type1 = np.zeros(ACF_min, dtype=float)
            ACF_rhos_mean_type3 = np.zeros(ACF_min, dtype=float)

            for ix, acf in enumerate(ACF_rhos):
                sim_type = observables_meta[ix]['sim_type']
                if int(sim_type) == 1:
                    ACF_rhos_mean_type1 += acf
                elif int(sim_type) == 3:
                    ACF_rhos_mean_type3 += acf

            if nof_type1 > 0:
                ACF_rhos_mean_type1 /= nof_type1
            if nof_type3 > 0:
                ACF_rhos_mean_type3 /= nof_type3

            tau_ac_type1 = stats.getTau(ACF_rhos_mean_type1)
            tau_ac_type3 = stats.getTau(ACF_rhos_mean_type3)

            tau_ac_Hai_type1 = stats.getTau_ac(ACF_rhos_mean_type1)
            tau_ac_Hai_type3 = stats.getTau_ac(ACF_rhos_mean_type3)

            PRINT__, PLOT__ = False, False
            if PLOT__:
                plot1D(
                    Y=obs_list_trimmed,
                    title="thermoIx trajectories (trimmed to min length)",
                    xlabel="Frame Index",
                    ylabel="thermoIx (State)",
                    labels=[f"Seed {observables_meta[ix]['seed']} Type {observables_meta[ix]['sim_type']} Rep {observables_meta[ix]['replicaIx']}" \
                            for ix in range(len(obs_list_trimmed))],
                    colors=[colorByType(observables_meta[ix]['sim_type']) \
                            for ix in range(len(obs_list_trimmed))],
                    ylim=(0, 13),
                    #save_path="thermoIx_trajectories.png",
                    linestyle="None", marker='.', alpha=0.7,
                    legend=False
                )
                plt.show() 
                plt.close()

            PRINT__, PLOT__ = True, False
            if PRINT__:
                for ix, meta in enumerate(obs_list_trimmed):
                    sim_type = observables_meta[ix]['sim_type']
                    seed = observables_meta[ix]['seed']
                    replicaIx = observables_meta[ix]['replicaIx']
                    print(f"sim_type={sim_type}, seed={seed},replicaIx={replicaIx}," + 
                          f" exxs: {repl_exxrs[ix]:.3f} stds: {obs_stds[ix]:.3f} skew: {obs_skews[ix]:.3f}," + 
                          f" tau_k: {obs_taus[ix]:.3f} statEnergy: {obs_statisticalEnergys[ix]:.9f}")
                    
                print(f"Mean tau_ac type 1: {tau_ac_type1:.3f}, Mean tau_ac type 3: {tau_ac_type3:.3f}")
                print(f"Mean tau_ac_Hai type 1: {tau_ac_Hai_type1:.3f}, Mean tau_ac_Hai type 3: {tau_ac_Hai_type3:.3f}")

                if PLOT__:
                    plot1D(
                        Y=cumSom_list,
                        title=obsStr + " cumulative running std",
                        xlabel="Frame",
                        ylabel=obsStr + " cumulative running std",
                        #labels=[f"Seed {observables_meta[ix]['seed']} Type {observables_meta[ix]['sim_type']}" \
                        #        for ix in range(len(cumSom_list))],
                        legend=False,
                        colors=[colorByType(observables_meta[ix]['sim_type']) \
                                for ix in range(len(cumSom_list))]
                    )            
                    plt.show()
                    plt.close()

            PRINT__, PLOT__ = False, True
            if PRINT__:
                print("ACF_rhos", ACF_rhos)
                print("obs_taus", obs_taus)
                print("obs_ESSs", obs_ESSs)

            if PLOT__:
                # plot1D(
                #     Y=ACF_rhos,
                #     title="Autocorrelation Functions (ACF) by Replica",
                #     xlabel="Lag",
                #     ylabel="ACF",
                #     labels=[f"Seed {observables_meta[ix]['seed']} Type {observables_meta[ix]['sim_type']} Rep {observables_meta[ix]['replicaIx']}" \
                #             for ix in range(len(ACF_rhos))],
                #     colors=[colorByType(observables_meta[ix]['sim_type']) \
                #             for ix in range(len(ACF_rhos))],
                #     legend=False
                # )

                plot1D(
                    Y=[ACF_rhos_mean_type1, ACF_rhos_mean_type3],
                    title="Mean Autocorrelation Function (ACF) Across Replicas (Nguyen & Minh, 2016 - eq 19)",
                    xlabel="Lag",
                    ylabel="Mean ACF",
                    labels=["Mean ACF Type 1", "Mean ACF Type 3"],
                    colors=["black", "red"],
                    legend=False
                )

                plt.show()
                plt.close()

        #endregion


        #region Paper figures: Efficiency
        if "tau_ac" in args.figures:

            rexEff = REXEfficiency(out_df)

            grouped = out_df.groupby('replicaIx')
            num_replicas = len(grouped)

            # Calculate exchange rate
            burnin_local = 1024

            # exch_df = rexEff.calc_exchange_rates(burnin=burnin)
            # print("Exchange rates at burnin", burnin)
            # print(exch_df)

            # Calculate autocorrelation function
            max_lag = 100

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

        #region Get filters if specified
        # Get filters if specified
        filters = {}
        if args.filterBy:
            filters = parse_filters(args.filterBy)
            for col, val in filters.items():
                print(f"filter: {col} = {val}")
        #endregion

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
        #endregion

        # Objective: Statistics from any observable from trajectories
        if "traj_stats" in args.figures:

            utilObj = REXFNManager()

            GLOBAL_BURNIN = 10000
            GLOBAL_END = None

            # pick your frame slice once:
            frames = slice(GLOBAL_BURNIN, GLOBAL_END)   # or slice(GLOBAL_BURNIN, None)

            FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols, topology=args.topology)
            FNManager.prepareTrajArraySize(filters=filters)

            #region Extractor functions for trajectory observables
            
            # Distances
            def distances(traj, pairs=[[8, 298], [100, 200]]):
                """ Calculate distances between multiple pairs. """
                result = md.compute_distances(traj, pairs)
                return result.T

            # Dihedral
            def dihedral_a1_a2_a3_a4(traj, a1=4, a2=6, a3=8, a4=14):
                """ Calculate dihedral angle defined by four atoms across all frames. """
                result = md.compute_dihedrals(traj, [[a1, a2, a3, a4]])

                return result.T

            def dihedral_adj_a1_a2_a3_a4_a5(traj, a1=4, a2=6, a3=8, a4=14, a5=16):
                dihedrals = md.compute_dihedrals(traj, [[a1, a2, a3, a4], [a2, a3, a4, a5]])
                return (1 - np.cos(dihedrals[:, 0] - dihedrals[:, 1]))

            def quaternion_multiply(q1, q2):
                """Vectorized quaternion multiplication."""
                w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
                w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
                
                res = np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2
                ]).T
                return res
            
            def dihedral_quat_a1_a2_a3_a4_a5(traj, a1=4, a2=6, a3=8, a4=14, a5=16):
                """ """
                phi = md.compute_dihedrals(traj, [[a1, a2, a3, a4]])
                psi = md.compute_dihedrals(traj, [[a2, a3, a4, a5]])
                q_phi = np.stack([np.cos(phi/2), np.sin(phi/2), np.zeros_like(phi), np.zeros_like(phi)], axis=-1)
                q_psi = np.stack([np.cos(psi/2), np.zeros_like(psi), np.sin(psi/2), np.zeros_like(psi)], axis=-1)
                #combined_q = quaternion_multiply(q_phi, q_psi)
                inner_q = np.dot(q_phi, q_psi)
    
                return inner_q

            def dihedral_phi_psi(traj, phi_psi="psi", resid=0):
                """  Calculate phi or psi for a given residue index. """
                mdtraj_result = None
                if phi_psi == "phi":
                    mdtraj_result = md.compute_phi(traj, periodic=False)
                elif phi_psi == "psi":
                    mdtraj_result = md.compute_psi(traj, periodic=False)
                else:
                    raise ValueError(f"Invalid phi_psi value: {phi_psi}. Must be 'phi' or 'psi'.")
                
                atom_indices = mdtraj_result[0] # shape (n_residues, 4)
                #print("atom_indices", atom_indices[resid, :])

                torsions_all = mdtraj_result[1] # shape (n_frames, n_residues)
                
                # Validate residue index
                if resid >= torsions_all.shape[1] or resid < 0:
                    raise ValueError(f"Invalid resid {resid}. Must be between 0 and {torsions_all.shape[1]-1}")
                
                torsions = torsions_all[:, resid].ravel()
                #torsions = np.where(torsions < -2.0, (torsions + 2.0) + (torsions * np.pi), torsions)
                #print(f"First 10 {phi_psi} values for residue {resid}: {torsions[:10]} radians")

                from batana import BATStats
                batStats = BATStats()
                torsions_mean = batStats.dihedralMean(torsions)
                torsions_var = batStats.dihedralVar(torsions) # technically a dimensionless ratio between 0 and 1. It isn't squared radians.
                torsions_std = batStats.dihedralStd(torsions)
                print(f"\nMean {phi_psi} for residue {resid}: {torsions_mean:.3f} radians ({np.rad2deg(torsions_mean):.1f} degrees)")
                #print(f"Var {phi_psi} for residue {resid}: {torsions_var:.3f} radians ({np.rad2deg(torsions_var):.1f} degrees)")
                print(f"Std {phi_psi} for residue {resid}: {torsions_std:.3f} radians ({np.rad2deg(torsions_std):.1f} degrees)")

                return torsions

            ALA1_BASINS = {
                "C5":       {"phi_min": np.deg2rad(-180), "phi_max": np.deg2rad(-95), "psi_min": np.deg2rad(105 ), "psi_max": np.deg2rad(180)},
                "PPII":     {"phi_min": np.deg2rad(-96 ), "phi_max": np.deg2rad(-45), "psi_min": np.deg2rad(105 ), "psi_max": np.deg2rad(180)},
                "C7_eq":    {"phi_min": np.deg2rad(-96 ), "phi_max": np.deg2rad(-45), "psi_min": np.deg2rad(-25 ), "psi_max": np.deg2rad(104)},
                "alpha_eq": {"phi_min": np.deg2rad( 35 ), "phi_max": np.deg2rad( 85), "psi_min": np.deg2rad(-180), "psi_max": np.deg2rad(25)}
            }

            def ala_PMF_indicator(traj, a1=4, a2=6, a3=8, a4=14, a5=16):
                # 1. Compute dihedrals for all frames
                phi = md.compute_dihedrals(traj, [[a1, a2, a3, a4]]).ravel()
                psi = md.compute_dihedrals(traj, [[a2, a3, a4, a5]]).ravel()
                
                # 2. Define the Boolean Masks for each state
                # C5, PPII, and C7_eq all map to 0
                is_c5 = (phi >= np.deg2rad(-180)) & (phi <= np.deg2rad(-95)) & \
                        (psi >= np.deg2rad(105))  & (psi <= np.deg2rad(180))
                        
                is_ppii = (phi >= np.deg2rad(-96)) & (phi <= np.deg2rad(-45)) & \
                        (psi >= np.deg2rad(105)) & (psi <= np.deg2rad(180))
                        
                is_c7eq = (phi >= np.deg2rad(-96)) & (phi <= np.deg2rad(-45)) & \
                        (psi >= np.deg2rad(-25)) & (psi <= np.deg2rad(104))
                        
                # alpha_eq maps to 1
                is_alpha = (phi >= np.deg2rad(35)) & (phi <= np.deg2rad(85)) & \
                        (psi >= np.deg2rad(-180)) & (psi <= np.deg2rad(25))

                # 3. Combine "0" states
                state_zero_mask = is_c5 | is_ppii | is_c7eq
                
                # 4. Use np.select to assign values
                # Logic: If in state_zero_mask -> 0.0
                #        Else if in is_alpha -> 1.0
                #        Else (default) -> 0.5
                conditions = [state_zero_mask, is_alpha]
                choices = [0.0, 1.0]
                
                states = np.select(conditions, choices, default=0.5)
                
                return states

            TRPCH_BASINS = {
                "basin1": {"psi_min": -1.5, "psi_max": 0.5, "ee_dist_min": 0.0, "ee_dist_max": 1.27},
                "basin2": {"psi_min": -1.5, "psi_max": 0.5, "ee_dist_min": 1.27, "ee_dist_max": 5.0},
                "basin3": {"psi_min":  2.0, "psi_max": 3.0, "ee_dist_min": 0.0, "ee_dist_max": 1.27},
                "basin4": {"psi_min":  2.0, "psi_max": 3.0, "ee_dist_min": 1.27, "ee_dist_max": 5.0},
            }

            def trpch_PMF_indicator(traj, phi_psi="psi", resid=0):
                psi = dihedral_phi_psi(traj, phi_psi=phi_psi, resid=resid)
                ee_dist = dist_atom1_atom2(traj, a1=8, a2=298)

                psi_range1 = (psi >= -1.5) & (psi <= 0.5)
                psi_range2 = (psi >= 2.0) & (psi <= 3.0)
                dist_short = (ee_dist <= 1.27)
                dist_long  = (ee_dist > 1.27) & (ee_dist <= 5.0)

                is_basin1 = psi_range1 & dist_short
                is_basin2 = psi_range1 & dist_long
                is_basin3 = psi_range2 & dist_short
                is_basin4 = psi_range2 & dist_long

                conditions = [is_basin1, is_basin2, is_basin3, is_basin4]
                choices = [0.0, 0.25, 0.5, 0.75]
                states = np.select(conditions, choices, default=1.0)
                return states
            
            #endregion # extractors

            DO_GEOMETRY, DO_PCA = False, True

            if DO_GEOMETRY:

                # Get trajectory data 
                obs_name = distances.__name__
                obs_title = "End-to-End Distance"
                #obs_title = "Trp-Cage PMF Indicator"
                (observables, u_types, u_repeats, u_thermos) = FNManager.getTrajDataFromAllFiles(
                    distances,
                    filters=filters,
                    frames=frames,
                    pairs=[[8, 298], [100, 200]],
                    #phi_psi="psi",  # optional; only for dihedral_phi_psi
                    #resid=11,       # optional; only for dihedral_phi_psi
                )

                print("observables.shape", observables.shape)
                n_types, n_repeats, n_thermos, n_observables, n_frames = observables.shape
                # print("observables", observables)
                # print("u_types", u_types)
                # print("u_repeats", u_repeats)
                # print("u_thermos", u_thermos)

                PRINT__, PLOT__ = True, True
                if PRINT__:
                    print("Unique sim types:", u_types)
                    print("Unique repeats:", u_repeats)
                    print("Unique thermos:", u_thermos)
                if PLOT__:
                    plotFN = None
                    if args.useAgg:
                        plotFN = f"traj_{obs_name}.png"

                    simIx = 0
                    replicaIx = 0
                    obsIx = 0
                    Y_series = observables[:, simIx, replicaIx, obsIx,:]
                    print(Y_series.shape)
                    
                    plot1D(
                        Y = Y_series,
                        title=obs_title,
                        xlabel="Frame",
                        ylabel=obs_name,
                        labels=[f"Seed {u_repeats[ix]} Type {u_types[ix]} Thermo {u_thermos[ix]}" for ix in range(len(observables))],
                        colors=[colorByType(u_types[ix]) for ix in range(len(observables))],
                        save_path=plotFN
                    )
                
                    if args.useAgg:
                        plt.savefig(plotFN)

                # Cumulative mean and std
                cummean_obs = np.full_like(observables, fill_value=np.nan)
                cumstd_obs = np.full_like(observables, fill_value=np.nan)

                for typeIx in range(n_types):
                    for repeatIx in range(n_repeats):
                        for thermoIx in range(n_thermos):
                            for obsIx in range(n_observables):
                                obs = observables[typeIx, repeatIx, thermoIx, obsIx,:]
                                cummean = np.cumsum(obs) / (np.arange(len(obs)) + 1)
                                cumstd = np.sqrt(np.cumsum((obs - cummean)**2) / (np.arange(len(obs)) + 1))
                                cummean_obs[typeIx, repeatIx, thermoIx, obsIx,:] = cummean
                                cumstd_obs[typeIx, repeatIx, thermoIx, obsIx,:] = cumstd

                PRINT__, PLOT__ = True, True
                if PRINT__:
                    print("cum_mean shape:", cummean_obs.shape)
                    print("cum_std shape:", cumstd_obs.shape)
                if PLOT__:
                    plot1D(
                        Y=cummean_obs[:, simIx, replicaIx, obsIx,:],
                        title=obs_title + " Cumulative Mean",
                        xlabel="Frame",
                        ylabel=obs_name + " Cumulative Mean",
                        labels=[f"Seed {u_repeats[ix]} Type {u_types[ix]} Thermo {u_thermos[ix]}" for ix in range(len(observables))],
                        colors=[colorByType(u_types[ix]) for ix in range(len(observables))],
                        save_path=f"traj_{obs_name}_cum_mean.png" if args.useAgg else None
                    )

                    plot1D(
                        Y=cumstd_obs[:, simIx, replicaIx, obsIx,:],
                        title=obs_title + " Cumulative Std",
                        xlabel="Frame",
                        ylabel=obs_name + " Cumulative Std",
                        labels=[f"Seed {u_repeats[ix]} Type {u_types[ix]} Thermo {u_thermos[ix]}" for ix in range(len(observables))],
                        colors=[colorByType(u_types[ix]) for ix in range(len(observables))],
                        save_path=f"traj_{obs_name}_cum_std.png" if args.useAgg else None
                    )

                if not args.useAgg:
                    plt.show()
                plt.close()            

            if DO_PCA:

                # Principal component analysis (PCA)
                from sklearn.decomposition import PCA
                (result, types, repeats, thermos) = \
                FNManager.PCA(filters=filters, frames=frames, verbose=True)

                # traj_infos = [entry['traj_info'] for entry in result]
                all_projections = [entry['projection'] for entry in result]
                all_projections_PC1 = [proj[:, 0] for proj in all_projections]
                all_projections_PC2 = [proj[:, 1] for proj in all_projections]
                explained_variances = [entry['explained_variance'] for entry in result]
                all_types = [types[entry['traj_info'][0]] for entry in result]
                all_repeats = [repeats[entry['traj_info'][1]] for entry in result]
                all_thermoIxs = [thermos[entry['traj_info'][2]] for entry in result]
                # print("PCA traj_infos", traj_infos)
                # print("PCA explained_variances", explained_variances)
                print("PCA all_projections_PC1 shapes", [proj.shape for proj in all_projections_PC1])
                # exit(2)

                minX = np.nanmin([proj[:, 0].min() for proj in all_projections])
                maxX = np.nanmax([proj[:, 0].max() for proj in all_projections])
                minY = np.nanmin([proj[:, 1].min() for proj in all_projections])
                maxY = np.nanmax([proj[:, 1].max() for proj in all_projections])
                print(f"PCA projection ranges: PC1 [{minX:.3f}, {maxX:.3f}], PC2 [{minY:.3f}, {maxY:.3f}]")

                # Get PCA projection index
                def get_projIx_by_type_repeat_thermo(type_val, repeatIx, thermoIx):
                    """ Get the index of the PCA projection for a specific
                        type, repeat, and thermo combination. """
                    all_proj_ix = None
                    for ix, entry in enumerate(result):
                        curr_typeIx, curr_repeat, curr_thermoIx = entry['traj_info']
                        curr_type = types[curr_typeIx]
                        if curr_type == type_val and curr_repeat == repeatIx and curr_thermoIx== thermoIx:
                            all_proj_ix = ix
                            break
                    return all_proj_ix
                #

                proj_T300_Type1_Ix = get_projIx_by_type_repeat_thermo(type_val=1, repeatIx=0, thermoIx=0)
                proj_T300_Type3_Ix = get_projIx_by_type_repeat_thermo(type_val=3, repeatIx=0, thermoIx=0)
                proj_T300_Type1 = all_projections[proj_T300_Type1_Ix]
                proj_T300_Type3 = all_projections[proj_T300_Type3_Ix]

                # region Iterate through the processed trajectories
                # for entry in result:
                #     traj_info = entry['traj_info']
                #     [curr_typeIx, curr_repeat, curr_thermoIx] = traj_info
                #     curr_type = types[curr_typeIx]
                #     curr_projection = entry['projection']
                #     explained_variance = entry['explained_variance']
                #     print(f"Trajectory info: {traj_info}, projection shape: {curr_projection.shape}, explained variance: {explained_variance}")
                #     type_idx = traj_info[0]
                #     # Plot PC1 vs PC2
                #     plotFN = None
                #     if args.useAgg:
                #         plotFN = f"traj_pca.png"
                #     #print(f"curr_type: {curr_type}, color: {colorByType(curr_type)}")
                #     plotScatter(
                #         Y=[curr_projection[:, 0]],
                #         X=[curr_projection[:, 1]],
                #         title="PCA Projection",
                #         xlabel="PC1",
                #         ylabel="PC2",
                #         xlim=(minX, maxX),
                #         ylim=(minY, maxY),
                #         labels=None,
                #         legend=True,
                #         colors=[colorByType(curr_type)],
                #         save_path=plotFN,
                #     )
                # endregion

                PRINT__, PLOT__ = True, True
                if PRINT__:
                    print(f"Explained variance by PC1 and PC2 for trajectory with Type 1, Repeat 0, Thermo 0: {explained_variances[proj_T300_Type1_Ix]}")
                    print(f"Explained variance by PC1 and PC2 for trajectory with Type 3, Repeat 0, Thermo 0: {explained_variances[proj_T300_Type3_Ix]}")
                if PLOT__:
                    plotFN = None
                    if args.useAgg:
                        plotFN = f"traj_pca.png"
                    stride =7
                    plotScatter(
                        X=[all_projections_PC1[proj_T300_Type1_Ix][::stride], all_projections_PC1[proj_T300_Type3_Ix][::stride]],
                        Y=[all_projections_PC2[proj_T300_Type1_Ix][::stride], all_projections_PC2[proj_T300_Type3_Ix][::stride]],
                        title="PCA Projection of Two Trajectories",
                        xlabel="PC1",
                        ylabel="PC2",
                        xlim=(minX, maxX),
                        ylim=(minY, maxY),
                        labels=[f"Seed {all_repeats[proj_T300_Type1_Ix]} Type {all_types[proj_T300_Type1_Ix]} Thermo {all_thermoIxs[proj_T300_Type1_Ix]}", f"Seed {all_repeats[proj_T300_Type3_Ix]} Type {all_types[proj_T300_Type3_Ix]} Thermo {all_thermoIxs[proj_T300_Type3_Ix]}"],
                        legend=True,
                        colors=[colorByType(all_types[proj_T300_Type1_Ix]), colorByType(all_types[proj_T300_Type3_Ix])],
                        save_path=f"traj_pca_example_point.png" if args.useAgg else None,
                        s=0.5,
                    )

                PRINT__, PLOT__ = True, True
                if PRINT__:
                    pass
                if PLOT__:
                    plotFN = None
                    if args.useAgg:
                        plotFN = f"traj_pca.png"
                    stride = 10
                    plotScatter(
                        X=[proj[::stride] for proj in all_projections_PC1],
                        Y=[proj[::stride] for proj in all_projections_PC2],
                        title=f"All PCA Projections at stride {stride}",
                        xlabel="PC1",
                        ylabel="PC2",
                        xlim=(minX, maxX),
                        ylim=(minY, maxY),
                        labels=None,
                        legend=True,
                        colors=[colorByType(type_val) for type_val in all_types],
                        save_path=plotFN,
                        s=0.5,
                    )

                MSM_results_0, MSM_results_1 = FNManager.MSM(result, lag=1,  n_states=5, verbose=True)

                #all_repeats all_thermoIxs 
                
                # Iterate through the MSM results for each type
                #for tyIx, typeVal in enumerate(types):
                for mIx, MSM_result in enumerate(MSM_results_0):
                    print(f"Type 0 MSM counter {mIx}")
                    print("MSM assignments", MSM_result["assignments"])
                    print("MSM transition matrix", MSM_result["transition_matrix"])
                    print("MSM stationary distribution", MSM_result["stationary_distribution"])
                    print("MSM implied timescales", MSM_result["implied_timescales"])
                    print("MSM cluster centers", MSM_result["cluster_centers"])

                for mIx, MSM_result in enumerate(MSM_results_1):
                    print(f"Type 1 MSM counter {mIx}")
                    print("MSM assignments", MSM_result["assignments"])
                    print("MSM transition matrix", MSM_result["transition_matrix"])
                    print("MSM stationary distribution", MSM_result["stationary_distribution"])
                    print("MSM implied timescales", MSM_result["implied_timescales"])
                    print("MSM cluster centers", MSM_result["cluster_centers"])

                if not args.useAgg:
                    plt.show()
                plt.close()


    #region Restart: write restart files into self.dir/restDir/restDir.<seed>
    if (args.restDir):
        TRAJECTORY_REQUIRED = True
        FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols)
        FNManager.write_restarts_from_trajectories(args.restDir, args.topology, dry=args.dry)
    #endregion

#endregion --------------------------------------------------------------------

if __name__ == "__main__":

    #region Parse arguments
    parser = argparse.ArgumentParser(description='REBAS: Replica Exchange Analysis Script')
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
    parser.add_argument('--useAgg', action='store_true', default=False, help="Use Agg backend for matplotlib (no display).")
    
    args = parser.parse_args()
    #endregion

    if args.useAgg:
        matplotlib.use('Agg')

    main(args)

    print("Robosample commit: 9b1a35012a0bda3e6c17540e8fc63fe087d258bb")
    print("Molmodel commit: f9e4dfd520451189b616159dcae43e2329e4dccd singularity")
    print("Simbody commit: 749f47ef9dfaef7e40594a917ee16d2ce4e9dbc8 master")
    print("OpenMM commit: 63e113d9557199d36587457deb159ae34fd75188 drilling")

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
