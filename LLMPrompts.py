# main.py
from mystats import *

from rex_data import REXData
from rex_efficiency import RoboAnalysis, REXEfficiency
from rex_trajdata import REXTrajData


# -----------------------------------------------------------------------------
#                      Utility Functions
#region Utility Functions ----------------------------------------------------

#endregion --------------------------------------------------------------------


# -----------------------------------------------------------------------------
#                      Robosample file manager
#region REXFileManager --------------------------------------------------------
#from MDAnalysis.coordinates.TRJ import Restart
import mdtraj as md

try:
    import parmed as pmd
    _HAS_PARMED = True
except Exception:
    _HAS_PARMED = False

class REXFNManager:
    def __init__(self, dir=None, FNRoots=None, SELECTED_COLUMNS=None, topology="trpch/ligand.prmtop"):
        self.dir = dir
        self.FNRoots = FNRoots
        self.SELECTED_COLUMNS = SELECTED_COLUMNS
        self.topology = topology
        self.OUTPUT_DATA = False
        self.TRAJECTORY_DATA = False

    # Get seed and simulation type from filename. Determine if OUT or DCD
    def get_Info_FromFN(self, FN):
        seed, sim_type, thermo_index = -1, -1, -1

        if FN.startswith("out."):
            #match = re.match(r"out\.(\d{7})", FN)
            #match = re.match(r"out\..*\.(\d{7})", FN)
            match = re.match(r"out\.(?:.*\.)?(\d{7})$", FN)
            if not match: raise ValueError(f"Filename '{FN}' does not match expected format 'out.<7-digit-seed>'")

            seed = match.group(1)
            sim_type = seed[2]  # third digit (0-based index)

            self.OUTPUT_DATA = True

        elif FN.endswith(".dcd"):
            pattern = r"([A-Za-z0-9]+)_(\d{7})\.repl(\d+)\.dcd$"
            match = re.match(pattern, FN)

            if not match:
                raise ValueError(f"Filename '{FN}' does not match expected format'<name_of_the_molecule>_<7-digit-seed>.repl<index>.dcd'")


            molName = match.group(1)
            seed = match.group(2)
            sim_type = seed[2]   # third digit
            thermo_index = match.group(3)

            self.TRAJECTORY_DATA = True

        return seed, sim_type, thermo_index
    #

    # Read data from a single file
    def getDataFromFile(self, FN, seed, sim_type, burnin=0):
        rex = REXData(FN, self.SELECTED_COLUMNS)
        df = rex.get_dataframe().copy()
        df['seed'] = seed
        df['sim_type'] = sim_type

        # Remove burn-in rows
        if burnin > 0:
            df = df.iloc[burnin:].reset_index(drop=True)

        return df
    #

    # Read data from all files
    def getDataFromAllFiles(self, burnin=0):
        all_data = []
        for FNRoot in self.FNRoots:
            matches = glob.glob(os.path.join(self.dir, FNRoot + "*"))
            if not matches:
                print(f"Warning: No files found matching {FNRoot}*", file=sys.stderr)
                continue

            for filepath in matches:

                print("Reading", filepath, "...", end = ' ', flush=True)

                try:
                    seed, sim_type = self.get_Info_FromFN(os.path.basename(filepath))
                except ValueError as e:
                    print(f"Skipping file due to naming error: {filepath}\n{e}", file=sys.stderr)
                    continue

                df = self.getDataFromFile(filepath, seed, sim_type, burnin=burnin)
                all_data.append(df)

                print("done.", flush=True)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    #

    # Get trajectory data from a single file
    def getTrajDataFromFile(self, FN, seed, sim_type, thermo_index, observable_fn, *, frames=None, **obs_kwargs):
        
        thermodynamicIndex = re.search(r"repl(\d+)", FN).group(1) if "repl" in FN else None

        trajData = REXTrajData(FN, topology=self.topology)

        obs, meta = trajData.get_traj_observable(
            observable_fn,
            frames=frames,
            **obs_kwargs
        )

        trajData.clear()

        # Enrich meta with filename-derived info
        meta["seed"] = seed
        meta["sim_type"] = sim_type
        meta["thermoIx"] = thermo_index

        return obs, meta
    #

    # Get trajectory data from all files. Deals with file globbing.
    def getTrajDataFromAllFiles(self, observable_fn, *, filters={}, frames=None, **obs_kwargs):
        
        obsList = []
        metadata_rows = []

        for FNRoot in self.FNRoots:
            pattern = os.path.join(self.dir, FNRoot + "*")
            FN_matches = glob.glob(pattern)

            if not FN_matches:
                print(f"Warning: No files found matching {FNRoot}*", file=sys.stderr)
                continue

            for FN in FN_matches:

                try:
                    seed, sim_type, thermo_index = self.get_Info_FromFN(os.path.basename(FN))
                except ValueError as e:
                    print(f"[SKIP] Bad filename: {FN} -> {e}", file=sys.stderr)
                    continue

                # Apply filters (if any)
                FN_eligible = True
                for col, val in filters.items():
                    if   (col == "seed") and (seed != val):
                        FN_eligible = False
                    elif (col == "sim_type") and (sim_type != val):
                        FN_eligible = False
                    elif (col == "thermoIx") and (int(thermo_index) != val):
                        FN_eligible = False
                if not FN_eligible:
                    continue

                print(f"Reading {FN} ...", end=" ", flush=True)
                try:
                    obs, meta = self.getTrajDataFromFile(
                        FN, seed, sim_type, thermo_index,
                        observable_fn,
                        frames=frames,
                        **obs_kwargs
                    )
                except Exception as e:
                    print(f"[FAIL] Error reading {FN}: {e}", file=sys.stderr)
                    continue

                obsList.append(obs)
                metadata_rows.append(meta)
                print("done.")

        metadata_df = pd.DataFrame(metadata_rows)
        return obsList, metadata_df
    #
    
    # Write restart files into self.dir/restDir/restDir.<seed>
    def write_restarts_from_trajectories(self, restDir, topology, out_ext='rst7', dry=True):
        
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
            #seed_dir = os.path.join(self.dir, restDir, f"{seed}")
            seed_dir = os.path.join(restDir, f"{seed}")
            print(f"Creating seed directory: {seed_dir} ...", end=' ', flush=True)

            if not dry:
                os.makedirs(seed_dir, exist_ok=True)
                print("done.", flush=True)
            else:
                print()

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
                    print(f"Writing restart file: {out_path} ... ", end='', flush=True)
                    
                    if not dry:
                        struct.save(out_path, overwrite=True)
                        print("done.", flush=True)
                    else:
                        print()

                elif ext == 'pdb':
                    out_path = os.path.join(seed_dir, f"ligand.s{replica}.pdb")
                    print(f"Writing restart file: {out_path} ... ", end='', flush=True)

                    if not dry:
                        last.save_pdb(out_path)
                        print("done.", flush=True)
                    else:
                        print()

                else:
                    out_path = os.path.join(seed_dir, f"ligand.s{replica}.pdb")
                    print(f"Writing restart file: {out_path} ... ", end='', flush=True)

                    if not dry:
                        last.save_pdb(out_path)
                        print("done.", flush=True)
                    else:
                        print()

                print(f"Wrote restart: {out_path}")

            except Exception as e:
                print(f"Error processing {traj}: {e}", file=sys.stderr)

#endregion --------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                                MAIN
#region Main ------------------------------------------------------------------

# Parse a set of filters given as arguments in the format: ---filterBy col=val
def parse_filters(filters):
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

def main(args):

    OUTPUT_REQUIRED, TRAJECTORY_REQUIRED = False, False # default flags
    if args.inFNRoots[0][0:3] == 'out':
        OUTPUT_REQUIRED = True
    else:
        TRAJECTORY_REQUIRED = True
    
    FNManager = None # classes
    out_df, traj_df = None, None # pandas

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


        #region
        if "rex_eff" in args.figures:

            observables = []
            observables_meta = []
            repl_exxrs = []
            obs_stds = []
            obs_vars = []
            obs_statisticalEnergys = []
            obs_skews = []
            obs_taus = []
            obs_ESSs = []
            PRINT__, PLOT__ = False, False

            ix = -1
            # We iterate through each group (replica) one by one
            for (sim_type, seed, replicaIx), subdf_group in out_df.groupby(["sim_type", "seed", "replicaIx"]):
                ix += 1
                #print(f"Processing sim_type={sim_type}, seed={seed}, replicaIx={replicaIx} (group {ix})")
                
                # Extract the data for THIS specific replica
                obs_data = subdf_group["thermoIx"].to_numpy()
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
                (ACF_rho, obs_tau, ess) = autocorr2_revised(obs_data, lag_fraction=0.5, max_lag=50000)
                #print("obs_tau", obs_tau)
                #(ACF_rho, obs_tau, ess) = autocorr3_revised(obs_data, lag_fraction=0.5, max_lag=50000)
                #print("obs_tau", obs_tau)

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
                        ylim=(0, 13),
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
            min_len = min(len(Y) for Y in observables)
            max_len = max(len(Y) for Y in observables)
            min_glob = min(Y.min() for Y in observables)
            max_glob = max(Y.max() for Y in observables)
            obs_list_trimmed = np.array([Y[:min_len] for Y in observables])
            #print(obs_list_trimmed.shape)


            cumMean_list = []
            cumSom_list = []
            for ix, obs in enumerate(obs_list_trimmed):
                cumMean, cumSom = cum_scum(obs)
                cumMean_list.append(cumMean)
                cumSom_list.append(cumSom)

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
                          f" tau: {obs_taus[ix]:.3f} statEnergy: {obs_statisticalEnergys[ix]:.9f}")

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

        # Objective: Ensemble mean of the cummulative mean for each T
        if "traj_stats" in args.figures:

            utilObj = REXFNManager()

            GLOBAL_BURNIN = 10
            GLOBAL_END = None

            # pick your frame slice once:
            frames = slice(GLOBAL_BURNIN, GLOBAL_END)   # or slice(GLOBAL_BURNIN, None)

            # define any observable you want:
            def dist_8_298(traj, a1=8, a2=298):
                return md.compute_distances(traj, [[a1, a2]]).ravel()

            # Get trajectory data: (list of arrays of some observable, and 
            # metadata containing filepath, n_frames, n_atoms, frames, seed, sim_type)
            FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols)
            (observables, traj_metadata_df) = FNManager.getTrajDataFromAllFiles(
                dist_8_298,
                filters=filters,
                frames=frames,
                a1=8, a2=298,   # optional; overrides defaults
            )

            #print("traj_metadata_df:\n", traj_metadata_df)
            #exit(2)

            observables_meta = []
            for ix, obs in enumerate(observables):
                observables_meta.append({
                    "sim_type": traj_metadata_df.iloc[ix]["sim_type"],
                    "seed": traj_metadata_df.iloc[ix]["seed"],
                    "filepath": traj_metadata_df.iloc[ix]["filepath"],
                    "thermoIx": traj_metadata_df.iloc[ix]["thermoIx"],
                    })

            #print("observables_meta", observables_meta)
            #exit(2)

            # Get cumulative mean and som for each trajectory
            min_len = min(len(Y) for Y in observables)
            max_len = max(len(Y) for Y in observables)
            min_glob = min(Y.min() for Y in observables)
            max_glob = max(Y.max() for Y in observables)
            obs_list_trimmed = np.array([Y[:min_len] for Y in observables])

            #print("obs_list_trimmed", obs_list_trimmed.shape, flush=True)
            #exit(2)

            cumMean_list = []
            cumSom_list = []
            for ix, obs in enumerate(obs_list_trimmed):
                cumMean, cumSom = cum_scum(obs)
                cumMean_list.append(cumMean)
                cumSom_list.append(cumSom)

            #print("cumMean_list", [cumMean_list_entry.shape for cumMean_list_entry in cumMean_list], flush=True)
            #exit(2)

            # Get ensemble means and stds across trajectories
            type1_obs = []
            type3_obs = []
            for ix, obs in enumerate(obs_list_trimmed):
                sim_type = observables_meta[ix]["sim_type"]
                if int(sim_type) == 1:
                    type1_obs.append(obs)
                elif int(sim_type) == 3:
                    type3_obs.append(obs)
                else:
                    sys.exit(f"Unknown sim_type {sim_type} encountered.")
                    
            # Get ensemble means and stds across trajectories
            type1_obs_means, type1_obs_stds = [], []
            type3_obs_means, type3_obs_stds = [], []
            for ix, cum_mean in enumerate(cumMean_list):
                sim_type = observables_meta[ix]["sim_type"]
                if int(sim_type) == 1:
                    type1_obs_means.append(cum_mean)
                    type1_obs_stds.append(cumSom_list[ix])
                elif int(sim_type) == 3:
                    type3_obs_means.append(cum_mean)
                    type3_obs_stds.append(cumSom_list[ix])
                else:
                    sys.exit(f"Unknown sim_type {sim_type} encountered.")

            type1_ens_means, type1_ensemble_stds = ensemble_mean_and_std(type1_obs_means)
            type3_ens_means, type3_ensemble_stds = ensemble_mean_and_std(type3_obs_means)

            print("type1_ens_means", type1_ens_means.shape, flush=True)
            print("type1_ensemble_stds", type1_ensemble_stds.shape, flush=True)

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
            print("type1_bin_centers", type1_ens_hists["bin_centers"])
            print("type3_bin_centers", type3_ens_hists["bin_centers"])
            assert np.allclose(type1_ens_hists["bin_centers"], type3_ens_hists["bin_centers"]), "Bin centers do not match between types."

            # Get autocorrelation functions per trajectory
            acf_list = []
            fit_curve_list = []
            tau_opt_list = []
            for ix, obs in enumerate(obs_list_trimmed):
                acf, fit_curve, tau_opt = normalized_autocorrelation(obs, max_lag=min_len-1, estimate_tau=True)
                acf_list.append(acf)
                fit_curve_list.append(fit_curve)
                tau_opt_list.append(tau_opt)

            PRINT__, PLOT__ = True, True

            if PRINT__:
                print("observables_meta", observables_meta)
                #print("observables", observables)
                print("Careful: trimmed to min length:", min_len, ". Max length:", max_len)
            if PLOT__:
                plotFN = None
                if args.useAgg:
                    plotFN = f"ee.png"                
                plot1D(obs_list_trimmed,
                       title="End-to-end distance",
                       xlabel="Frame",
                       ylabel=dist_8_298.__name__,
                       labels=[f"{observables_meta[ix]['seed']} type {observables_meta[ix]['sim_type']}" \
                               for ix in range(len(obs_list_trimmed))],
                        colors=[colorByType(observables_meta[ix]['sim_type']) \
                               for ix in range(len(obs_list_trimmed))],
                        save_path=plotFN
                       )

            if PRINT__:
                pass
                #print("cumMean_list", cumMean_list)
            if PLOT__:
                plotFN = None
                if args.useAgg:
                    plotFN = f"eecm.png"                 
                plot1D(cumMean_list,
                       title="Cumulative mean of end-to-end distance",
                       xlabel="Frame",
                       ylabel=dist_8_298.__name__ + " cumulative mean",
                       labels=[f"{observables_meta[ix]['seed']} type {observables_meta[ix]['sim_type']}" \
                               for ix in range(len(cumMean_list))],
                        colors=[colorByType(observables_meta[ix]['sim_type']) \
                               for ix in range(len(cumMean_list))],
                        save_path=plotFN
                       )

            if PRINT__:    
                pass
            if PLOT__:
                plotFN = None
                if args.useAgg:
                    plotFN = f"eecs.png"
                plot1D(cumSom_list,
                       title="Cumulative standard deviation of the mean of end-to-end distance",
                       xlabel="Frame",
                       ylabel=dist_8_298.__name__ + " cumulative som",
                       labels=[f"{observables_meta[ix]['seed']} type {observables_meta[ix]['sim_type']}" \
                               for ix in range(len(cumSom_list))],
                        colors=[colorByType(observables_meta[ix]['sim_type']) \
                               for ix in range(len(cumSom_list))],
                        save_path=plotFN
                       )

            if PRINT__:
                print("type1_ens_means", type1_ens_means.shape)
                print("type1_ensemble_stds", type1_ensemble_stds.shape)
            if PLOT__:
                plotFN = None
                if args.useAgg:
                    plotFN = f"eecm_ens.png"
                plot1D([type1_ens_means, type3_ens_means],
                       #X=[type1_ens_means, type3_ens_means],
                       Yerr=[type1_ensemble_stds, type3_ensemble_stds], Yerr_every=10,
                       title="Ensemble mean of cumulative mean of end-to-end distance",
                       xlabel="Frame",
                       ylabel=dist_8_298.__name__ + " ensemble cumulative mean",
                       labels=["type 1", "type 3"],
                       colors=[colorByType(1), colorByType(3)],
                       save_path=plotFN
                       )
                plotFN = None
                if args.useAgg:
                    plotFN = f"eecs_ens.png"
                plot1D([type1_ensemble_stds, type3_ensemble_stds],
                       #X=[type1_ens_means, type3_ens_means],
                       #Yerr=[type1_ensemble_stds, type3_ensemble_stds],
                       title="Ensemble mean of cumulative std of the mean of EE distance",
                       xlabel="Frame",
                       ylabel=dist_8_298.__name__ + " ensemble cumulative som",
                       labels=["type 1", "type 3"],
                       colors=[colorByType(1), colorByType(3)],
                       save_path=plotFN
                       )

            if PRINT__:
                print("type1 stats (mean, std, entropy, num_modes):")
                print(type1_ens_hists["entropy"], type1_ens_hists["num_modes"])
                print("type3 stats (mean, std, entropy, num_modes):")
                print(type3_ens_hists["entropy"], type3_ens_hists["num_modes"])
            if PLOT__:
                plotFN = None
                if args.useAgg:
                    plotFN = f"ee_dist_distrib.png"
                plot1D([type1_ens_hists["mean"], type3_ens_hists["mean"]],
                       X=[type1_ens_hists["bin_centers"], type3_ens_hists["bin_centers"]],
                       Yerr=[type1_ens_hists["std"], type3_ens_hists["std"]],
                       title="End-to-end distance distribution",
                       xlabel=dist_8_298.__name__,
                       ylabel="Probability density",
                       labels=["type 1", "type 3"],
                       colors=[colorByType(1), colorByType(3)],
                       save_path=plotFN
                       )

            PRINT__, PLOT__ = True, True

            if PRINT__:
                #print("fit_curve_list", fit_curve_list)
                print("tau_opt_list", tau_opt_list)
            if PLOT__:
                plotFN = None
                if args.useAgg:
                    plotFN = f"ee_acf.png"
                plot1D(acf_list,
                       title="Autocorrelation of end-to-end distance",
                       xlabel="Lag (frames)",
                       ylabel="Autocorrelation",
                       labels=[f"{observables_meta[ix]['seed']} type {observables_meta[ix]['sim_type']}" \
                               for ix in range(len(acf_list))],
                        colors=[colorByType(observables_meta[ix]['sim_type']) \
                               for ix in range(len(acf_list))],
                        save_path=plotFN
                       )
                plotFN = None
                if args.useAgg:
                    plotFN = f"ee_acf_fit.png"
                plot1D(fit_curve_list,
                       title="Fitted exponential decay to ACF of end-to-end distance",
                       xlabel="Lag (frames)",
                       ylabel="Fitted ACF",
                       labels=[f"{observables_meta[ix]['seed']} type {observables_meta[ix]['sim_type']} (τ={tau_opt_list[ix]:.2f})" \
                               for ix in range(len(fit_curve_list))],
                        colors=[colorByType(observables_meta[ix]['sim_type']) \
                               for ix in range(len(fit_curve_list))],
                        save_path=plotFN
                          )

            if not args.useAgg:    
                plt.show()
            plt.close()       

            PRINT__, PLOT__ = False, False

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
    # ......
    parser.add_argument('--useAgg', action='store_true', default=False, help="Use Agg backend for matplotlib (no display).")
    
    args = parser.parse_args()
    #endregion

    if args.useAgg:
        matplotlib.use('Agg')

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
#endregion
