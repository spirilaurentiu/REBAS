
# -----------------------------------------------------------------------------
#                      Robosample file manager
#region REXFileManager --------------------------------------------------------
from unittest import result

import MDAnalysis as mda
#from MDAnalysis.coordinates.TRJ import Restart
import mdtraj as md

import os
import re
import sys
import glob
import pandas as pd
import numpy as np
from utils import *

import scipy.stats
from scipy.signal import find_peaks

try:
    import parmed as pmd
    _HAS_PARMED = True
except Exception:
    _HAS_PARMED = False

from rex_data import REXData
from rex_trajdata import REXTrajData


class REXFNManager:
    """ File manager
    Attributes:
        dir: Directory containing the files
        FNRoots: File prefixes
        SELECTED_COLUMNS: columns selected to be read from file
    """
    def __init__(self, dir=None, FNRoots=None, SELECTED_COLUMNS=None, topology="trpch/ligand.prmtop"):
        self.dir = dir
        self.FNRoots = FNRoots
        self.SELECTED_COLUMNS = SELECTED_COLUMNS
        self.topology = topology
        self.OUTPUT_DATA = False
        self.TRAJECTORY_DATA = False
        
        self.entries, (self.n_types, self.n_sims, self.n_reps) = None, (0, 0, 0)
        #self.rexTrajData = None


    # Get seed and simulation type from filename. Determine if OUT or DCD
    def get_Info_FromFN(self, FN):
        """ Parse filename of type out.<7-digit seed>
        """

        seed, sim_type, thermo_index = -1, -1, -1

        if FN.startswith("out."):

            # pattern = r"out\.(?:.*\.)?(\d{7})$"
            # match = re.match(pattern, FN)            
            # if not match: raise ValueError(f"Filename '{FN}' does not match expected format 'out.<7-digit-seed>'")
            # seed = match.group(1)
            # sim_type = seed[2]  # third digit (0-based index)

            parts = FN.split('.')
            seed = parts[-1] # Gets the last element
            if len(seed) == 7 and seed.isdigit():
                sim_type = seed[2]
            else:
                raise ValueError("Invalid seed format")

            self.OUTPUT_DATA = True

        elif FN.endswith(".dcd"):
            pattern = r"([A-Za-z0-9]+)_(\d{7})\.repl(\d+)\.dcd$"
            match = re.match(pattern, FN)

            if not match:
                raise ValueError(f"Filename '{FN}' does not match expected format'<name_of_the_molecule>_<7-digit-seed>.repl<index>.dcd'")

            # mIx = 0
            # for match_group in match.groups():
            #     print("match group:", mIx, match_group)
            #     mIx += 1

            molName = match.group(1)
            seed = match.group(2)
            sim_type = seed[2]   # third digit
            thermo_index = match.group(3)

            self.TRAJECTORY_DATA = True

        return seed, sim_type, thermo_index
    #

    # Read data from a single file
    def getDataFromFile(self, FN, seed, sim_type, burnin=0):
        """ Read data from file
        """
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
                    seed, sim_type, thermo_index = self.get_Info_FromFN(os.path.basename(filepath))
                except ValueError as e:
                    print(f"Skipping file due to naming error: {filepath}\n{e}", file=sys.stderr)
                    continue

                df = self.getDataFromFile(filepath, seed, sim_type, burnin=burnin)
                all_data.append(df)

                print("done.", flush=True)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    #

    # Get trajectory data from a single file
    def getTrajDataFromFile(self, FN, seed, sim_type, thermo_index, observable_func, *, frames=None, **obs_kwargs):
        """ Get trajectory data from a single file
        :param FN: Filename
        :param seed: seed
        :param sim_type: simulation type
        :param observable_fn: Observable function or key
        :param frames: Frames to include
        :param obs_kwargs: Additional arguments for observable function
        :return: tuple of (obs, meta) where meta is a dict with keys:
            - filepath
            - n_frames
            - n_atoms
            - frames
            - seed
            - sim_type
        """
        
        thermodynamicIndex = re.search(r"repl(\d+)", FN).group(1) if "repl" in FN else None

        rexTrajData = REXTrajData(FN, topology=self.topology)

        obs, meta = rexTrajData.get_traj_observable(
            observable_func,
            frames=frames,
            **obs_kwargs
        )

        rexTrajData.clear()

        # Enrich meta with filename-derived info
        meta["seed"] = seed
        meta["sim_type"] = sim_type
        meta["thermoIx"] = thermo_index

        return obs, meta
    #

    # Prepare output array read from all filenames
    def prepareOutputArraySize(self, filters={}):
        """ Prepare output array metadata and calculate dimensions """
        entries = []

        for FNRoot in self.FNRoots:
            pattern = os.path.join(self.dir, FNRoot + "*")
            FN_matches = glob.glob(pattern)

            if not FN_matches:
                print(f"Warning: No files found matching {FNRoot}*", file=sys.stderr)
                continue

            for FN in FN_matches:
                try:
                    FN_basename = os.path.basename(FN)
                    seed, sim_type, thermo_index = self.get_Info_FromFN(FN_basename)
                    repeatIx = int(seed) % 100
                except ValueError as e:
                    print(f"[SKIP] Bad filename: {FN} -> {e}", file=sys.stderr)
                    continue

                # Apply filters
                FN_eligible = True
                for col, val in filters.items():
                    current_val = {"seed": seed,
                                   "sim_type": sim_type,
                                   "thermoIx": int(thermo_index)}.get(col)
                    if val is not None:
                        if isinstance(val, list):
                            if current_val not in val:
                                FN_eligible = False
                        elif current_val != val:
                            FN_eligible = False

                if FN_eligible:
                    # Column indices: 0:type, 1:seed, 2:repeat, 3:thermo, 4:filepath
                    entries.append([int(sim_type),
                                    int(seed),
                                    int(repeatIx),
                                    int(thermo_index),
                                    FN])

        if not entries:
            raise ValueError("No eligible files found after filtering.")

        # Convert to object array to handle mixed int and string (path)
        entries = np.array(entries, dtype=object)

        # Sort: Type (0) -> Repeat (2) -> Thermo (3)
        sort_indices = np.lexsort((
            entries[:, 3].astype(int), 
            entries[:, 2].astype(int), 
            entries[:, 0].astype(int)
        ))
        
        sorted_entries = entries[sort_indices]

        n_types = len(np.unique(sorted_entries[:, 0]))
        n_sims = len(np.unique(sorted_entries[:, 2]))
        n_replicas = len(np.unique(sorted_entries[:, 3]))
        
        self.entries = sorted_entries
        self.n_types = n_types
        self.n_sims = n_sims
        self.n_reps = n_replicas

        return sorted_entries, (n_types, n_sims, n_replicas)
    #

    # Get trajectory data extracted with passed functions
    def getTrajDataFromAllFiles(self, observable_func, *, filters={}, frames=None, **obs_kwargs):
        
        # 1. Get metadata and dimensions from filenames
        if self.entries is None:
            self.entries, (self.n_types, self.n_sims, self.n_reps) = self.prepareOutputArraySize(filters)
        
        uniq_sorted_types = np.unique(self.entries[:, 0].astype(int))
        uniq_sorted_repeats = np.unique(self.entries[:, 2].astype(int))
        uniq_sorted_thermos = np.unique(self.entries[:, 3].astype(int))

        # print("n_types", n_types, "n_sims", n_sims, "n_reps", n_reps)
        # print("Unique types:", uniq_sorted_types)
        # print("Unique repeats:", uniq_sorted_repeats)
        # print("Unique thermos:", uniq_sorted_thermos)

        # --- PASS 1: Probe lengths and determine n_obs_dim ---
        valid_results = []
        n_frames_list = []
        n_obs_dim = 1

        for row in self.entries:
            s_type, seed, repeatIx, thermoIx, FN = row
            #print("s_type", s_type, "seed", seed, "r_ix", repeatIx, "t_ix", thermoIx, "FN", FN)

            try:
                print(f"Reading {FN} ...", end=" ", flush=True)
                obs, meta = self.getTrajDataFromFile(
                    FN, seed, s_type, thermoIx, observable_func,
                    frames=frames, **obs_kwargs
                )

                # Capture dimensions
                n_frames_list.append(obs.shape[-1])
                if obs.ndim > 1:
                    n_obs_dim = obs.shape[0]

                # Map to indices immediately
                typeIx = np.searchsorted(uniq_sorted_types, int(s_type)) # is NOT the same as type
                repeatIx = np.searchsorted(uniq_sorted_repeats, int(repeatIx))
                thermoIx = np.searchsorted(uniq_sorted_thermos, int(thermoIx))
                
                # Store data and indices in memory
                valid_results.append(([typeIx, repeatIx, thermoIx], obs))
                print("done.")

            except Exception as e:
                print(f"  [SKIP] {os.path.basename(FN)}: {e}")

        if not valid_results:
            return None

        min_n_frames = min(n_frames_list)
        max_n_frames = max(n_frames_list)

        #print(f"Probed {len(valid_results)} valid trajectories. Frame counts range from {min_n_frames} to {max_n_frames}. Observable dimension: {n_obs_dim}.")
        # for vr in valid_results:
        #     print(vr)

        result = np.full((self.n_types, self.n_sims, self.n_reps, n_obs_dim, min_n_frames), np.nan)

        for vr in valid_results:
            (typeIx, repeatIx, thermoIx), obs = vr
            n_frames = obs.shape[-1]

            if n_frames < min_n_frames:
                print(f"  [WARN] Trajectory has {n_frames} frames, which is less than the minimum of {min_n_frames}. Skipping.")
                continue

            # Truncate if necessary
            obs_to_store = obs[..., :min_n_frames] if n_frames > min_n_frames else obs
            result[typeIx, repeatIx, thermoIx, ..., :obs_to_store.shape[-1]] = obs_to_store

        return (result, uniq_sorted_types, uniq_sorted_repeats, uniq_sorted_thermos)
    #

    # Perform PCA on all trajectories combined and return PCA for each trajectory
    def PCA(self, filters={}, frames=None, top="trpch/ligand.prmtop", verbose=False, **obs_kwargs):
        """ Perform PCA on all trajectories combined and return PCA for each trajectory """
        from sklearn.decomposition import IncrementalPCA

        ipca = IncrementalPCA(n_components=2, batch_size=5000)

        # 1. Get metadata
        if self.entries is None:
            self.entries, (self.n_types, self.n_sims, self.n_reps) = self.prepareOutputArraySize(filters)

        uniq_sorted_types = np.unique(self.entries[:, 0].astype(int))
        uniq_sorted_repeats = np.unique(self.entries[:, 2].astype(int))
        uniq_sorted_thermos = np.unique(self.entries[:, 3].astype(int))

        # Load trajectories
        valid_trajItraj = []
        n_frames_list = []

        for row in self.entries:
            s_type, seed, repeatIx, thermoIx, FN = row

            try:
                print(f"Reading {FN} ...", end=" ", flush=True)
                traj = md.load_dcd(FN, top=top, stride=1)

                # Convert to float32 to save memory
                traj.xyz = traj.xyz.astype('float32')

                n_frames_list.append(traj.n_frames)

                typeIx = np.searchsorted(uniq_sorted_types, int(s_type))
                repeatIx = np.searchsorted(uniq_sorted_repeats, int(repeatIx))
                thermoIx = np.searchsorted(uniq_sorted_thermos, int(thermoIx))

                valid_trajItraj.append(([typeIx, repeatIx, thermoIx], traj))
                print(f"{traj.n_frames} frames. Done.", flush=True)

            except Exception as e:
                print(f"  [SKIP] {os.path.basename(FN)}: {e}")

        if not valid_trajItraj:
            return None

        # Truncate to minimum frames
        min_n_frames = min(n_frames_list)
        print(f"Truncating trajectories to {min_n_frames} frames ...", flush=True)
        for ix, vr in enumerate(valid_trajItraj):
            valid_trajItraj[ix] = (vr[0], vr[1][:min_n_frames])
        print("done.", flush=True)

        # Use CA selection from the reference trajectory
        ref_traj = valid_trajItraj[0][1]
        selection = ref_traj.top.select("name CA")
        ref_frame = ref_traj[0]

        # Superpose all trajectories to the reference
        print("Superimposing trajectories ...", flush=True)
        for traj_info, traj in valid_trajItraj:
            traj.superpose(ref_frame, atom_indices=selection)
        print("done.", flush=True)

        # First pass: partial_fit
        print("Partial Fitting PCA model ...", flush=True)
        for traj_info, traj in valid_trajItraj:
            X = traj.xyz[:, selection, :].reshape(traj.n_frames, -1)
            ipca.partial_fit(X)
        print("done.", flush=True)

        # Second pass: transform
        print("Transforming trajectories ...", flush=True)
        result = []
        for traj_info, traj in valid_trajItraj:
            X = traj.xyz[:, selection, :].reshape(traj.n_frames, -1)
            proj = ipca.transform(X)

            result.append({
                "traj_info": traj_info,
                "projection": proj,
                "explained_variance": ipca.explained_variance_ratio_
            })
        print("done.", flush=True)

        return (result, uniq_sorted_types, uniq_sorted_repeats, uniq_sorted_thermos)
    #

    # Build a Markov State Model using PCA
    def MSM(self, pca_result, lag=50, n_states=100, verbose=False):
        """
        Build a Markov State Model (MSM) from PCA projections.

        Parameters
        ----------
        pca_result : list of dicts
            Output from self.PCA(), i.e. a list where each element contains:
                - "traj_info": metadata
                - "projection": (n_frames, n_components) PCA coordinates
        lag : int
            Lag time (in frames) for MSM transition counting.
        n_states : int
            Number of discrete microstates for clustering.

        Returns
        -------
        msm : dict
            {
                "assignments": list of arrays of state indices,
                "transition_matrix": T,
                "stationary_distribution": pi,
                "implied_timescales": its,
                "cluster_centers": kmeans.cluster_centers_
            }
        """

        from sklearn.cluster import MiniBatchKMeans

        # ------------------------------------------------------------
        # 1. Extract PCA projections
        # ------------------------------------------------------------
        if verbose:
            print("Extracting PCA projections...")

        traj_pca_projections = [entry["projection"] for entry in pca_result]  # list of (n_frames, n_components)

        if verbose:
            print(f"PCA projections shapes:", [proj.shape for proj in traj_pca_projections])

        X_concat = np.vstack(traj_pca_projections)

        if verbose:
            print(f"Concatenated PCA shape: {X_concat.shape}")
            print(f"Total frames: {X_concat.shape[0]}, PCA dim: {X_concat.shape[1]}")

        # ------------------------------------------------------------
        # 2. Cluster PCA space into microstates
        # ------------------------------------------------------------
        if verbose:
            print(f"Clustering into {n_states} states...")

        kmeans = MiniBatchKMeans(
            n_clusters=n_states,
            batch_size=10000,
            n_init=10
        )
        kmeans.fit(X_concat)

        # Assign each trajectory separately
        assignments = [kmeans.predict(proj) for proj in traj_pca_projections]

        # ------------------------------------------------------------
        # 3. Build transition matrix
        # ------------------------------------------------------------
        if verbose:
            print(f"Building transition matrix at lag = {lag}...")

        T = np.zeros((n_states, n_states), dtype=float)

        for assign in assignments:
            for i in range(len(assign) - lag):
                a = assign[i]
                b = assign[i + lag]
                T[a, b] += 1

        # Normalize rows
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        T = T / row_sums

        # ------------------------------------------------------------
        # 4. Stationary distribution
        # ------------------------------------------------------------
        if verbose:
            print("Computing stationary distribution...")

        w, v = np.linalg.eig(T.T)
        idx = np.argmax(np.real(w))
        pi = np.real(v[:, idx])
        pi = pi / pi.sum()

        # ------------------------------------------------------------
        # 5. Implied timescales
        # ------------------------------------------------------------
        if verbose:
            print("Computing implied timescales...")

        eigvals = np.linalg.eigvals(T)
        eigvals = np.sort(np.abs(eigvals))[::-1]  # descending
        its = -lag / np.log(eigvals[1:])          # skip eigenvalue 1

        # ------------------------------------------------------------
        # 6. Return MSM object
        # ------------------------------------------------------------
        msm = {
            "assignments": assignments,
            "transition_matrix": T,
            "stationary_distribution": pi,
            "implied_timescales": its,
            "cluster_centers": kmeans.cluster_centers_
        }

        if verbose:
            print("MSM construction complete.")

        return msm

    # Get trajectory data from all files. Deals with file globbing.
    def getTrajDataFromAllFiles_Old(self, observable_func, *, filters={}, frames=None, **obs_kwargs):
        """ Get trajectory data from all files
        :param observable_fn: Observable function or key
        :param frames: Frames to include
        :param obs_kwargs: Additional arguments for observable function
        :return: list of observable arrays and metadata containing:
            - filepath
            - n_frames
            - n_atoms
            - frames
            - seed
            - sim_type
        """
        
        entries, (n_types, n_sims, n_replicas, n_obss, n_frames) = self.prepareOutputArraySize(filters)
        print("n_types", n_types, "n_sims", n_sims, "n_replicas", n_replicas, "n_obss", n_obss, "n_frames", n_frames)
        exit(2)

        obsList = []
        metadata_rows = []

        for FNRoot in self.FNRoots:
            pattern = os.path.join(self.dir, FNRoot + "*")
            FN_matches = glob.glob(pattern)

            # print("self.dir", self.dir, "FNRoot", FNRoot)
            # print("pattern", pattern)
            # print("FN_matches", FN_matches)

            if not FN_matches:
                print(f"Warning: No files found matching {FNRoot}*", file=sys.stderr)
                continue

            for FN in FN_matches:

                try:
                    seed, sim_type, thermo_index = self.get_Info_FromFN(os.path.basename(FN))
                except ValueError as e:
                    print(f"[SKIP] Bad filename: {FN} -> {e}", file=sys.stderr)
                    continue

                # print("seed", "sim_type", "thermo_index", seed, sim_type, thermo_index)
                # for col, val in filters.items():
                #     print("filters:", "col", "val", col, val)

                # Apply filters (if any)
                FN_eligible = True
                for col, val in filters.items():
                    if col == "seed":
                        # Check if seed matches scalar OR is inside the list of allowed values
                        if isinstance(val, list):
                            if seed not in val:
                                FN_eligible = False
                        elif seed != val:
                            FN_eligible = False

                    elif col == "sim_type":
                        if isinstance(val, list):
                            if sim_type not in val:
                                FN_eligible = False
                        elif sim_type != val:
                            FN_eligible = False

                    elif col == "thermoIx":
                        current_thermo = int(thermo_index)
                        if isinstance(val, list):
                            if current_thermo not in val:
                                FN_eligible = False
                        elif current_thermo != val:
                            FN_eligible = False

                if not FN_eligible:
                    #print("File NOT eligible")
                    continue

                print(f"Reading {FN} ...", end=" ", flush=True)
                try:
                    obs, meta = self.getTrajDataFromFile(
                        FN, seed, sim_type, thermo_index,
                        observable_func,
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
        """ Read trajectory files of the form <mol>_<seed>.<replica>.dcd from self.dir,
        extract the last frame, and write restart files into self.dir/restDir/restDir.<seed>.
        Arguments:
            restDir : Name of the subdirectory (inside self.dir) to store restart files
            topology: Path to a topology file (AMBER prmtop or PDB) required to read the DCDs
            out_ext : Output extension/format. 'rst7' (default) requires ParmEd; if
                      ParmEd is unavailable, a PDB will be written instead.
            dry     : If True, do not actually write files, just print what would be done.
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
    #
#endregion --------------------------------------------------------------------
