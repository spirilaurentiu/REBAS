
# -----------------------------------------------------------------------------
#                      Robosample file manager
#region REXFileManager --------------------------------------------------------
import MDAnalysis as mda
#from MDAnalysis.coordinates.TRJ import Restart
import mdtraj as md

import os
import re
import sys
import glob
import pandas as pd
import numpy as np

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
    def getTrajDataFromFile(self, FN, seed, sim_type, thermo_index, observable_fn, *, frames=None, **obs_kwargs):
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
