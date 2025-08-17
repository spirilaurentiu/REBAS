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
from rex_efficiency import RoboAnalysis, REXEfficiency
from rex_efficiency import REXEnergy



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
    def write_restarts_from_trajectories_mda(self, restDir, topology):
        '''
        Read trajectory files of the form <mol>_<seed>.<replica>.dcd,
        extract the last frame, and write restart (rst7) files into restDir.

        Arguments:
            restDir: Directory to store restart files
            topology: Path to a topology file (prmtop or pdb) required to read the DCDs

        Output:
            For each trajectory file <mol>_<seed>.<replica>.dcd, writes:
                restDir/<mol>_<seed>.<replica>.rst7
        '''
        print(f"Writing restarts with MDAnalysis is not implemented", file=sys.stderr)
        return
    
        target_dir = os.path.join(self.dir, restDir)
        os.makedirs(target_dir, exist_ok=True)

        traj_files = glob.glob(os.path.join(self.dir, "*.dcd"))
        if not traj_files:
            print(f"Warning: No trajectory files found in {self.dir}", file=sys.stderr)
            return

        for traj in traj_files:
            base = os.path.basename(traj)
            root, _ = os.path.splitext(base)  # <mol>_<seed>.<replica>

            try:
                u = mda.Universe(topology, traj)
                u.trajectory[-1]  # jump to last frame
                rst_path = os.path.join(target_dir, f"{root}.rst7")
                with mda.Writer(rst_path, n_atoms=u.atoms.n_atoms) as w:
                    w.write(u.select_atoms("all"))
                print(f"Wrote restart: {rst_path}")
            except Exception as e:
                print(f"Error processing {traj}: {e}", file=sys.stderr)
    #

    def write_restarts_from_trajectories(self, restDir, topology, out_ext='rst7'):
        '''
        Read trajectory files of the form <mol>_<seed>.<replica>.dcd from self.dir,
        extract the last frame, and write restart files into self.dir/restDir.

        Arguments:
            restDir : Name of the subdirectory (inside self.dir) to store restart files
            topology: Path to a topology file (AMBER prmtop or PDB) required to read the DCDs
            out_ext : Output extension/format. 'rst7' (default) requires ParmEd; if
                      ParmEd is unavailable, a PDB will be written instead.

        Output:
            For each trajectory file <mol>_<seed>.<replica>.dcd, writes:
                self.dir/restDir/<mol>_<seed>.<replica>.<out_ext>
        '''
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
                    struct.save(out_path, overwrite=True)

                elif ext == 'pdb':
                    out_path = os.path.join(seed_dir, f"ligand.s{replica}.pdb")
                    last.save_pdb(out_path)

                else:
                    out_path = os.path.join(seed_dir, f"ligand.s{replica}.pdb")
                    last.save_pdb(out_path)

                print(f"Wrote restart: {out_path}")

            except Exception as e:
                print(f"Error processing {traj}: {e}", file=sys.stderr)

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

    OUTPUT, TRAJECTORY = False, True

    if OUTPUT:
        
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

            if args.restDir:
                FNManager.write_restarts_from_trajectories(args.restDir, args.topology)

        # Apply filters if specified
        if args.filterBy:
            filters = parse_filters(args.filterBy)
            for col, val in filters.items():
                if col not in df.columns:
                    raise ValueError(f"Filter column '{col}' not found in DataFrame columns.")
                df = df[df[col] == val]

        print(df)

        #region Efficiency figures
        if 1 in args.figures:
            roboAna = RoboAnalysis(df)

            # Acceptance rate
            acc_df = roboAna.compute_acceptance(cumulative=True)
            print(acc_df)

        if 2 in args.figures:

            rexEff = REXEfficiency(df)

            # Calculate exchange rate
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
        #endregion  


        potentialEnergyFigure = False
        efficiencyFigure = False

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
        #region Validation figures


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

    if TRAJECTORY:
        FNManager = REXFNManager(args.dir, args.inFNRoots, args.cols)
        
        FNManager.write_restarts_from_trajectories(args.restDir, args.topology)

#endregion --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True,
                   help='Directory with data files')
    parser.add_argument('--inFNRoots', nargs='+', required=True,
                   help='Robosample processed output file names')
    parser.add_argument('--restDir',
                   help='Directory with restart files')
    parser.add_argument('--topology',
                   help='Topology file')
    parser.add_argument('--cols', nargs='+',
                   help='Columns to be read')
    parser.add_argument('--filterBy', nargs='*', default=[],
                   help='Optional filters in the format col=value (e.g. wIx=0)')
    parser.add_argument('--useCache', action='store_true', help='Load data from cache file if it exists')
    parser.add_argument('--writeCache', action='store_true', help='Write new cache file (fails if file exists)')
    parser.add_argument('--cacheFile', default='rex_cache.pkl', help='Path to cache file')
    parser.add_argument('--figures', nargs='+', default=[], type=int,
                   help='Figures or data that will be produced.')    
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
