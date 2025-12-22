# rex_trajdata.py
import pandas as pd
import sys
import mdtraj as md
import numpy as np

# -----------------------------------------------------------------------------
#                      Robosample trajectory reader
#region REXTrajData ---------------------------------------------------------------
class REXTrajData:
    """Read a single DCD trajectory using MDTraj."""

    def __init__(self, filepath, topology="trpch/ligand.prmtop"):
        self.filepath = filepath
        self.topology = topology
        self.traj = self._load_trajectory()

    def _load_trajectory(self):
        """ MDTraj load trajectory
        Returns:
            MDTraj object
        """        
        try:
            traj = md.load_dcd(self.filepath, top=self.topology)
            return traj

        except Exception as e:
            print(f"Error loading {self.filepath}: {e}", file=sys.stderr)
            raise

    def get_traj(self):
        return self.traj
    #
    
    def get_traj_observable(self):
        """
        Get observable from trajectory
        """

        meta = {
            "filepath": self.filepath,
            "n_frames": self.traj.n_frames,
            "n_atoms": self.traj.n_atoms,
        }

        #    "seed": self.seed,
        #    "sim_type": sim_type

        aIx1 = 8
        aIx2 = 298
        observable = md.compute_distances(self.traj[:80000], [[aIx1, aIx2]]).ravel()
        #observable = md.compute_inertia_tensor(traj[::10])
        #observable = md.compute_rg(self.traj, masses=None)
        #print(observable)

        # ca_ix = self.traj.topology.select("name CA")     # indices of Cα atoms
        # traj_ca = self.traj.atom_slice(ca_ix)            # new traj with only Cα
        # I = md.compute_inertia_tensor(traj_ca)      # (n_frames, 3, 3)
        # evals, _ = np.linalg.eigh(I)                # (n_frames, 3), sorted ascending
        # I1, I2, I3 = evals.T
        # kappa2 = ((I1 - I2)**2 + (I2 - I3)**2 + (I3 - I1)**2) / (2.0 * (I1 + I2 + I3)**2)
        # asphericity = I3 - 0.5*(I1 + I2)
        # observable = kappa2

        return (observable, meta)
    #

    def clear(self):
        """ For memory """
        del self.traj
        self.traj = None
        #

#endregion --------------------------------------------------------------------
