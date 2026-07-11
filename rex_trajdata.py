# rex_trajdata.py
import sys
import mdtraj as md
import numpy as np

from batana import BATIndexBuilder
from observables import Observables

# -----------------------------------------------------------------------------
#                      Robosample trajectory reader
#region REXTrajData ---------------------------------------------------------------
class REXTrajData:
    """Read a single DCD trajectory using MDTraj."""

    def __init__(self, filepath, topology="trpch/ligand.prmtop"):
        self.filepath = filepath
        self.topology = topology
        self.traj = self._load_trajectory()
        self._bat_indices = None
        
                
        self._observables = {
            "distance": lambda t, pair: md.compute_distances(t, [pair]),
            "rg": lambda t: md.compute_rg(t),
            "bat_rmsf": lambda t: self._compute_bat_rmsf(t),
        }        
    #

    def _get_bat_indices(self):
        """Build and cache BAT indices from trajectory topology."""
        if self._bat_indices is None:
            self._bat_indices = BATIndexBuilder.from_topology(self.traj.topology)
        return self._bat_indices

    def _compute_bat_rmsf(self, traj):
        """Compute BAT-space RMSF from a trajectory slice or full trajectory."""
        boIxs, angIxs, dihIxs = self._get_bat_indices()
        bos, angs, dihs = Observables.compute_bat_values(traj, boIxs, angIxs, dihIxs)
        return Observables.compute_bat_rmsf(bos, angs, dihs)

    # Actually helper for __init__, but kept separate for clarity and potential reuse
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
    #

    # Getter for trajectory
    def get_traj(self):
        return self.traj
    #

    # Get XYZ coordinates for PCA or other analyses
    def get_xyz(self, traj, superpose=True):
        selection = self.traj.top.select("name CA")
        if superpose:
            self.traj.superpose(self.traj, frame=0, atom_indices=selection)

        return self.traj.xyz[:, selection, :]
    #

    # Get observable from trajectory using external function
    def get_traj_observable(self, observable="rg", *, frames=None, verbose=False, **kwargs):
        """ Get observable from trajectory
        Args:
            observable (str or callable): Observable to compute. If str, must be a key in self._observables.
            frames (list or slice, optional): Frames to include. If None, use all frames.
            **kwargs: Additional arguments to pass to the observable function.
            Returns:
            Tuple[Any, Dict[str, Any]]: Observable values and metadata.
        """
        
        traj = self.traj[frames] if frames is not None else self.traj

        meta = {
            "filepath": self.filepath,
            "n_frames": traj.n_frames,
            "n_atoms": traj.n_atoms,
            "frames": frames,
        }

        if isinstance(observable, str):
            if observable not in self._observables:
                raise ValueError(f"Unknown observable '{observable}'. Options: {list(self._observables)}")
            fn = self._observables[observable]
            meta["observable"] = observable

        elif callable(observable):
            fn = observable
            meta["observable"] = getattr(fn, "__name__", str(fn))
        
        else:
            raise TypeError("observable must be a string key or a callable")

        obs_raw = fn(traj, **kwargs)
        obs = obs_raw if isinstance(obs_raw, dict) else np.asarray(obs_raw)
        return (obs, meta)
    #

    # Clear trajectory from memory
    def clear(self):
        """ For memory """
        del self.traj
        self.traj = None
        ##
    #
    
#endregion --------------------------------------------------------------------
