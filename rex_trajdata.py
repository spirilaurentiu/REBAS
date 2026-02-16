# rex_trajdata.py
from typing import Callable, Any, Dict, Tuple, Optional

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
                
        self._observables = {
            "distance": lambda t, pair: md.compute_distances(t, [pair]).ravel(),
            "rg": lambda t: md.compute_rg(t),
        }        

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

    def get_traj_observable(self, observable="rg", *, frames=None, **kwargs):
        """ Get observable from trajectory
        Args:
            observable (str or callable): Observable to compute. If str, must be a key in self._observables.
            frames (list or slice, optional): Frames to include. If None, use all frames.
            **kwargs: Additional arguments to pass to the observable function.
            Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Observable values and metadata.
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

        obs = np.asarray(fn(traj, **kwargs))
        return (obs, meta)
    #

    def clear(self):
        """ For memory """
        del self.traj
        self.traj = None
        ##

#endregion --------------------------------------------------------------------
