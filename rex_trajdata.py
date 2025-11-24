# rex_trajdata.py
import pandas as pd
import sys
import mdtraj as md

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
        try:
            traj = md.load_dcd(self.filepath, top=self.topology)
            return traj

        except Exception as e:
            print(f"Error loading {self.filepath}: {e}", file=sys.stderr)
            raise

    def get_traj(self):
        return self.traj

#endregion --------------------------------------------------------------------
