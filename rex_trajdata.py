# rex_trajdata.py
import pandas as pd
import sys

# -----------------------------------------------------------------------------
#                      Robosample trajectory reader
#region REXTrajData ---------------------------------------------------------------
class REXTrajData:
    ''' Read Robosample trajectory data from a file '''
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self._load_data()

    def _load_data(self):
        """ Get trajectory data from a file """
        try:
            traj = md.load(self.filepath)   # Load trajectory
            df = traj.xyz.reshape(traj.n_frames, -1)  # Example: convert coords to DataFrame-like array
            return df

        except Exception as e:
            print(f"Error loading trajectory data from {self.filepath}: {e}", file=sys.stderr)
            raise
    #

    def get_dataframe(self):
        return self.df
    #

#endregion --------------------------------------------------------------------
