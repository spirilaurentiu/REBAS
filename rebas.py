# main.py
import argparse
import os
import re
import sys
import glob
import pandas as pd
from rex_data import REXData

class REXFNManager:
    def __init__(self, directory, fn_roots):
        self.directory = directory
        self.fn_roots = fn_roots

    def extract_seed_and_type(self, filename):
        match = re.match(r"out\.(\d{7})", filename)
        if not match:
            raise ValueError(f"Filename '{filename}' does not match expected format 'out.<7-digit-seed>'")
        seed = match.group(1)
        sim_type = seed[2]  # third digit (0-based index)
        return seed, sim_type

    def process_single_file(self, filepath, seed, sim_type):
        rex = REXData(filepath)
        df = rex.get_dataframe().copy()
        df['seed'] = seed
        df['sim_type'] = sim_type
        return df

    def load_and_process_all(self):
        all_data = []
        for fn_root in self.fn_roots:
            matches = glob.glob(os.path.join(self.directory, fn_root + "*"))
            if not matches:
                print(f"Warning: No files found matching {fn_root}*", file=sys.stderr)
                continue

            for filepath in matches:

                print("Reading", filepath, "...", end = ' ', flush=True)

                try:
                    seed, sim_type = self.extract_seed_and_type(os.path.basename(filepath))
                except ValueError as e:
                    print(f"Skipping file due to naming error: {filepath}\n{e}", file=sys.stderr)
                    continue

                df = self.process_single_file(filepath, seed, sim_type)
                all_data.append(df)

                print("done.", flush=True)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def main(args):
    FNManager = REXFNManager(args.dir, args.inFNRoots)
    df = FNManager.load_and_process_all()
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Directory with data files')
    parser.add_argument('--inFNRoots', nargs='+', required=True, help='Robosample processed output file names')
    args = parser.parse_args()
    main(args)

