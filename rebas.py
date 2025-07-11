# main.py
import argparse
import os
import re
import pandas as pd
from rex_data import REXData
from rex_validation import ExperimentsValidations

def extract_seed_and_type(filename):
    match = re.match(r"out\.(\d{7})", filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected format 'out.<7-digit-seed>'")
    seed = match.group(1)
    sim_type = seed[2]  # third digit (0-based index)
    return seed, sim_type

def process_single_file(filepath, seed, sim_type):
    rex = REXData(filepath)
    df = rex.get_dataframe().copy()
    df['seed'] = seed
    df['sim_type'] = sim_type
    ev = ExperimentsValidations(df)

    print(f"\n[Simulation {seed} | Type {sim_type}] Exchange counts:")
    print(ev.count_exchanges_per_replica())

    plot_prefix = f"sim_{seed}_type_{sim_type}"
    ev.plot_replica_trajectories(f"{plot_prefix}_trajectories.pdf")
    ev.plot_pairwise_exchange_matrix(f"{plot_prefix}_matrix.pdf")

    return df

def load_and_process_all(directory, filenames):
    all_data = []
    for fn in filenames:
        print("Reading from", fn)
        filepath = os.path.join(directory, fn)
        seed, sim_type = extract_seed_and_type(fn)
        df = process_single_file(filepath, seed, sim_type)
        print(df)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def main(args):
    df = load_and_process_all(args.dir, args.inFNRoots)
    # Optionally save combined dataset or perform cross-simulation analysis
    # df.to_csv("combined_output.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Directory with data files')
    parser.add_argument('--inFNRoots', nargs='+', required=True, help='Robosample processed output file names')
    args = parser.parse_args()
    main(args)

