# rex_data.py
import pandas as pd
import sys

REX_COLUMNS = [
    "replicaIx", "thermoIx", "wIx", "T", "boostT", "ts", "mdsteps",
    "DISTORT_OPTION", "NU", "nofSamples", "pe_o", "pe_n", "pe_set",
    "ke_prop", "ke_n", "fix_o", "fix_n", "logSineSqrGamma2_o", "logSineSqrGamma2_n",
    "etot_n", "etot_proposed", "JDetLog", "acc", "MDorMC", "unif", "HARDMOLNAME"
]

class REXData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self._load_data()

    def _load_data(self):
        with open(self.filepath, 'r') as f:
            all_lines = [line.strip() for line in f if line.strip()]

        data_lines = [line for line in all_lines if line.startswith("REX,")]

        if not data_lines:
            raise ValueError("No REX data found in the file.")

        header = [h.strip() for h in data_lines[0].split(',')[1:]]  # Skip 'REX'
        parsed_data = [line.split(',')[1:] for line in data_lines[1:]]

        # Pad rows if needed
        max_len = max(len(header), max(len(row) for row in parsed_data))
        while len(header) < max_len:
            header.append(f"col{len(header)}")
        for row in parsed_data:
            while len(row) < max_len:
                row.append('')

        df = pd.DataFrame(parsed_data, columns=header)
        df = df.apply(pd.to_numeric, errors='ignore')
        return df

    def get_dataframe(self):
        return self.df

    def filter_by_temperature(self, temp):
        return self.df[self.df['T'] == temp]

