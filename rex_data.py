# rex_data.py
import pandas as pd
import sys

SELECTED_COLUMNS = [
    "replicaIx", "thermoIx", "wIx", "T",
    "ts", "mdsteps",
    "pe_o", "pe_n", "pe_set",
    "ke_prop", "ke_n",
    "fix_o", "fix_n",
    #"logSineSqrGamma2_o", "logSineSqrGamma2_n",
    #"etot_n", "etot_proposed",
    #"JDetLog",
    "acc", "MDorMC"
]

class REXData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self._load_data()

    def _load_data(self):
        try:
            selected_indices = []
            data_rows = []

            with open(self.filepath, 'r') as inF:
                lix = -1
                for line in inF:
                    lix += 1
                    line = line.strip()

                    if (line.startswith("REX,") or line.startswith(" REX,")) :

                        fields = line.split(',')[1:]  # remove "REX"
                        fields = [field.strip() for field in fields] # remove spaces

                        if not selected_indices:
                            header = fields
                            selected_indices = [colIx for colIx, name in enumerate(header) if name in SELECTED_COLUMNS]
                            selected_names = [header[seleIx] for seleIx in selected_indices]
                            continue  # skip header line

                        if len(fields) < max(selected_indices) + 1:
                            continue  # skip malformed

                        row = [fields[i] for i in selected_indices]
                        data_rows.append(row)

            if not data_rows:
                raise ValueError("No valid REX data rows found.")

            df = pd.DataFrame(data_rows, columns=selected_names)
            df = df.apply(pd.to_numeric, errors='ignore')
            return df

        except Exception as e:
            print(f"Error loading REX data from {self.filepath}: {e}", file=sys.stderr)
            raise

    def get_dataframe(self):
        return self.df

    def filter_by_world(self, wIx):
        return self.df[self.df['wIx'] == wIx]
