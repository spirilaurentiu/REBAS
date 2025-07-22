# rex_data.py
import pandas as pd
import sys

# -----------------------------------------------------------------------------
#                      Robosample output reader
#region REXData ---------------------------------------------------------------
class REXData:
    ''' 
    '''
    def __init__(self, filepath, SELECTED_COLUMNS):
        self.SELECTED_COLUMNS = SELECTED_COLUMNS
        self.filepath = filepath
        self.df = self._load_data()

    def _load_data(self):
        """ Get data from a file """
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
                            selected_indices = [colIx for colIx, name in enumerate(header) if name in self.SELECTED_COLUMNS]
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
    #

    def get_dataframe(self):
        return self.df
    #

    def filter_by_world(self, wIx):
        return self.df[self.df['wIx'] == wIx]
    #
#endregion --------------------------------------------------------------------
