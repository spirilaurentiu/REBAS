import numpy as np


# Print Numpy array of any dimension
def printNumpyND(name, input_array, indent=0, label="Root"):

    spacing = "  " * indent
    
    # Print the metadata for the current level
    if input_array.ndim > 1:
        #print(f"{spacing}[Dimension {input_array.ndim} | {label} | Shape: {input_array.shape}]")
        for i, sub_arr in enumerate(input_array):
            printNumpyND(name, sub_arr, indent + 1, label=f"Index {i}")
    else:
        # We've reached the 1D leaf node (the actual data)
        # Print input_array without square brackets
        print(f"{name} {spacing} {' '.join(map(str, input_array))}")
#