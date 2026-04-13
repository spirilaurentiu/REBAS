# Process output files from REX simulations:
# extract relevant information and write it to new files
# with a specified suffix.

#region Imports
import os
import argparse
import glob
#endregion # Imports

#region Parse arguments 
import argparse
import sys
parser = argparse.ArgumentParser(
    description="Process REX output files to extract only necessary columns and write it to new files with a specified suffix."
)
parser.add_argument('--dir', default=None, help='Directory with input data files')
parser.add_argument('--inFNRoots', nargs='+', required=True, help='Name roots for input data files')
parser.add_argument('--procSuffix', help='Suffix to add to output files')
parser.add_argument('--dry', action='store_true', help='If set, print results to console instead of writing to output file')
args = parser.parse_args()
#endregion # Parse arguments

# Get all files in the specified directory that match the given roots
def find_files(dir, inFNRoots, heavyPrinting=False):
    """
    Find all files in the specified directory that match the given roots.

    Args:
        dir (str): The directory to search in.
        inFNRoots (list of str): The list of filename roots to match.

    Returns:
        list of str: The list of matching file paths.
    """
    fullFNs = []
    dirs = []
    FNs = []
    suffixes = []
    for root in inFNRoots:
        matched = glob.glob(os.path.join(dir, root + '*'))
        fullFNs.extend(matched)
        dirs = [os.path.dirname(FN) for FN in fullFNs]
        
        FNs_all = [FN.replace('/', ' ').split()[-1] for FN in fullFNs]

        #print([FN for FN in FNs_all if (FN.replace('.', ' ').split()[1]).isdigit() and (FN.replace('.', ' ').split()[0] == root)])
        FNs = [FN for FN in FNs_all if (FN.replace('.', ' ').split()[1]).isdigit() and (FN.replace('.', ' ').split()[0] == root)]

        FNs_split = [FN.replace('.', ' ').split() for FN in FNs]
        suffixes = [FN[-1] for FN in FNs_split]
        recovered_roots = [FN[0] for FN in FNs_split]

        if heavyPrinting:
            print("matched:", matched)
            print("fullFNs:", fullFNs)
            print("dirs:", dirs)
            print("FNs:", FNs)
            print("FNs_split:", FNs_split)
            print("suffixes:", suffixes)

        if any([len(FN) > 2 for FN in FNs_split]):
            pass

        if any(recovered_root != root for recovered_root in recovered_roots):
            print("Roots should be of type 'root.suffix' and the recovered root should match the original root.", file=sys.stderr)
            print(f"root=|{root}|, recovered_roots={recovered_roots}", file=sys.stderr)
            sys.exit(1)

    return (FNs, recovered_roots, suffixes)

# Process file
def process_rex(inFN, outFN, StartsWith_Pattern="REXdetails", dry = True):
    """ Process the input file and write the results to the output file.

    Args:
        inFN (str): The input file name.
        outFN (str): The output file name.
        StartsWith_Pattern (str): The pattern to search for in each line.
    """

    if(dry == False):
       print("Error: Still in development. Writing files not allowed yet.", file=sys.stderr)
       return

    if not os.path.exists(inFN):
        print(f"Error: Input file {inFN} not found.", file=sys.stderr)
        return

    if os.path.exists(outFN):
        print(f"Error: Output file {outFN} already exists.", file=sys.stderr)
        return

    with open(inFN, 'r') as infile:

        outfile = None
        if not dry: outfile = open(outFN, 'w')

        lIx = -1
        pIx = -1

        # Get any line
        for line in infile:
            lIx += 1

            # Remove commas and leading/trailing whitespace (sed 's/^ //' and 's/,//g')
            clean_line = line.strip().replace(',', '')
            clean_line = ' '.join(clean_line.split())  # Replace multiple spaces with a single space

            # print(clean_line)
            # print("|" + StartsWith_Pattern + ' ' + "|")
            # print("|" + line[0:len(StartsWith_Pattern)+1] + "|")

            # Get line that starts with the specified pattern
            if (StartsWith_Pattern + ' ') == clean_line[0:len(StartsWith_Pattern)+1]:
                pIx += 1

                # Split by whitespace
                cols = clean_line.split()

                if StartsWith_Pattern == "REXdetails":
                    #if pIx == 0: continue

                    # Get columns for REXdetails
                    try:
                        thermoState_C = cols[1]
                        thermoState_H = cols[2]
                        replica_X   = cols[3]
                        replica_Y   = cols[4]
                        beta_C      = cols[5]
                        beta_H      = cols[6]
                        uC_Xset     = cols[7]
                        uH_Yset     = cols[8]
                        uH_Xset     = cols[9]
                        uC_Yset     = cols[10]
                        uC_Xtau     = cols[11]
                        uH_Ytau     = cols[12]
                        uH_Xtau     = cols[13]
                        uC_Ytau     = cols[14]
                        ref_uC_Xset = cols[15]
                        ref_uH_Yset = cols[16]
                        ref_uH_Xset = cols[17]
                        ref_uC_Yset = cols[18]
                        ref_uC_Xtau = cols[19]
                        ref_uH_Ytau = cols[20]
                        ref_uH_Xtau = cols[21]
                        ref_uC_Ytau = cols[22]
                        lnJac_X     = cols[23]
                        lnJac_Y     = cols[24]
                        W_X    = cols[25]
                        W_Y    = cols[26]
                        s_X    = cols[27]
                        s_Y    = cols[28]
                        s_X_1  = cols[29]
                        s_Y_1  = cols[30]
                        qC_s_X  = cols[31]
                        qH_s_Y  = cols[32]
                        qH_s_X_1  = cols[33]
                        qC_s_Y_1  = cols[34]
                        ETerm_equil  = cols[35]
                        WTerm    = cols[36]
                        correctionTerm  = cols[37]
                        acc  = cols[38]
                        unif = cols[39]

                        if not dry:
                            outfile.write(f"REXdetails, {thermoState_C}, {thermoState_H}, {replica_X}, {replica_Y}\n")
                        else:
                            print(f"REXdetails, {thermoState_C}, {thermoState_H}, {replica_X}, {replica_Y}")
                            if(pIx > 10): break

                    except IndexError:
                        continue
                
                elif StartsWith_Pattern == "REX":
                    #if pIx == 0: continue

                    #print(f"Processing line {pIx}: {cols}")

                    # Get columns for REX,
                    try:
                        replicaIx = cols[1]
                        thermoIx  = cols[2]
                        wIx       = cols[3]
                        T         = cols[4]
                        boostT    = cols[5]
                        ts        = cols[6]
                        mdsteps   = cols[7]
                        DISTORT_OPTION = cols[8]
                        NU        = cols[9]
                        nofSamples= cols[10]
                        pe_o      = cols[11]
                        pe_n      = cols[12]
                        pe_set    = cols[13]
                        ke_prop   = cols[14]
                        ke_n      = cols[15]
                        fix_o     = cols[16]
                        fix_n     = cols[17]
                        logSineSqrGamma2_o = cols[18]
                        logSineSqrGamma2_n = cols[19]
                        etot_n    = cols[20]
                        etot_proposed  = cols[21]
                        JDetLog   = cols[22]
                        acc       = cols[23]
                        MDorMC    = cols[24]
                        unif = cols[25] if len(cols) > 25 else "N/A"

                        #print(f"Try {pIx} {cols}")

                        if not dry:
                            #outfile.write(f"REX, {replicaIx}, {thermoIx}, {wIx}, {pe_o}, {acc}\n")
                            outfile.write(f"REX, {replicaIx}, {thermoIx}, {wIx}, {pe_o}, {pe_n}, {pe_set}, {ke_prop}, {ke_n}, {JDetLog}, {acc}\n")
                        else:
                            #print(        f"REX, {replicaIx}, {thermoIx}, {wIx}, {pe_o}, {acc}")
                            print(        f"REX, {replicaIx}, {thermoIx}, {wIx}, {pe_o}, {pe_n}, {pe_set}, {ke_prop}, {ke_n}, {JDetLog}, {acc}")
                            if(pIx > 10): break

                    except IndexError:
                        continue

# # end function process_rex


if __name__ == "__main__":

    FNs, recovered_roots, suffixes = find_files(args.dir, args.inFNRoots, heavyPrinting=False)

    print("FNs:", FNs)
    print("recovered_roots:", recovered_roots)
    print("suffixes:", suffixes)

    for rIx, reroot in enumerate(recovered_roots):
        inFN = os.path.join( args.dir, f"{reroot}.{suffixes[rIx]}")

        print(inFN, FNs[rIx])

        outFN = os.path.join(args.dir, f"{reroot}.{args.procSuffix}.{suffixes[rIx]}")
        print(inFN, outFN)

        process_rex(inFN, outFN, StartsWith_Pattern="REX", dry=args.dry)



