# Imports
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from datana2_0 import DataReader, Statistics, npHistIx_edge, npHistIx_data, HistIx_delta
import argparse
import copy


# -----------------------------------------------------------------------------
#                            Parse arguments
#region Parse arguments -------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None,
  help='Directory with data files')
parser.add_argument('--extension', default='',
  help='Data files extension.')
parser.add_argument('--inFNRoots', default=[], nargs='+',
    help='Robosample processed output file root names')
parser.add_argument('--Ts', default=[], nargs='+', type=float,
    help='Temperatures for every root')
parser.add_argument('--skipheadrows', default=0, type=int,
    help='Skip header rows.')
parser.add_argument('--stride', default=1, type=int,
    help='Stride for the read lines.')
parser.add_argument('--nbins', default=10, type=int,
    help='Number of histograms bins.')
args = parser.parse_args()
#endregion --------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                            Main function
#region -----------------------------------------------------------------------
# Constants
ERROR_THRESHOLD = 0.00001

def main(dir, inFNRoots, skipheadrows, nbins, Ts):
    """
    Main function to process and analyze thermodynamic data.

    Parameters:
    dir (str): Directory with data files.
    nbins (int): Number of histogram bins.
    Ts (list of float): Temperatures for every root.

    Raises:
    ValueError: If the number of temperatures does not match the number of file roots.
    """

    # Checks
    if len(inFNRoots) != len(Ts):
        raise ValueError("The number of temperatures provided does not match the number of input file roots.")

    # Conv vars
    nofRoots = len(inFNRoots)

    # Read data
    dataReader = DataReader(dir, inFNRoots, skipheadrows)
    data = dataReader.getData()
    allSeeds = dataReader.getSeeds()
    nofCols = data.shape[2]

    #region Statistics ========================================================
    statsObj = Statistics(data)
    calcHistsWay = "probability"

    # Get histograms
    histograms = statsObj.calcHistograms(nbins = nbins, way=calcHistsWay, rangePerFile=False)
    for rootIx, inFNRoot in enumerate(inFNRoots):
        for Kx_col in range(nofCols):
            edgesShouldBeZero =  np.sum(np.nanstd(histograms[rootIx, :, Kx_col, npHistIx_edge], axis = 0))
            if edgesShouldBeZero > ERROR_THRESHOLD:
                print("Histograms edges should be the same. Error =", edgesShouldBeZero)

    meanHists = np.full((nofRoots, nofCols, nbins),  np.nan)
    stdHists  = np.full((nofRoots, nofCols, nbins), np.nan)
    for rootIx, inFNRoot in enumerate(inFNRoots):
        for Kx_col in range(nofCols):
            meanHists[rootIx, Kx_col] = np.nanmean(histograms[rootIx, :, Kx_col, npHistIx_data], axis = 0)
            stdHists[rootIx, Kx_col] = np.nanstd(histograms[rootIx, :, Kx_col, npHistIx_data], axis = 0)
    #endregion ----------------------------------------------------------------

    #region Get thermodynamic quantities ======================================
    kB = 0.008314472 # kJ / (mol K) #kB = 0.0019872041 # kcal / (mol K)
    kTs = kB * Ts
    betas = 1.0 / kTs

    # Column indicators
    Kx_col_T, Kx_col_pe_o, Kx_col_pe_n, Kx_col_acc, Kx_col_rand, Kx_col_dist, Kx_col_ang, Kx_col_dih, = 0, 1, 2, 3, 4, 5, 6, 7

    # Error analysis
    beta2sinh = np.ones((nofRoots, nofCols, nbins), dtype=float)
    deltaObs = np.ones((nofRoots, nofCols, nbins), dtype=float)
    for rootIx, inFNRoot in enumerate(inFNRoots):
        beta2sinh[rootIx] *= 2.0
        beta2sinh[rootIx] *= kTs[rootIx]

        for Kx_col in range(nofCols):
            deltaObs[rootIx, Kx_col] = histograms[rootIx, 0, Kx_col, HistIx_delta]
            Sinh = np.sinh(0.5 * deltaObs[rootIx, Kx_col])
            betaSinh__ = betas[rootIx] * Sinh
            #print("betaSinh__", betaSinh__)
            beta2sinh[rootIx, Kx_col] *= (betaSinh__)
    #print("Error analysis beta2sinh - deltaObs", beta2sinh[:, 0, :] - histograms[:, 0, 0, HistIx_delta])

    # avgE, PMF
    avgObs = np.full((nofRoots, nofCols, nbins), np.nan)
    boltz = np.full((nofRoots, nofCols, nbins), np.nan)
    PartFunc = np.full((nofRoots, nofCols, nbins), np.nan)
 
    Fks = np.full((nofRoots, nofCols, nbins), np.nan)
    multiplicity = 2 # true only for energy 
    Omega = np.full((nofRoots, nofCols, nbins), multiplicity)
    Sks = np.full((nofRoots, nofCols, nbins), np.nan)

    for rootIx, inFNRoot in enumerate(inFNRoots):
        for Kx_col in range(nofCols):

            # Average energy
            avgObs[rootIx, Kx_col] =  histograms[rootIx, 0, Kx_col_pe_o, npHistIx_edge]

            # Boltzmann factor
            boltz[rootIx, Kx_col] = np.exp((-1.0 * betas[rootIx]) * avgObs[rootIx, Kx_col])

            # Partition function
            PartFunc[rootIx, Kx_col] = 1.0 / meanHists[rootIx, Kx_col]
            PartFunc[rootIx, Kx_col] *= Omega[rootIx, Kx_col]
            PartFunc[rootIx, Kx_col] *= boltz[rootIx, Kx_col]
            
            PartFunc[rootIx, Kx_col] *= beta2sinh[rootIx, Kx_col]
            #PartFunc[rootIx, Kx_col] /= deltaObs[rootIx, Kx_col]
            
            if(Kx_col == Kx_col_pe_o):
                print(f"Partition function (Q) for column Kx_col_pe_o: {PartFunc[rootIx, Kx_col]}")

            # PMF
            if calcHistsWay == "probability":
                Fks[rootIx, Kx_col] = (-1.0 * kTs[rootIx] ) * np.log(  meanHists[rootIx, Kx_col]  )
            else:
                print("PMF not calculated.")

            # Local entropy
            Sks[rootIx, Kx_col] = kB * np.log(Omega[rootIx, Kx_col]) # true only for energy

    #endregion ----------------------------------------------------------------

    # Free energy
    #np.set_printoptions(precision=7, suppress=True)
    A_local = np.empty((nofRoots, nofCols, nbins)) * np.nan
    for rootIx, inFNRoot in enumerate(inFNRoots):
        for Kx_col in range(nofCols):
            A_local[rootIx, Kx_col] = -kTs[rootIx] * np.log(PartFunc[rootIx, Kx_col])
    #print("A_local", A_local)
    #region Plot ==============================================================
    plotPEDist, plotPMF, plotAcceptance, plotGeometry = True, False, False, False

    cmap = cm.get_cmap('rainbow', nofRoots)
    rootColors = [cmap(ix) for ix in range(nofRoots)]
    rootColors = ['darkblue', 'blue', 'cyan', 'maroon', 'red', 'orange']
    labels = ['Ref 300 K', 'REMC 300 K', 'REBAS 300 K', 'Ref 600 K', 'REMC 600 K', 'REBAS 600 K']

    # Potential energy
    if plotPEDist:
        fig, ax = plt.subplots()
        for rootIx, inFNRoot in enumerate(inFNRoots):
            X__ = histograms[rootIx, 0, Kx_col_pe_o, npHistIx_edge]
            Y__ = meanHists[rootIx, Kx_col_pe_o]

            #Y__ = PartFunc[rootIx, Kx_col_pe_o]

            # Horizontal shift to prevent overlap
            shift = 0.2 * (rootIx - len(inFNRoots)/2)

            YErr = stdHists[rootIx, Kx_col_pe_o]

            # 1. Plot the actual data (lines/markers) at the original X
            ax.plot(X__, Y__,
                    #label=inFNRoot + "_" + str(allSeeds[rootIx]),
                    label=labels[rootIx],
                    color=rootColors[rootIx])

            # 2. Plot ONLY the error bars at the shifted X 
            ax.errorbar(X__ + shift, Y__, yerr=YErr, 
                        fmt='none', # ensures no new markers or lines are drawn
                        ecolor=rootColors[rootIx],
                        capsize=3)

            #ax.errorbar(X__ + shift, Y__, yerr=YErr,
            #            #label=inFNRoot + "_" + str(allSeeds[rootIx]),
            #            label = labels[rootIx],
            #            color=rootColors[rootIx])

            ax.legend()
            ax.set_xlabel("Potential energy (kJ/mol)")
            ax.set_ylabel("Probability density")
            ax.set_title("Potential energy distribution")
        plt.show()

    
    def Kolmogorov_Smirnov_Test(sample1, sample2):
        from scipy.stats import ks_2samp
        # Flatten the arrays to 1D
        s1 = sample1.flatten()
        s2 = sample2.flatten()
        
        # Remove NaNs manually
        s1_clean = s1[~np.isnan(s1)]
        s2_clean = s2[~np.isnan(s2)]
        
        # Check if we have enough data left to perform a test
        if len(s1_clean) == 0 or len(s2_clean) == 0:
            return np.nan, np.nan
            
        # Run the test without the nan_policy argument
        ks_statistic, p_value = ks_2samp(s1_clean, s2_clean)
        
        return ks_statistic, p_value

    print("KS", end=' ')
    for rootIx, inFNRootI in enumerate(inFNRoots):
        print(rootIx, end=' ')
    print()
    for rootIx, inFNRootI in enumerate(inFNRoots):
        print(rootIx, end=' ')
        for rootJx, inFNRootJ in enumerate(inFNRoots):
            #if rootIx >= rootJx:
            #    continue  # Avoid redundant comparisons and self-comparison
    
            #print(f"Comparing root {rootIx} with root {rootJx}:")
            ks_statistic, p_value = Kolmogorov_Smirnov_Test(meanHists[rootIx, Kx_col_pe_o], meanHists[rootJx, Kx_col_pe_o])
            #print(f"Kolmogorov-Smirnov Test: KS Statistic = {ks_statistic}, p-value = {p_value}")
            #print(p_value, end=' ' )
            print(ks_statistic, end=' ' )
        print()

    exit(0)
#endregion --------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                            Main call
#region -----------------------------------------------------------------------
if __name__=="__main__":

    if not args.inFNRoots:
        parser.error("Output filenames roots are required.")

    Ts = np.array(args.Ts, dtype=float)

    main(args.dir, args.inFNRoots, args.skipheadrows, args.nbins, Ts)
#endregion --------------------------------------------------------------------
