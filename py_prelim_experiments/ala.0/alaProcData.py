# Imports
import io, os, sys, types, glob
from itertools import chain
import numpy.ma as ma

import numpy as np
from scipy import stats
import scipy

import mdtraj as md

import matplotlib
import matplotlib.pyplot as plt

# Debug
#import pdb
#pdb.set_trace()

from getHistograms import *


# My imports
myScriptsFolder = "/home/pcuser/scripts"

if myScriptsFolder not in sys.path:
    sys.path.append(myScriptsFolder)
    
paperFolder = "/home/pcuser/0Work/robo/tfep/"
if paperFolder not in sys.path:
    sys.path.append(paperFolder)

#from RobosampleAnalysis import *
import RobosampleAnalysis as RA
    
from alaGlobalExperimentParams import *

# Argument parser
parser = argparse.ArgumentParser()
if(1):
    parser.add_argument('--actions', default=["processOut", "processLog", "test"], nargs='+',
        help='Action: simulateBash, generateInps, processOuts')
    parser.add_argument('--skipheadrows', default=0, type=int,
        help='Skip first N values')
    
    # TypeIx, subTypeIx, repeatIx, roundIx, replicaIx, worldIx
    parser.add_argument('--TypeIx', default=1, type=int,
        help='Type = HMC, HMCS, REX, RENS')
    parser.add_argument('--subTypeIx', default=0, type=int,
        help='subType = BA')
    parser.add_argument('--repIx', default=0, type=int,
        help='repIx = repeat')
    parser.add_argument('--batchIx', default=-1, type=int,
        help='repIx = repeat')
    parser.add_argument('--replicaIx', default=0, type=int,
        help='replicaIx = replica')
    parser.add_argument('--worldIx', default=0, type=int,
        help='replicaIx = replica')

    parser.add_argument('--xmin', default=-666.0, type=float,
        help='Min x')
    parser.add_argument('--xmax', default=+666.0, type=float,
        help='Max x')
args = parser.parse_args()

# Load data
outD = np.load("outD.npy")
logD = np.load("logD.npy")


# Filtering function
def thermoFilter(dataMaskIndicator):
    return np.abs(dataMaskIndicator - 1000.0) > 0.05

# This takes the raw data and computes histograms of every column:

###############################################################################
# OUTPUT HISTOGRAMS
###############################################################################
# T, pe_o, pe_n, ke_prop, ke_n, fix_o, fix_n, acc

wantTypes = [HMC_, HMCS_, REX_, RENS_]
wantSubTypes = [BA_]
repeats = np.array(range(nofRepeats), dtype = int)
wantReplicas = [0, 1]
wantWorlds = [0, 1, 2]

# Number of bins for histograms
nbins = 300

if "processOut" in args.actions:

    outHists, rangeLeft, rangeRight = getAll_1DHistograms(outD, nbins,
        wantTypes, wantSubTypes, repeats, wantReplicas, wantWorlds,
        filterFunc = thermoFilter, indicatorIndex = RA.Ix_T)

# Save
if "processOut" in args.actions:

    outDict = {'nbins':nbins, 'rangeLeft':rangeLeft, 'rangeRight':rangeRight, 'histData':outHists}
    np.save("hists.npy", outDict)

###############################################################################
# LOG HISTOGRAMS
###############################################################################
# T, wIx, NU, acc, pe_o, pe_n, ke_prop, ke_n, fix_o, fix_n, fix_x, geom1, geom2, geom3

wantTypes = [HMC_, HMCS_, REX_, RENS_]
repeats = np.array(range(nofRepeats), dtype = int)
wantSubTypes = [BA_]
wantReplicas = [0, 1]
wantWorlds = [0, 1, 2]

# Number of bins for histograms
nbins = 300

if "processLog" in args.actions:

    logHists, rangeLeft, rangeRight = getAll_1DHistograms(logD, nbins,
        wantTypes, wantSubTypes, repeats, wantReplicas, wantWorlds,
        filterFunc = thermoFilter, indicatorIndex = RA.logIx_T)    

# Save
if "processLog" in args.actions:

    logDict = {'nbins':nbins, 'rangeLeft':rangeLeft, 'rangeRight':rangeRight, 'histData':logHists}
    np.save("logHists.npy", logDict)

###############################################################################
# LOG GEOMETRIC DATA
###############################################################################
# T, wIx, NU, acc, pe_o, pe_n, ke_prop, ke_n, fix_o, fix_n, fix_x, geom1, geom2, geom3

wantTypes = [HMC_, HMCS_, REX_, RENS_]
repeats = np.array(range(nofRepeats), dtype = int)
wantSubTypes = [BA_]
wantReplicas = [0, 1]
wantWorlds = [0, 1, 2]

# Indicator for filtering data
Indexes = [13, 14]
nbins = 300
range = np.array( [[-np.pi, np.pi], [-np.pi, np.pi]] )

log2DHists, rangeLeft, rangeRight = getAll_2DHistograms(logD, Indexes, nbins, range,
    wantTypes, wantSubTypes, repeats, wantReplicas, wantWorlds,
    filterFunc = thermoFilter, indicatorIndex = RA.logIx_T)

# Save
if "processLogGeom" in args.actions:

    logGeom = {'logPhiPsi':log2DHists}
    np.save("logGeom.npy", logGeom)

###############################################################################
# Test what we saved 
###############################################################################
# hist shape:
# (nofTypes, nofSubTypes, nofRepeats, nofReplicas, nofWorldsPerReplica, nbins)

if "test" in args.actions:

    # Load
    outDict = np.load("hists.npy", allow_pickle=True)
    outHists = outDict.item().get('histData')
    rangeLeft = outDict.item().get('rangeLeft')
    rangeRight = outDict.item().get('rangeRight')
    nbins = outDict.item().get('nbins')

    logDict = np.load("logHists.npy", allow_pickle=True)
    logHists = logDict.item().get('histData')
    logRangeLeft = logDict.item().get('rangeLeft')
    logRangeRight = logDict.item().get('rangeRight')
    logNbins = logDict.item().get('nbins')

    logGeomDict = np.load("logGeom.npy", allow_pickle=True)
    log2dHists = logGeomDict.item().get('logPhiPsi')

    # Test plot
    wantTypes = [HMC_, HMCS_, REX_, RENS_]
    wantSubTypes = [BA_]
    wantIndexes, wantLogIndexes = [RA.Ix_pe_o], [RA.logIx_pe_o]
    replicaIx, worldIx = 0, 0

    f1, outAx = plt.subplots(nrows = 1, ncols = 1)
    f2, logAx = plt.subplots(nrows = 1, ncols = len(wantLogIndexes))
    f3, logGeomAx = plt.subplots(nrows = 1, ncols = 1)

    # SUBTYPES: BA
    for subTypeIx_i, subTypeIx in enumerate(wantSubTypes):
            
            # TYPES: 'HMC', 'REX', 'REBAS'
            for TypeIx_i, TypeIx in enumerate(wantTypes):

                #for repIx_i, repIx in enumerate(repeats):
                for repIx_i, repIx in enumerate([0]):

                    # Output indeces
                    label = "out" + subTypes[subTypeIx] + "_" + Types[TypeIx] + "_" + str(repIx)
                    for Index_i, Index in enumerate(wantIndexes):

                        whichHist = (TypeIx, subTypeIx, repIx, replicaIx, worldIx, Index)
                        X = np.linspace(rangeLeft[subTypeIx, Index], rangeRight[subTypeIx, Index], nbins)
                        Y = outHists[whichHist]

                        outAx.plot(X, Y, label = label)

                    # Log indeces
                    label = "log" + subTypes[subTypeIx] + "_" + Types[TypeIx] + "_" + str(repIx)
                    for Index_i, Index in enumerate(wantLogIndexes):

                        whichHist = (TypeIx, subTypeIx, repIx, 0, 0, Index)

                        X = np.linspace(logRangeLeft[subTypeIx, Index], logRangeRight[subTypeIx, Index], logNbins)
                        Y = logHists[whichHist]

                        logAx.plot(X, Y, label = label)
                        logAx.legend()

            
                    logGeomAx.contour(np.linspace(-3.14, 3.14, 300), np.linspace(-3.14, 3.14, 300),
                        np.transpose(log2dHists[TypeIx, subTypeIx, repIx, replicaIx, worldIx]))


    outAx.legend()
    plt.show()
