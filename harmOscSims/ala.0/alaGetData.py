# Imports
import io, os, sys, types, glob
from itertools import chain

import numpy as np
from scipy import stats
import scipy

import mdtraj as md

# My imports
myScriptsFolder = "/home/pcuser/scripts"

if myScriptsFolder not in sys.path:
    sys.path.append(myScriptsFolder)
    
paperFolder = "/home/pcuser/0Work/robo/tfep/"
if paperFolder not in sys.path:
    sys.path.append(paperFolder)

import RobosampleAnalysis as RA
    
from alaGlobalExperimentParams import *

# Allocate output data
outD = np.zeros((nofTypes, nofSubTypes, nofRepeats, nofRounds, nofReplicas, nofWorldsPerReplica, RA.nofIndexes),
                dtype = float) * np.nan

# Allocate log data
logD = np.zeros((nofTypes, nofSubTypes, nofRepeats, nofRounds, nofReplicas, nofWorldsPerReplica, RA.nofLogIndexes),
                dtype = float) * np.nan

# Search out and log files
logFNs = glob.glob("log.?????")
outFNs = glob.glob("out.ala1.*")

print(logFNs)
print(outFNs)

for fi, outFN in enumerate(outFNs):

    FNChuncks = outFN.split(".")
    isREX = ((FNChuncks[2][0:3] != 'HMC'))
    inpFN = "inp." + '.'.join(FNChuncks[1:])
    seed = FNChuncks[-1]
    logFN = 'log.' + str(seed)

    (molIx, TypeIx, subTypeIx, batchIx, rep) = [int(digit) for digit in seed]
    print(outFN, molIx, TypeIx, subTypeIx, batchIx, rep)
    print(logFN, molIx, TypeIx, subTypeIx, batchIx, rep)

    if os.path.isfile(outFN) and os.path.isfile(logFN):

        # Get output data
        RA.getOutputData(outFN, outD[TypeIx][subTypeIx][rep], REX = isREX)

        # Get log data
        RA.getLogData(logFN, logD[TypeIx][subTypeIx][rep], nofReplicas, nofWorldsPerReplica, REX = isREX)

""" # nofTypes, nofSubTypes, nofRepeats, nofRounds, nofReplicas, nofWorldsPerReplica, RA.nofIndexes
print((nofTypes, nofSubTypes, nofRepeats, nofRounds, nofReplicas, nofWorldsPerReplica, RA.nofIndexes))

seed = "10000"
(molIx, TypeIx, subTypeIx, batchIx, rep) = [int(digit) for digit in seed]
roundi, replicai, wIx = 0, 0, 2
data = outD[TypeIx, subTypeIx, rep, :, :, :]

print("outD 10000")
print(data) """

# Save
np.save("outD.npy", outD)
np.save("logD.npy", logD)
