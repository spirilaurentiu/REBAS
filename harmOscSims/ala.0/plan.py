from RobosampleAnalysis import *
from alaGlobalExperimentParams import *

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--actions', default=["simulateBash"], nargs='+',
	help='Action: simulateBash, generateInps, processOuts')
args = parser.parse_args()

# ------------------------------
# Generate simulate.bash
# ==============================
inpFNs, outFNs, logFNs = [], [], []
if  (("simulateBash" == args.actions) or ("simulateBash" in args.actions)) or \
    (("generateInps" == args.actions) or ("generateInps" in args.actions)) or \
    (("getOuts" == args.actions)  or ("getOuts" in args.actions))  or \
    (("processOuts" == args.actions)  or ("processOuts" in args.actions)):
    for subTypeIx in range(len(subTypes)):
        if  (("simulateBash" == args.actions) or ("simulateBash" in args.actions)):
            print("#", subTypes[subTypeIx])
        for TypeIx in range(len(Types)):
            for rep in range(nofRepeats):

                seed  = power2seed(molIx, TypeIx, subTypeIx, rep)

                inpOutKernel = (molDir + '.' + Types[TypeIx]
                                + '.' + subTypes[subTypeIx]
                                + '.' + str(seed))

                inpFN = 'inp' + '.' + inpOutKernel
                outFN = "out" + '.' + inpOutKernel
                command = "nohup" + " " + exeFN + " " + inpFN + " > " + outFN + " " + "2>&1&"

                inpFNs.append(inpFN)
                outFNs.append(outFN)
                logFNs.append("log." + str(seed))

                if  (("simulateBash" == args.actions) or ("simulateBash" in args.actions)):
                    if((subTypes[subTypeIx] != "B") and (subTypes[subTypeIx] != "B")): print("#", end = ' ')
                    print(command)

# ------------------------------
# Generate inputs
# ==============================
if ("generateInps" == args.actions) or ("generateInps" in args.actions):

    for fi, inpFN in enumerate(inpFNs):
        seed = inpFN.split(".")[-1]
        (molIx, TypeIx, subTypeIx, batchIx, rep) = [int(digit) for digit in seed]
        #print(molIx, TypeIx, subTypeIx, batchIx, rep)

        varTypeChunks = []
        varTypeChunks.append("""
# Molecule information
MOLECULES {foMolName}
""".format(foMolName = molDir))

        varTypeChunks.append("""
FLEXFILE ligand.cart.flex ligand.td.flex {w1Flex}
""".format(w1Flex = "ligand." + subTypes[subTypeIx] + ".flex"))

        varTypeChunks.append("""
RUN_TYPE {foType}
REX_FILE {foTrexFile}
DISTORT_OPTION 0 0 {distortType}
INTEGRATORS VV VV {integrator}
""".format(foType = Types[TypeIx], foTrexFile = trexFNs[TypeIx],
    distortType = distortTypes[TypeIx],
    integrator = integrators[TypeIx]))

        varTypeChunks.append("""
REX_SWAP_EVERY 1
REX_SWAP_FIXMAN 0
""".format())

        varTypeChunks.append("""
ROUNDS {foNofRounds}
SEED {foSeed} {foSeed}
""".format(foSeed = seed, foNofRounds = nofRounds))

        varTypeChunk = ""
        for vi, vtc in enumerate(varTypeChunks):
            varTypeChunk += vtc

        inpContent = constInpChunk + varTypeChunk + varSimChunk

        inpF = open(inpFN, "w")
        inpF.write(inpContent)
        inpF.close()
        
        #print(varTypeChunks[4])

        """ if(fi > 1):
            exit(0) """





