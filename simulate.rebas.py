#region: Imports
import sys
sys.path.append("/home/laurentiu/git6/Robosample/bin")
import flexor
import mdtraj as md
import argparse
import robosample
#import batstat
import numpy as np
#endregion

mobilityMap = {
        "Cartesian": robosample.BondMobility.Translation,
        "Pin": robosample.BondMobility.Torsion,
        "Torsion": robosample.BondMobility.Torsion,
        "Slider": robosample.BondMobility.Slider,
        "AnglePin": robosample.BondMobility.AnglePin,
        "BendStretch": robosample.BondMobility.BendStretch,
        "Spherical": robosample.BondMobility.Spherical,
        "Cylinder": robosample.BondMobility.Cylinder,
        "OrthoSpherical": robosample.BondMobility.OrthoSpherical
}

# Read the flexibilities from a file
def getFlexibilitiesFromFile(flexFile):
        """
        Backward compatibility: Reads the flexibilities from a file.
        """
        flexibilities = []
        with open(flexFile, 'r') as f:
                for line in f:
                        if line[0] == '#':
                                continue
                        tokens = line.split()
                        if(len(tokens) >= 3):
                                if(tokens[2] != "Weld"):
                                        aIx_1 = int(tokens[0])
                                        aIx_2 = int(tokens[1])
                                        mobility = mobilityMap[tokens[2]]
                                        flexibilities.append( robosample.BondFlexibility(aIx_1, aIx_2, mobility) )
        return flexibilities
#

# Print the flexibilities
def printFlexibilities(flexibilities):
        """
        Prints the flexibilities.
        """
        for flexIx, flex in enumerate(flexibilities):
                print(flexibilities[flexIx].i, flexibilities[flexIx].j, flexibilities[flexIx].mobility)
#

#region: Parse the arguments
# python simulate.py baseName prmtop rst7 equil_steps prod_steps write_freq temperature_init seed
# python simulate.py 1a1p ../data/1a1p/1a1p.prmtop ../data/1a1p/1a1p.rst7 1000 10000 300.00 666
parser = argparse.ArgumentParser(description='Process PDB code and seed.')
parser.add_argument('--name', type=str, help='Name of the simulation.')
parser.add_argument('--top', type=str, help='Relative path to the .prmtop file.')
parser.add_argument('--rst7', type=str, help='Relative path to the .rst7 file.')
parser.add_argument('--rstDir', type=str, help='Restart directory.')
parser.add_argument('--equilSteps', type=int, help='The number of equilibration steps.')
parser.add_argument('--prodSteps', type=int, help='The number of production steps.')
parser.add_argument('--writeFreq', type=int, help='CSV and DCD write frequency.')
parser.add_argument('--baseTemperature', default=300.00, type=float, help='Temperature of the first replica.')
parser.add_argument('--baseTdiff', type=float, default=10.0, help='Temperature difference between the first two replicas.')
parser.add_argument('--nofReplicas', type=int, default=1, help='Number of replicas.')
parser.add_argument('--runType', type=str, help='Run type: DEFAULT, REMC, RENEMC, RENE.')
parser.add_argument('--seed', type=int, help='The seed.')
parser.add_argument('--flexFNs', type=str, nargs='+', default=[], help='The flexFNs.')
args = parser.parse_args()
#endregion

# Create robosample context
run_type = getattr(robosample.RunType, args.runType)
context = robosample.Context(args.name, args.seed, 0, 1, run_type, 1, 0)

# Set parameters
context.setPdbRestartFreq(0) # WRITE_PDBS
context.setPrintFreq(args.writeFreq) # PRINT_FREQ
context.setNonbonded(0, 1.2)
context.setGBSA(0)
context.setVerbose(True)

# Load system
context.loadAmberSystem(args.top, args.rst7)

# Prepare flexor generator
mdtrajObj = md.load(args.rst7, top=args.top)
flexorObj = flexor.Flexor(mdtrajObj)

# Add default Openmm Cartesian world
nofWorlds = 0

flexes_Cart = flexorObj.create(range="all", subset=["all"], jointType="Cartesian")
context.addWorld(False, 1, robosample.RootMobility.CARTESIAN, flexes_Cart, True, False, 0)
nofWorlds += 1

# Add worlds
for flexFNIx, flexFN in enumerate(args.flexFNs):
        flexibilities = getFlexibilitiesFromFile(flexFN)
        printFlexibilities(flexibilities)
        context.addWorld(False, 1, robosample.RootMobility.WELD, flexibilities, True, False, 0)
        nofWorlds += 1

# Add samplers
sampler = robosample.SamplerName.HMC # rename to type
thermostat = robosample.ThermostatName.ANDERSEN
context.getWorld(0).addSampler(sampler, robosample.IntegratorType.OMMVV, thermostat, False)

for worldIx in range(1, nofWorlds):
        context.getWorld(worldIx).addSampler(sampler, robosample.IntegratorType.VERLET, thermostat, True)

# Replica exchange
nof_replicas = args.nofReplicas
temperatures = np.zeros(nof_replicas, dtype=np.float64)
Tratio = (args.baseTemperature + args.baseTdiff) / args.baseTemperature
for replIx in range(nof_replicas):
    temperatures[replIx] = args.baseTemperature * (Tratio**replIx)

worldIndexes = range(nofWorlds)
accept_reject_modes = nofWorlds * [robosample.AcceptRejectMode.MetropolisHastings]

# -----------------------------------------------------------
# ------------------ TIMESTEPS and MDSTEPS ------------------
# -----------------------------------------------------------
# region sensitive parameters
timesteps = nofWorlds * [0.0007]
mdsteps = nofWorlds * [10]
distort_options = nofWorlds * [0]
distort_args = nofWorlds * ["0"]
flow = nofWorlds * [0]
work = nofWorlds * [0]

# World 0 # Cartesian
if nofWorlds > 0:
        timesteps[0] = 0.001
        mdsteps[0] = 500

# World 1 TD
if nofWorlds > 1:
        timesteps[1] = 0.008
        mdsteps[1] = 10


# World 2 mid TD
if nofWorlds > 2:
        timesteps[2] = 0.2
        mdsteps[2] = 5

# World 3 mid TD
if nofWorlds > 3:
        timesteps[3] = 0.2
        mdsteps[3] = 5

# World 4 mid TD
if nofWorlds > 4:
        timesteps[4] = 0.2
        mdsteps[4] = 5

# World 5 mid TD
if nofWorlds > 5:
        timesteps[5] = 0.2
        mdsteps[5] = 5

# World 6 mid TD
if nofWorlds > 6:
        timesteps[6] = 0.2
        mdsteps[6] = 5

# World 7 mid TD
if nofWorlds > 7:
        timesteps[7] = 0.2
        mdsteps[7] = 5


# World 8 # BAT stats
if nofWorlds > 8:
        timesteps[8] = 0.0007
        mdsteps[8] = 0

# World 9 # BAT REBAS
if nofWorlds > 9:
        timesteps[9] = 0.0007
        mdsteps[9] = 0
        accept_reject_modes[9] = robosample.AcceptRejectMode.AlwaysAccept
        distort_options[9] = -6
        distort_args[9] = "deterministic"

#endregion
# -----------------------------------------------------------

# Add replicas
integrators = [robosample.IntegratorType.OMMVV] + ((nofWorlds - 1) * [robosample.IntegratorType.VERLET])

context.addReplicasAndLoadCoordinates(args.top, args.rstDir, nof_replicas)

for replIx in range(nof_replicas):
    #context.addReplica(replIx)
    context.addThermodynamicState(replIx,
                temperatures[replIx],
                accept_reject_modes,
                distort_options,
                distort_args,
                flow,
                work,
                integrators,
                worldIndexes,
                timesteps,
                mdsteps)

# Initialize
context.Initialize()

# Run
context.RunREX(args.equilSteps, args.prodSteps)
