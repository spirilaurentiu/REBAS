# Imports
import sys, os, glob, copy
import numpy as np
import argparse

exeFN = "/home/laurentiu/git5/Robosample/build/release/robosample/src/GMOLMODEL_robo"

constInpChunk = """
PRMTOP ligand.prmtop ligand.prmtop ligand.prmtop
INPCRD ligand ligand ligand
RBFILE ligand.rb ligand.rb ligand.rb
ROOTS 0 0 0
ROOT_MOBILITY Weld Weld Weld
OUTPUT_DIR temp

ROUNDS_TILL_REBLOCK 10 10 10
WORLDS R0 R1 R2
RANDOM_WORLD_ORDER FALSE FALSE FALSE
REPRODUCIBLE FALSE FALSE FALSE

# New keywords
DISTORT_OPTION 0 0 0
FLOW_OPTION 0 0 0
WORK_OPTION 0 0 0
SAMPLERS MC MC MC
SAMPLES_PER_ROUND 1 1 1

FIXMAN_POTENTIAL TRUE TRUE TRUE
FIXMAN_TORQUE TRUE TRUE TRUE
GBSA 0.0 0.0 0.0

PRINT_FREQ 1 1 1
WRITEPDBS 0 0 0
VISUAL FALSE FALSE FALSE

GEOMETRY TRUE TRUE TRUE
DISTANCE 0 18
DIHEDRAL 4 6 8 14 6 8 14 16


THREADS 0 0 0
OPENMM TRUE TRUE TRUE
OPENMM_CalcOnlyNonbonded FALSE FALSE FALSE
NONBONDED_METHOD 0 0 0
NONBONDED_CUTOFF 1.2 1.2 1.2
"""

# This will not be used
varTypeChunk = """
# Molecule information
MOLECULES ala1
FLEXFILE ligand.cart.flex ligand.td.flex ligand.BA.flex

RUN_TYPE RENS
REX_SWAP_EVERY 1
REX_FILE trex.rens.dat
REX_SWAP_FIXMAN 0

ROUNDS 120000
SEED 12300 12300 12300
"""

varSimChunk = """
TIMESTEPS 0.001 0.004 0.001
MDSTEPS 30 10 10
BOOST_MDSTEPS 1 1 1
TEMPERATURE_INI 1000 1000 1000
TEMPERATURE_FIN 1000 1000 1000
BOOST_TEMPERATURE 1000 1000 1000
THERMOSTAT Andersen Andersen Andersen
FFSCALE AMBER AMBER AMBER
"""

# ------------------------------
# Experiment details
# ==============================
molIx = 1
molDir = 'ala1'
Types = ['HMC', 'HMCS', 'RENEMC', 'RENE']
nofTypes = len(Types)
trexFNs = ['trex.dat', 'trex.dat', 'trex.renemc.dat', 'trex.rene.dat']

distortTypes = [0, -1, -1, -1]
integrators = ["VV", "EMPTY", "EMPTY", "EMPTY"]

HMC_, HMCS_, REX_, RENS_ = range(nofTypes)

subTypes = ['BA']
nofSubTypes = len(subTypes)
BA_ = 0

trexFNs = ['trex.dat', 'trex.dat', 'trex.renemc.dat', 'trex.rene.dat']

# Useful global vars
nofRepeats = 4
nofRounds = 120000
nofReplicas = 2
nofWorldsPerReplica = 3




