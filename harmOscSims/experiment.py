# General imports
import os, sys
import math
import numpy as np
import copy
from scipy import optimize

# Specific imports
import distribs
import proposals as propose
from mcfuncs import *

# Parse arguments
import argparse
if 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('--nofExperiments', default=1, type=int,
                help='# of experiments.')
        parser.add_argument('--worlds', default = [0], nargs='+',
                help='Worlds to be run.')
        parser.add_argument('--sampleSize', default=10, type=int,
                help='# of samples in an experiment.')
        
        parser.add_argument('--coldTargetMean', default=7.0, type=float,
                help='Cold target distribution mean.')
        parser.add_argument('--coldTargetStd', default=2.0, type=float,
                help='Cold target distribution standard deviation.')

        parser.add_argument('--hotTargetMean', default=9.0, type=float,
                help='Hot target distribution mean.')
        parser.add_argument('--hotTargetStd', default=3.0, type=float,
                help='Hot target distribution standard deviation.')

        parser.add_argument('--distribOpt', default="gauss", type=str,
                help='Scaling factor distribution type.')

        parser.add_argument('--MC', action='store_true', default=False,
                help='Do MC on each replica.')
        parser.add_argument('--RE', action='store_true', default=False,
                help='Exchange replicas based on acc-rej exchange.')
        parser.add_argument('--RENS', action='store_true', default=False,
                help='Do RENS.')

        parser.add_argument('--dev', action='store_true', default=False,
                help='Scale the deviations instead of direct scaling.')  
        args = parser.parse_args()

# Probability to energy
def probToEnergy(prob, kT = 1.0, G = 0.0):
        """
        E = -kT log(prob) + G
        """
        if prob != 0.0:
                logp = np.log(prob)
        else:
                logp = 0.0

        return (-1.0 * kT * logp) + G
#

# Run equilibrium worlds
def runEquiliriumWorlds(x0, targetPDistrib, MC = False):
        """
        Run equilibrium worlds
        """

        xtemp = np.NaN

        pert_X_distrib = distribs.gauss()
        pert_X_distrib.setParams(0.0, 1.0)
        pert_X = pert_X_distrib.random()
        xpert, J, (pert_X_dual) = propose.f_add(x0, (pert_X))

        # Monte Carlo
        if MC:
                if MHAccept(targetPDistrib.PDF(x0), targetPDistrib.PDF(xpert)):
                        xtemp = xpert
                else:
                        xtemp = x0
        else:
                xtemp = xpert

        return xtemp, J, (pert_X_dual)
#

# Run nonequilibrium worlds
def runNonequiliriumWorlds(x0, s_X, targetPDistrib, qDistrib, MC = False):
        """
        Run nonequilibrium worlds
        """

        xtemp = np.nan

        xtau, Jacob, (s_X_dual) = propose.f_mul(x0, (s_X))

        xtau = p_C._mean_ + (1.0 * (x0 - p_C._mean_) * s_X)

        # Acception rejection if requested
        if MC:
                if MHAccept(
                        targetPDistrib.PDF(x0) *   qDistrib.PDF(np.abs(s_X)) * Jacob,
                        targetPDistrib.PDF(xtau) * qDistrib.PDF(np.abs(s_X_dual))):
                        
                        xtemp = xtau
                else:
                        xtemp = x0
        else:
                xtemp = xtau

        return xtemp, Jacob, (s_X_dual)
#

# Update parameters for nonequilibrium simulation
def updWorldsNonequilibriumParameters(proposalDistrib):
        """
        Update parameters for nonequilibrium simulation
        """

        return proposalDistrib.randomTrunc()
#


# Experiment
sampleSize = args.sampleSize

# Initial conditions
x0, y0 = 4.0, 4.0
xn, yn = np.NaN, np.NaN
xtau, ytau = np.NaN, np.NaN
s_X, s_X_1, s_Y, s_Y_1 = np.nan, np.nan, np.nan, np.nan
Jacob_C, Jacob_H = np.nan, np.nan
exchanges = 0.0

# Choose target distributions
p_C = distribs.gauss()
p_C.setParams(args.coldTargetMean, args.coldTargetStd)
p_H = distribs.gauss()
p_H.setParams(args.hotTargetMean, args.hotTargetStd)

# Choose proposal distributions
q_C = distribs.gauss()
q_C.setParams(0.0, 1.0)
q_C.setLimits(-4.0, 4.0)
q_H = distribs.gauss()
q_H.setParams(1.0, 0.0)
q_H.setLimits(-4.0, 4.0)

distribOpt = args.distribOpt
print("Got", distribOpt, "for scaling factor distributions",
        file = sys.stderr)

if distribOpt == "deterministic":
        scaleFactor = 4.0
        sf_C = scaleFactor
        sf_H = 1.0 / scaleFactor
        
        q_C = distribs.constant()
        q_C.setParams(sf_C, sf_C)
        q_C.setLimits(sf_C, sf_C)
        q_H = distribs.constant()
        q_H.setParams(sf_H, sf_H)
        q_H.setLimits(sf_H, sf_H)

elif distribOpt == "Bernoulli":
        q_C = distribs.bernoulli()
        q_C.setParams(0.25, 4.0, 0.5)
        q_C.setLimits(-99999, 99999)
        q_H = distribs.bernoulli()
        q_H.setParams(0.25, 4.0, 0.5)
        q_H.setLimits(-99999, 99999)

elif distribOpt == "gauss":
        q_C = distribs.gauss()
        q_C.setParams(1.1, 1.0)
        q_C.setLimits(0.5, 2.0)
        q_H = distribs.gauss()
        q_H.setParams(0.9, 1.0)
        q_H.setLimits(0.5, 2.0)

# Nof worlds
worlds = [int(worlds) for worlds in args.worlds]
nofWorlds = len(worlds)
print("Got", nofWorlds, "worlds:", worlds, file = sys.stderr)


# Gather results
sample_x = np.full((sampleSize * (1)), np.nan, dtype = float) 
sample_y = np.full((sampleSize * (1)), np.nan, dtype = float)

runNonequilibriumWorlds_Opt = True
runEquilibriumWorlds_Opt = True
nofReplicas = 2
acceptance = np.zeros((nofReplicas, nofWorlds), dtype = float)

# Run
for mixi in range(sampleSize):


        # =====================================================================
        # ------------------------------ REPLICA C ----------------------------
        # ---------------------------------------------------------------------
                
        # Run equilibrium worlds
        if 0 in worlds:
                x_set, J, (pert_X_dual)  = runEquiliriumWorlds(x0, p_C, MC = True)
                
                # Record acceptance
                if(x_set != x0): acceptance[0, 0] += 1.0

                # Reset
                x0 = x_set      

        # --------------------------------

        # Run non-equilibrium worlds
        if 1 in worlds:

                # Get a scaling factor
                s_X    = updWorldsNonequilibriumParameters(q_C)
                #s_X *= distribs.randomSign()
                s_X_1 = 1.0 / s_X

                # Propose
                xtau, Jacob_C, (s_X_dual)  = runNonequiliriumWorlds(x0, s_X, p_C, q_C,
                                                MC = args.MC)
                assert(s_X_1 == s_X_dual)

        else:
                xtau = x0

        # =====================================================================
        # ------------------------------ REPLICA H ----------------------------
        # ---------------------------------------------------------------------

        # Run equilibrium worlds
        if 0 in worlds:
                y_set, J, (pert_Y_dual)  = runEquiliriumWorlds(y0, p_H, MC = True)

                # Record acceptance
                if(y_set != y0): acceptance[1, 0] += 1.0

                # Reset
                y0 = y_set
                
        # --------------------------------

        # Run non-equilibrium worlds
        if 1 in worlds:

                # Get a scaling factor
                s_Y    = updWorldsNonequilibriumParameters(q_H)
                #s_Y *= distribs.randomSign()
                s_Y_1 = 1.0 / s_Y
                
                # Propose
                ytau, Jacob_H, (s_Y_dual)  = runNonequiliriumWorlds(y0, s_Y, p_H, q_H,
                                                MC = args.MC)

                assert(s_Y_1 == s_Y_dual)

        else:
                ytau = y0

        # =====================================================================
        # ------------------------------ attemptREX ---------------------------
        # ---------------------------------------------------------------------

        xn, yn   = x0, y0

        # Initial target probabilities (eii, ejj, eij, eji)
        pC_x0, pH_y0   = p_C.PDF(x0), p_H.PDF(y0)
        pH_x0, pC_y0   = p_H.PDF(x0), p_C.PDF(y0)

        # Last target probabilities (lii, ljj, lij, lji)
        pC_xtau, pH_ytau = p_C.PDF(xtau), p_H.PDF(ytau)
        pH_xtau, pC_ytau = p_H.PDF(xtau), p_C.PDF(ytau)

        # Correction term
        correctionTerm = 0.0
        qC_s_X,   qH_s_Y   = q_C.PDFTrunc(s_X),   q_H.PDFTrunc(s_Y)
        qH_s_X_1, qC_s_Y_1 = q_H.PDFTrunc(s_X_1), q_C.PDFTrunc(s_Y_1)

        if (qC_s_X * qH_s_Y) != 0.0:
                correctionTerm = (qH_s_X_1 * qC_s_Y_1) / (qC_s_X * qH_s_Y)

        # Overall division by 0 avoidance
        Denominator = (pC_x0 * Jacob_C) * (pH_y0 * Jacob_H) * (qC_s_X * qH_s_Y)



        if args.RE == True: # Exchange

                if MHAccept(pC_xtau * pH_ytau,
                            pH_xtau * pC_ytau):

                        xn, yn = ytau, xtau
                        exchanges += 1

                else:

                        xn, yn = xtau, ytau

        elif args.RENS: # exchange the scaled versions

                # Work
                Work_A = probToEnergy(pH_xtau) - probToEnergy(pC_x0) - probToEnergy(Jacob_C)
                Work_B = probToEnergy(pC_ytau) - probToEnergy(pH_y0) - probToEnergy(Jacob_H)
                WTerm = -1.0 * (Work_A + Work_B)

                unifSample = np.random.uniform()

                log_p_accept = WTerm + np.log(correctionTerm)

                if( (log_p_accept >= 0.0) or (unifSample < np.exp(log_p_accept))):

                        xn, yn = ytau, xtau
                        
                        exchanges += 1

                else:
                        xn, yn = x0, y0

        else: # No exchanges
                xn, yn = xtau, ytau

        # Reset
        x0, y0 = xn, yn

        # Record
        sample_x[mixi] = xn
        sample_y[mixi] = yn


# Print results
print("acceptances", acceptance, file = sys.stderr)
print(exchanges, "exchanges", file=sys.stderr)
for mixi in range(sampleSize):
        print(  sample_x[mixi],
                sample_y[mixi])




