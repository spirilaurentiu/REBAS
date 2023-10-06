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
        parser.add_argument('--nofWorlds', default=1, type=int,
                help='# of worlds per replica.')
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
def runEquiliriumWorlds(x0, targetPDistrib):
        """
        Run equilibrium worlds
        """

        xtemp = np.NaN

        pert_X_distrib = distribs.gauss()
        pert_X_distrib.setParams(0.0, 1.0)
        pert_X = pert_X_distrib.random()
        xpert, J, (pert_X_dual) = propose.f_add(x0, (pert_X))

        # Monte Carlo
        if MHAccept(targetPDistrib.PDF(x0), targetPDistrib.PDF(xpert)):
                xtemp = xpert
        else:
                xtemp = x0
        return xtemp, J, (pert_X_dual)
#

# Run nonequilibrium worlds
def runNonequiliriumWorlds(x0, s_X):
        """
        Run nonequilibrium worlds
        """

        xtemp = np.nan
        sgn = distribs.randomSign()

        xtau, Jacob, (s_X_dual) = propose.f_mul(x0, (s_X))

        #xtau *= sgn

        xtau = p_C._mean_ + (1.0 * (x0 - p_C._mean_) * s_X)

        # No Monte Carlo
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
xmc, ymc = np.NaN, np.NaN
exchanges = 0.0
s_X, s_X_1, s_Y, s_Y_1 = np.nan, np.nan, np.nan, np.nan

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
        #scaleFactor = 1.0 # TEST 3 FAILED
        sf_C = scaleFactor
        sf_H = 1.0 / scaleFactor
        
        q_C = distribs.constant()
        q_C.setParams(sf_C, sf_C)
        q_C.setLimits(sf_C, sf_C)
        q_H = distribs.constant()
        q_H.setParams(sf_H, sf_H)
        q_H.setLimits(sf_H, sf_H)

elif distribOpt == "gauss":
        q_C = distribs.gauss()
        q_C.setParams(1.1, 1.0)
        q_C.setLimits(0.5, 2.0)
        q_H = distribs.gauss()
        q_H.setParams(0.9, 1.0)
        q_H.setLimits(0.5, 2.0)

# Nof worlds
nofWorlds = args.nofWorlds
worlds = np.arange(0, nofWorlds)

# Gather results
sample_x = np.full((sampleSize * (nofWorlds - 1)), np.nan, dtype = float) 
sample_y = np.full((sampleSize * (nofWorlds - 1)), np.nan, dtype = float)

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
        x_set, J, (pert_X_dual)  = runEquiliriumWorlds(x0, p_C)
        
        # Record acceptance
        if(x_set != x0): acceptance[0, 0] += 1.0

        # Reset
        x0 = x_set      

        # --------------------------------

        # Get a scaling factor
        s_X    = updWorldsNonequilibriumParameters(q_C)
        s_X_1 = 1.0 / s_X

        # Propose
        xtau, Jacob_C, (s_X_dual)  = runNonequiliriumWorlds(x0, s_X)
        assert(s_X_1 == s_X_dual) 

        # =====================================================================
        # ------------------------------ REPLICA H ----------------------------
        # ---------------------------------------------------------------------

        # Run equilibrium worlds
        y_set, J, (pert_Y_dual)  = runEquiliriumWorlds(y0, p_H)

        # Record acceptance
        if(y_set != y0): acceptance[1, 0] += 1.0

        # Reset
        y0 = y_set
                
        # --------------------------------

        # Get a scaling factor
        s_Y    = updWorldsNonequilibriumParameters(q_H)
        s_Y_1 = 1.0 / s_Y
        
        # Propose
        ytau, Jacob_H, (s_Y_dual)  = runNonequiliriumWorlds(y0, s_Y)

        assert(s_Y_1 == s_Y_dual)

        # ================================

        # Initial target probabilities (eii, ejj, eij, eji)
        pC_x0   = p_C.PDF(x0)
        pH_y0   = p_H.PDF(y0)
        pH_x0   = p_H.PDF(x0)
        pC_y0   = p_C.PDF(y0)

        # Last target probabilities (lii, ljj, lij, lji)
        pC_xtau = p_C.PDF(xtau)
        pH_ytau = p_H.PDF(ytau)
        pH_xtau = p_H.PDF(xtau)
        pC_ytau = p_C.PDF(ytau)

        # Work
        Work_A = probToEnergy(pH_xtau) - probToEnergy(pC_x0) - probToEnergy(Jacob_C)
        Work_B = probToEnergy(pC_ytau) - probToEnergy(pH_y0) - probToEnergy(Jacob_H)
        WTerm = -1.0 * (Work_A + Work_B)

        # Correction term
        qC_s_X   = q_C.PDFTrunc(s_X)
        qC_s_X_1 = q_C.PDFTrunc(s_X_1)
        qH_s_Y   = q_H.PDFTrunc(s_Y)
        qH_s_Y_1 = q_H.PDFTrunc(s_Y_1)

        qH_s_X_1 = q_H.PDFTrunc(s_X_1)
        qC_s_Y_1 = q_C.PDFTrunc(s_Y_1)

        correctionTerm = 0.0
        if (qC_s_X * qH_s_Y) != 0.0:
                correctionTerm = (qH_s_X_1 * qC_s_Y_1) / (qC_s_X * qH_s_Y)

        # Overall division by 0 avoidance
        Denominator = (pC_x0 * Jacob_C) * (pH_y0 * Jacob_H) * (qC_s_X * qH_s_Y)

        # Metropolis-Hastings acc-rej step
        xn, yn   = np.NaN, np.NaN
        xmc, ymc = np.NaN, np.NaN

        if args.MC:

                # Do Monte Carlo per replica
                if MHAccept(pC_x0   * qC_s_X  * Jacob_C,
                            pC_xtau * qC_s_X_1):
                        xmc = xtau
                        acceptance[0, 1] += 1.0
                else:
                        xmc = x0

                if MHAccept(pH_y0   * qH_s_Y  * Jacob_H,
                            pH_ytau * qH_s_Y_1):
                        ymc = ytau
                        acceptance[1, 1] += 1.0

                else:
                        ymc = y0

                pC_xmc   = p_C.PDF(xmc)
                pH_ymc   = p_H.PDF(ymc)
                pH_xmc   = p_H.PDF(xmc)
                pC_ymc   = p_C.PDF(ymc)

                if args.RE == True: # Exchange

                        if MHAccept(pC_xmc * pH_ymc,
                                    pH_xmc * pC_ymc):

                                xn, yn = ymc, xmc
                                exchanges += 1
                        else:
                                xn, yn = xmc, ymc
                        
                else:                # Don't exchange

                        xn, yn = xmc, ymc

        elif args.RENS: # exchange the scaled versions

                unifSample = np.random.uniform()

                log_p_accept = WTerm + np.log(correctionTerm)

                if( (log_p_accept >= 0.0) or (unifSample < np.exp(log_p_accept))):
                        xn, yn = ytau, xtau
                        exchanges += 1
                else:
                        xn, yn = x0, y0

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




