# General imports
import os, sys
import math
import numpy as np
import copy
from scipy import optimize

# Specific imports
import distribs
import proposals as propose

# Debugging functions
# Trace
def TRACE(debugString, *debugVars, END = '\n'):
        """
        Trace

        :param debugString: message
        :param debugVars: variables to print out
        :return: print
        """
        if(len(debugString) > 0):
                print(debugString, end = ' ')
        for debugVar in debugVars:
                print(debugVar, end = ' ')
        print(end = END)

# Harmonic oscillator potential
def HarmOscPotential(bond, bondEquil, k_bond):
        """
        Harmonic oscillator potential

    :param bond: bond length
    :param bondEquil: equilibrium bond length
    :param k_bond: elastic constant
    :return: bond length potential
        """
        return 0.5 * k_bond * ((bond - bondEquil)**2)

# Boltzmann factor
def BoltzmannFactor(pe, beta):
        """
        Boltzmannr factor

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
        """
        return np.exp(-beta * pe)

# Metropolis-Hastings operator
def MHOperator(pdf_o, pdf_n):
        """
        Metropolis-Hastings operator

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
        """
        #r = np.exp(np.log(float(pdf_n)) - np.log(float(pdf_o)))
        r = np.NaN
        if pdf_o == 0.0 :
                return 1.0
        else:
                r = float(pdf_n) / float(pdf_o)

        return np.min([1.0, r])

# Metropolis-Hastings operator dual
def invMHOperator(pdf_o, pdf_n):
        """
        Metropolis-Hastings operator dual

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
        """
        r = float(pdf_n) / float(pdf_o)
        return np.max([1.0, r])

# Metropolis-Hastings criteria
def MHAccept(pdf_o, pdf_n):
        """
        Metropolis-Hastings acceptance providing the pdfs

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
        """
        unif = np.random.uniform(0, 1)
        if unif < MHOperator(pdf_o, pdf_n) : return True
        else: return False

# Metropolis-Hastings criteria
def MHAcceptLog(pdf_o_ln, pdf_n_ln):
        """
        Metropolis-Hastings acceptance providing the pdfs

        :param p1: describe about parameter p1
        :param p2: describe about parameter p2
        :param p3: describe about parameter p3
        :return: describe what it returns
        """

        r = np.exp(pdf_n_ln - pdf_o_ln)
        unif = np.random.uniform(0, 1)
        if unif < r : return True
        else: return False


# Swaps the last elements from two lists
def swapEnds(list_A, list_B):
        """
        Swaps last element in listA with last from list_B

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
        """
        temp = list_A[-1]
        list_A[-1] = list_B[-1]
        list_B[-1] = temp

# Exchange criteria for parallel tempering
def acceptExchange(coldPE, hotPE, coldBeta, hotBeta):
        """
        Parallel tempering exchange criteria

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
        """
        condition = np.exp((coldBeta - hotBeta) * (coldPE - hotPE))
        if np.random.uniform() < condition: return True
        else: return False

# Exchange criteria for non-equilibrium switches
def acceptWork(coldPE_0, hotPE_0, coldPE_tau, hotPE_tau,
        coldBeta, hotBeta):
        """
        Non-equil switches exchange criteria

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
        """
        condition  = (-1.0 * coldBeta *  hotPE_tau)
        condition += (-1.0 *  hotBeta * coldPE_tau)
        condition += (+1.0 * coldBeta * coldPE_0)
        condition += (+1.0 *  hotBeta *  hotPE_0)
        condition = np.exp(condition)
        if np.random.uniform() < condition: return True
        else: return False

# Convenient vars
COLD, HOT, W_0, W_1 = 0, 1, 0, 1
