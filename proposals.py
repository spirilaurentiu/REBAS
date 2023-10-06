
# Genral imports
import os, sys
import copy
import math
import numpy as np

# Specific imports
import distribs


# Propose a stretching with flipped lognormal distributed scale factor
def propose(mean, _a_, qDistrib = distribs.lognormal(), func = None, dev = False):
	"""
	Propose a stretch by a scaling factor drawn from QDistrib

    :param mean: target probability mean
    :param _a_: starting sample
	:param logn_a: lognormal left limit
	:param logn_b: lognormal right limit
    :return: describe what it returns
	"""
	
	# Initialize return values
	_b_ = _a_
	Q_a2b = Q_b2a = 1.0
	chi, chi_dual = 1.0, 1.0
	Jacob = 1.0
	
	# Draw paramter
	chi = qDistrib.randomTrunc()

	# Get new sample
	if(dev == False):
		_b_, Jacob, (chi_dual) = func(_a_, chi)
	else:
		stretchedDev, Jacob, (chi_dual) = func((_a_ - mean), chi)
		_b_ = mean + stretchedDev

	# Fwd and bkd probabilities
	Q_a2b = qDistrib.PDFTrunc(chi)
	Q_b2a = qDistrib.PDFTrunc(chi_dual)

	return (_b_, Q_a2b, Q_b2a, Jacob, chi, chi_dual)

# Addition function
def f_add(X, *chi):
	
	# Rule
	Y = chi[0] + X

	# Unary inverse parameters
	chi_dual = -1.0 * chi[0]

	# Jacobian
	Jacob = np.abs(1.0)

	return Y, Jacob, (chi_dual)

# Multiplication function
def f_mul(X, *chi):
	
	# Rule
	Y = chi[0] * X

	# Unary inverse parameters
	if chi[0] == 0:
		chi_dual = np.Inf
	else:
		chi_dual = 1.0 / chi[0]

	# Jacobian
	Jacob = np.abs(chi[0])

	return Y, Jacob, (chi_dual)

	
