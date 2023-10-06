
# General imports
import os, sys
import numpy as np

# Specific imports
import random
import scipy
import scipy
import scipy.stats
from scipy.stats import kstest

# Recursively update mean
def updateMean(prevMean, newValue, N):
	"""
	Update the mean given a new value

    :param prevMean: previous mean
    :param newValue: new value
    :param N: size of the series
    :return: new mean
	"""
	N = float(N)
	if(N == 1.0):
		return newValue
	else:
		N_1overN = ((N-1.0) / N)
		NInv = 1.0 / N

	return (N_1overN * prevMean) + (NInv * newValue)

def randomSign():
	u = np.random.uniform()
	if u < 0.5: return -1.0
	else: return 1.0

# Gaussian Distribution
class constant:
	"""
	Returns a constant distribution
	"""

	def __init__(self):
		"""
		Declares _mean_,_var_ and _std_
		"""

		self._mean_ = np.NaN
		self._std_ = np.NaN
		self._var_ = np.NaN

		self._L = np.NINF
		self._R = np.Inf

	# Set params
	def setParams(self, *params):
		"""
		:param mean: mean
		:param: std: std
		"""

		self._mean_ = params[0]
		self._std_ = 0.0
		self._var_ = 0.0

	# Set limits
	def setLimits(self, L, R):
		"""
		:param L: left limit
		:param: R: right limit
		"""

		assert((L == R) and (L == self._mean_))
		self._L = L
		self._R = R

	# CDF
	def CDF(self, X):
		"""
		CDF
		:param X: variable
		:return : CDF
		"""
		if(X == self._L):
			return 1.0
		else:
			return 0.0

	# Normal pdf
	def PDF(self, X):
		"""
		Pdf

		:param x: variable
		:return: pdf of x
		"""

		if(X == self._L):
			return 1.0
		else:
			return 0.0

	# Truncated PDF
	def PDFTrunc(self, X):
		"""
		Pdf

		:param x: variable
		:return: pdf of x
		"""

		if(X == self._L):
			return 1.0
		else:
			return 0.0

	# Draw from a random normal
	def random(self):
		"""
		Draw from a random Gaussian
		"""

		return self._L

	# Draw from a truncated Gaussian distribution
	def randomTrunc(self):
		"""
		Generate random numbers
		"""

		return self._L



# Bernoulli Distribution
class bernoulli:
	"""
	Our Bernoulli distribution
	"""

	def __init__(self):
		"""
		Declares the two values
		"""

		self._lval_ = np.NaN
		self._rval_ = np.NaN
		self._p_ = 0.5
		self._q_ = 0.5

		self._L = np.NINF
		self._R = np.Inf

	# Set params
	def setParams(self, *params):
		"""
		:param lval: left value
		:param: rval: right value
		:param p: prob. of the left value
		"""

		self._lval_ = params[0]
		self._rval_ = params[1]
		self._q_ = params[2]
		self._p_ = 1.0 - self._q_

	# Set limits
	def setLimits(self, L, R):
		"""
		:param L: left limit
		:param: R: right limit
		"""

		self._L = L
		self._R = R

	# CDF
	def CDF(self, X):
		"""
		CDF
		:param X: variable
		:return : CDF
		"""

		if X < self._lval_:
			return 0.0
		elif X >= self._lval_:
			return self._q_
		else:
			return 1.0

	# PDF
	def PDF(self, X):
		"""
		Normal pdf

		:param mean: mean
		:param: std: std
		:param x: variable
		:return: pdf of x
		"""
		if X == self._lval_:
			return self._q_
		elif X == self._rval_:
			return self._p_
		else:
			return 0.0

	# Truncated PDF
	def PDFTrunc(self, X):
		"""
		PDF

		:param x: variable
		:return: pdf of x
		"""

		if (X <= self._L) or (X > self._R):
			return 0.0
		else:
			num = self.PDF(X)

			den = (self.CDF(self._R) - self.CDF(self._L))
			
			return num / den

	# Draw from a random normal
	def random(self):
		"""
		Draw from a random Bernoulli
		"""
		if np.random.uniform() < 0.5:
			return self._lval_
		else:
			return self._rval_

	# Draw from a truncated Gaussian distribution
	def randomTrunc(self):
		"""
		Generate random numbers
		"""

		for i in range(100):
			X = self.random()
			if (X > self._L) and (X <= self._R):
				return X
		return np.NaN



# Uniform distribution
class uniform:
	"""
	Uniform distribution
	"""

	def __init__(self):
		"""
		Declares limits
		"""
		self._L = np.NaN
		self._R = np.NaN

	# Set params
	def setParams(self, *params):
		"""
		:param L: left limit
		:param R: right limit
		"""

		self._L = params[0]
		self._R = params[1]
		if self._R < self._L:
			print("Error: right limit is less than left limit. Exiting...", self._L, self._R)
			sys.exit(1)

	# Set limits
	def setLimits(self, L, R):
		"""
		:param L: left limit
		:param: R: right limit
		"""

		self._L = L
		self._R = R
		if self._R < self._L:
			print("Error: right limit is less than left limit. Exiting...", self._L, self._R)
			sys.exit(1)

	# CDF
	def CDF(self, X):
		"""
		CDF
		:param X: variable
		:return : CDF
		"""

		cdf = np.NaN
		if (X < self._L):
			cdf = 0.0
		elif (X > self._R):
			cdf = 1.0
		else:
			cdf = (X - self._L) / (self._R - self._L)

		return cdf

	# PDF
	def PDF(self, X):
		"""
		PDF

		:param x: variable
		:return: pdf of x
		"""
		
		pdf = np.NaN
		if (X < self._L):
			pdf = 0.0
		elif (X > self._R):
			pdf = 0.0
		else:
			pdf = 1.0 / (self._R - self._L)

		return pdf

	# Truncated PDF
	def PDFTrunc(self, X):
		"""
		trunchated PDF

		:param x: variable
		:return: pdf of x
		"""

		pdf = self.PDF(X)

		return pdf

	# Draw from uniform
	def random(self):
		"""
		Draw from a uniform
		"""
		r = np.random.uniform(low = self._L, high = self._R)
		return r

	# Draw from a truncated uniform
	def randomTrunc(self):
		"""
		Draw from a truncated uniform (uniform)
		"""
		r = np.random.uniform(low = self._L, high = self._R)
		return r



# Gaussian Distribution
class gauss:
	"""
	Our normal distribution
	"""

	def __init__(self):
		"""
		Declares _mean_,_var_ and _std_
		"""

		self._mean_ = np.NaN
		self._std_ = np.NaN
		self._var_ = np.NaN

		self._L = np.NINF
		self._R = np.Inf

	# Set params
	def setParams(self, *params):
		"""
		:param mean: mean
		:param: std: std
		"""

		self._mean_ = params[0]
		self._std_ = params[1]
		self._var_ = self._std_**2

	# Set limits
	def setLimits(self, L, R):
		"""
		:param L: left limit
		:param: R: right limit
		"""

		self._L = L
		self._R = R

	# Un-normalized normal pdf
	def normalPdf_unNorm(self, X):
		"""
		Un-normalized normal pdf

		:param mean: mean
		:param: std: std
		:param x: variable
		:return: unnormalized pdf of x
		"""

		num = (X - self._mean_)**2 
		den = (self._std_)**2

		return np.exp(-0.5 * num / den)

	# CDF
	def CDF(self, X):
		"""
		CDF
		:param X: variable
		:return : CDF
		"""

		if X == np.NINF:
			return 0.0
		if X == np.Inf:
			return 1.0

		num = X - self._mean_
		den = self._std_ * 1.4142135623731

		erf = scipy.special.erf(num / den)

		return 0.5 * (1.0 + erf)

	# Normal pdf
	def PDF(self, X):
		"""
		Normal pdf

		:param mean: mean
		:param: std: std
		:param x: variable
		:return: pdf of x
		"""

		normFactor = 1.0 / (self._std_ * np.sqrt(2.0 * np.pi))

		return normFactor * self.normalPdf_unNorm(X)

	# Truncated PDF
	def PDFTrunc(self, X):
		"""
		PDF

		:param x: variable
		:return: pdf of x
		"""

		if (X <= self._L) or (X > self._R):
			return 0.0
		else:
			num = self.PDF(X)

			den = (self.CDF(self._R) - self.CDF(self._L))
			
			return num / den

	# Draw from a random normal
	def random(self):
		"""
		Draw from a random Gaussian
		"""
		X = np.random.normal(loc = self._mean_, scale = self._std_)

		return X

	# Draw from a truncated Gaussian distribution
	def randomTrunc(self):
		"""
		Generate random numbers
		"""

		for i in range(100):
			X = self.random()
			if (X > self._L) and (X <= self._R):
				return X
		return np.NaN



# Lognormal Distribution
class lognormal:
	"""
	Lognormal distribution
	"""

	def __init__(self):
		"""
		Parameters

		:param normMiu: corresponding normal mean
		:param normStd: corresponding normal std
		:param L: left limit
		:param R: right limit
		"""

		# Corresponding normal distribution params
		self._normMiu = 0.0
		self._normStd = 1.0

		# Lognormal distribution params
		self._mode, self._median, self._mean_ = np.NaN, np.NaN, np.NaN

		# Limits if any truncation
		self._R, self._L = np.NaN, np.NaN

	def setParams(self, *params):
		"""
		Set parameters

		:param normMiu: corresponding normal mean
		:param normStd: corresponding normal std
		:param L: left limit
		:param R: right limit
		"""

		self._mode = params[0]
		self._median = params[1]
		self._mean_ = params[2]

		self._R = 2.0 * self._median
		self._L = 1.0 / self._R

		lnp = np.log(self._mode)
		self._normMiu = np.log(self._median)
		self._normVar = self._normMiu - lnp
		self._normStd = np.sqrt(self._normVar)

	# Set limits
	def setLimits(self, L, R):
		"""
		:param mean: mean
		:param: std: std
		"""

		self._L = L
		self._R = R

	def printParams(self):
		"""
		Print the parameters
		"""
		print(self._normMiu, self._normStd, self._L, self._R,
			self._mode, self._median, self._mean_)

	# Log-normal distribution PDF
	def PDF(self, X):
		"""
		Log-normal PDF

		:param X: variable
		:return: lognormal probability of x
		"""
		
		f1 = 1.0 / (self._normStd * X * np.sqrt(2.0 * np.pi))

		num = (np.log(X) - self._normMiu)**2
		den = 2.0 * (self._normStd**2)

		unormLnPdf = -1.0 * (num / den)

		pdf = f1 * np.exp(unormLnPdf)

		# sciPdf = scipy.stats.lognorm.pdf(X,
		# 	self._normStd, 0.0, np.exp(self._normMiu)) # Scipy PDF
		#print('lognormal', self._normMiu, self._normStd, X, pdf, sciPdf)

		return pdf

	# Log-normal distribution CDF
	def CDF(self, X):
		"""
		Log-normal CDF

		:param X: variable
		:return: lognormal CDF of X
		"""

		if X == np.NINF:
			return 0.0
		if X == np.Inf:
			return 1.0

		if X == 0.0:
			return 0.0

		num = np.log(X) - self._normMiu
		den = self._normStd * np.sqrt(2.0)
		
		erf = scipy.special.erf(num / den)

		cdf = 0.5 * (1.0 + erf)

		#sciCdf = scipy.stats.lognorm.cdf(X,
		# 	self._normStd, 0.0, np.exp(self._normMiu)) # Scipy PDF
		#print('lognormal', self._normMiu, self._normStd, X, cdf, sciCdf)

		return cdf

	# PDF 
	def PDFTrunc(self, X):
		"""
		Truncated log-normal PDF

		:param X: variable
		:return: lognormal probability of x
		"""

		if (X <= self._L) or (X > self._R):
			return 0.0
		else:
			num = self.PDF(X)

			den = (self.CDF(self._R) - self.CDF(self._L))
			
			return num / den


	# Draw from a lognormal distribution
	def random(self):
		"""
		Generate random numbers from a log-normal

		:param p1: describe about parameter p1
		:return: describe what it returns
		"""

		normX_0_1 = np.random.normal(loc = 0.0, scale = 1.0)

		return np.exp(self._normMiu + (self._normStd * normX_0_1))

	# Draw from a truncated lognormal distribution
	def randomTrunc(self):
		"""
		Generate random numbers from a log-normal within an interval

		:param p1: describe about parameter p1
		:return: describe what it returns
		"""

		for i in range(100):
			X = self.random()
			if (X > self._L) and (X <= self._R):
				return X
		return np.NaN



# Lognormal Distribution Flipped
class lognormalFlipped(lognormal):
	"""
	Lognormal distribution flipped
	"""

	def __init_(self):
		super().__init__()


	# Flip a variable within two limits (use base class L and R)
	def flip(self, X):
		"""
		Flip X given L and R limits
		The function is its own index

		:param X: lognormal random variables
		:return: flipped variables
		"""

		X_F = self._R - X + self._L

		return X_F

	# Log-normal distribution CDF
	def CDF(self, X_F):
		"""
		Log-normal CDF

		:param X: variable
		:return: lognormal CDF of X
		"""

		X = self.flip(X_F) # flip

		return super().CDF(X)

	# PDF flipped
	def PDFTrunc(self, X_F):
		"""
		Truncated log-normal PDF

		:param X: variable
		:return: lognormal probability of x
		"""

		X = self.flip(X_F) # flip

		if (X <= self._L) or (X > self._R):
			return 0.0
		else:
			num = super().PDF(X)

			den = (super().CDF(self._R) - super().CDF(self._L))

			return num / den

	# Draw from a flipped truncated lognormal distribution
	def randomTrunc(self):
		"""
		Generate random numbers from a log-normal within an interval

		:return: random number
		"""

		for i in range(100):
			X_F = super().random()
			if (X_F > self._L) and (X_F <= self._R):
				X = self.flip(X_F) # flip
				return X
		return np.NaN




