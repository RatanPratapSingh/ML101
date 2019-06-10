import numpy as np 


def logistic(weights, X):

	weights = np.array(weights)
	X = np.array(X)
	z = np.dot(weights,X.T)

	deno = 1 + np.exp(-1*z)
	return 1/deno