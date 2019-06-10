import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

from Logistic import logistic
from ComputeEntropyCost import computeEntropyCost
from ComputeGradient import computeGradient
from GradientDescent import gradientDescent

# Reading and loading X and y
dataset = pd.read_csv("C://Users//Ratan Singh//Desktop//ML Training Code//LogisticRegression//Banknote_authentication.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3)


# Defining parameters for gradient descent

initalWeights = np.random.random([1,X.shape[1]])
maxIter = 5000
learningRate = 0.01


# Training a Logistic Regression

weights = initalWeights
cost = []

for i in range(maxIter):

	yp = logistic(weights, X_train)
	J = computeEntropyCost(Y_train, yp)
	G = computeGradient(X_train, Y_train, yp)
	weights = gradientDescent(weights, G, learningRate)

	if i%10 ==0:
		print("Cost of the model is {}".format(J))
	cost.append(J)


print("Weights {} after the training are".format(weights))



# Prediction using the model

yp =logistic(weights, X_test)
yp = (yp >= 0.5)*1.0

print(np.sum(np.abs(yp-Y_test)))



