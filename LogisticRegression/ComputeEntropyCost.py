import numpy as np

# code to compute L2 Cost

def computeEntropyCost(y,yp):

    # Write your code here
    y = np.array(y)
    yp = np.array(yp)
    m = len(y)    
    J = -1*np.sum(y*np.log(yp)+(1-y)*np.log(1-yp))/m
    return J
