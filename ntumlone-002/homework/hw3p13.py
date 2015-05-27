import numpy as np
import matplotlib.pyplot as plot

def linregr(X, y):
    m, n = np.shape(X)
    wt = np.dot(np.linalg.pinv(X), y)    
    return wt
    
def hw3q13(ntrain=100, niter=100):
    X = np.random.rand(ntrain, 2) * 2 - 1
    y = np.sign(X[:,0] ** 2 + X[:,1] ** 2 - 0.6)
    flip = (np.random.rand(ntrain) < 0.1)
    y[flip] = -y[flip]
    
    X = np.concatenate((np.ones((ntrain, 1)), X), axis=1)
    w = linregr(X, y)
    
    Ein = (np.sign(np.dot(X, w)) != y).mean()
    
    #plot.plot(np.dot(X, w), '.')
    
    print Ein
    
def hw3q14(ntrain=100):
    X = np.random.rand(ntrain, 2) * 2 - 1
    y = np.sign(X[:,0] ** 2 + X[:,1] ** 2 - 0.6)
    flip = (np.random.rand(ntrain) < 0.1)
    y[flip] = -y[flip]
    
    X = np.concatenate((np.ones((ntrain, 1)), \
                        np.reshape(X[:,0], (ntrain, 1)), \
                        np.reshape(X[:,1], (ntrain, 1)), \
                        np.reshape(X[:,0] * X[:,1], (ntrain, 1)), \
                        np.reshape(X[:,0] ** 2, (ntrain, 1)), \
                        np.reshape(X[:,1] ** 2, (ntrain, 1))), axis=1)
    w = linregr(X, y)
    
    Ein = (np.sign(np.dot(X, w)) != y).mean()
    
    print w
    
    print Ein
    
    Xtest = np.random.rand(ntrain, 2) * 2 - 1
    ytest = np.sign(Xtest[:,0] ** 2 + Xtest[:,1] ** 2 - 0.6)
    flip = (np.random.rand(ntrain) < 0.1)
    ytest[flip] = -ytest[flip]
    Xtest = np.concatenate((np.ones((ntrain, 1)), \
                        np.reshape(Xtest[:,0], (ntrain, 1)), \
                        np.reshape(Xtest[:,1], (ntrain, 1)), \
                        np.reshape(Xtest[:,0] * Xtest[:,1], (ntrain, 1)), \
                        np.reshape(Xtest[:,0] ** 2, (ntrain, 1)), \
                        np.reshape(Xtest[:,1] ** 2, (ntrain, 1))), axis=1)
    Eout = (np.sign(np.dot(Xtest, w)) != ytest).mean()
    
    print Eout
    
    return w
    