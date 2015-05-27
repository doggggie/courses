import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logisticregr(X, y, w0, rate=0.1, niter=100):
    m, n = np.shape(X)
    wt = w0.copy()
    
    for i in range(niter):
        gradw = np.zeros(n)
        for j in range(m):
            z = -y[j] * np.dot(wt, X[j,:])
            gradw = gradw + sigmoid(z) * (-y[j] * X[j, :]) 
        gradw = gradw / float(m)
        wt = wt - rate * gradw
        dwnorm2 = np.linalg.norm(gradw, 2)
        cost = (np.log(1 + np.exp(-np.dot(X, wt) * y))).mean()
        print i, dwnorm2, cost
    return wt
    
def logisticregr_sgd_cyclic(X, y, w0, rate=0.1, niter=100):
    m, n = np.shape(X)
    wt = w0.copy()
    
    j = 0
    for i in range(niter):
        gradw = np.zeros(n)
        z = -y[j] * np.dot(wt, X[j,:])
        gradw = gradw + sigmoid(z) * (-y[j] * X[j, :]) 
        wt = wt - rate * gradw
        dwnorm2 = np.linalg.norm(gradw, 2)
        cost = (np.log(1 + np.exp(-np.dot(X, wt) * y))).mean()
        print i, dwnorm2, cost
        j = j + 1
        if j >= m:
            j = 0
    return wt    
    
def hw3q18(niter, rate):
    trainfn = 'C:\\shanying\\mooc\\ntumlone-002\\ntumlone-hw3-hw3_train.dat'
    testfn = 'C:\\shanying\\mooc\\ntumlone-002\\ntumlone-hw3-hw3_test.dat'
    traindata = np.loadtxt(trainfn)
    m, n = np.shape(traindata)
    
    y = traindata[:, n-1]
    X = traindata[:, 0:n-1]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    
    w0 = np.zeros(n)
    
    w = logisticregr(X, y, w0, rate, niter)    
    Ein = (np.sign(np.dot(X, w)) != y).mean()
    
    plt.plot(np.dot(X, w), '.')
    
    testdata = np.loadtxt(testfn)
    m, n = np.shape(testdata)
    y = testdata[:, n-1]
    X = testdata[:, 0:n-1]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    Eout = (np.sign(np.dot(X, w)) != y).mean()
    
    print "Ein, Eout", Ein, Eout
    
    
def hw3q20(niter, rate):
    trainfn = 'C:\\shanying\\mooc\\ntumlone-002\\ntumlone-hw3-hw3_train.dat'
    testfn = 'C:\\shanying\\mooc\\ntumlone-002\\ntumlone-hw3-hw3_test.dat'
    traindata = np.loadtxt(trainfn)
    m, n = np.shape(traindata)
    
    y = traindata[:, n-1]
    X = traindata[:, 0:n-1]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    
    w0 = np.zeros(n)
    
    w = logisticregr_sgd_cyclic(X, y, w0, rate, niter)    
    Ein = (np.sign(np.dot(X, w)) != y).mean()
    
    plt.plot(np.dot(X, w), '.')
    
    testdata = np.loadtxt(testfn)
    m, n = np.shape(testdata)
    y = testdata[:, n-1]
    X = testdata[:, 0:n-1]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    Eout = (np.sign(np.dot(X, w)) != y).mean()
    
    #plot.plot(np.dot(X, w), '.')
    
    print "Ein, Eout", Ein, Eout
        