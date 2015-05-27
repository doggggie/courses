import numpy as np

def kernel_ridge_regression(X, y, lbda, gamma):
    m, n = np.shape(X)
    K = np.zeros((m, m))
    for i in range(m):
        K[i, i] = 1.0
        for j in range(i):
            dX = X[i, :] - X[j, :]
            K[i, j] = np.exp(-gamma * np.dot(dX, dX))
            
#            if j == 0:
#                print 'i,j, K', i, j, K[i, j], '  ', dX
            K[j, i] = K[i, j]
    beta = np.dot(np.linalg.pinv(lbda * np.eye(m) + K), y)
    return beta
    
def computeE(X, y, beta, gamma, Xtest, ytest):
    
    e = 0.0
    m, n = np.shape(X)
    mtest, ntest = np.shape(Xtest)
    for k in range(mtest):
        gx = 0.0
        for i in range(m):
            dX = X[i, :] - Xtest[k, :]
            gx = gx + beta[i] * np.exp(-gamma * np.dot(dX, dX))   
        e = e + (np.sign(gx) != ytest[k])
    e = e / float(mtest)
    return e
    
#def hw2p19():
if True:
    fn = 'hw2_lssvm_all.dat'
    
    alldata = np.loadtxt(fn)
    
    m, n = np.shape(alldata)    
    n = n - 1
    Xtrain = alldata[0:400, 0:n]
    ytrain = alldata[0:400, n]
    Xtest = alldata[400:, 0:n]
    ytest = alldata[400:, n]
    
    gammas = (32, 2, 0.125)
    lambdas = (0.001, 1, 1000)
    
    for lbda in lambdas:
        for gamma in gammas:
            beta = kernel_ridge_regression(Xtrain, ytrain, lbda, gamma)
            
            #print np.shape(beta)
            
            Ein = computeE(Xtrain, ytrain, beta, gamma, Xtrain, ytrain)
            Eout = computeE(Xtrain, ytrain, beta, gamma, Xtest, ytest)
            
            print 'gamma', gamma, 'lambda', lbda, 'Ein/Eout', Ein, Eout