import numpy as np

DEBUG_PRT = False

class KNearestNeighbor(object):
    
    # d-M-1 has 2 layers, and layers = {0:d, 1:M, 2:1}
    # assume the output layer has d_L = 1
    
    def __init__(self, k):
        self.k = k
    
    def train(self, X, y):    
        m, n = np.shape(X)
        # lazy training
        self.X = X
        self.y = y
           
    def predict(self, X):
        m, n = np.shape(self.X)
        
        dists = np.zeros(m)
        

        for i in range(m):
            dists[i] = np.linalg.norm(self.X[i,:] - X, 2)
        
        if self.k == 1:
            ind = np.argmin(dists)
            return y[ind]
        else:     
            inds = np.argsort(dists)[:self.k]
            return np.sign(sum(y[inds]))
            


    def compute_err(self, X, y):
        m, n = np.shape(X)
    
        err = 0.0
        for i in range(m):
            ypredict = self.predict(X[i,:])
            if ypredict != y[i]:
                err += 1
        err /= float(m)
        return err
    

#print(chr(27)+"[2J")
q15 = True

if q15:
    
    trainfn = 'hw4_knn_train.dat'
    traindata = np.loadtxt(trainfn)    
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]    
    y = traindata[:, n]
    
    k = 5

    knn = KNearestNeighbor(k)
    knn.train(X, y)
        
    Ein = knn.compute_err(X, y)
        
    testfn = 'hw4_knn_test.dat'
    testdata = np.loadtxt(testfn)    
    m, n = np.shape(testdata)    
    n = n - 1
    Xtest = testdata[:, 0:n]    
    ytest = testdata[:, n]
    Eout = knn.compute_err(Xtest, ytest)
            
    print "Ein", Ein, "Eout", Eout

