import numpy as np
import random

DEBUG_PRT = False

class KMeans(object):
    
    def __init__(self, k):
        self.k = k
        self.mu = None
        self.nf = 0 # num of features
    
        
    
    def train(self, X):    
        m, n = np.shape(X)
        
        self.mu = X[random.sample(range(m), self.k), :]
        
        oldmu = self.mu.copy()
        
        self.inds = np.zeros(m)
        oldinds = self.inds.copy()
        
        dists = np.zeros(self.k)
        
        niter = 0
        
        while True:
            oldinds = self.inds.copy()
            for i in range(m):
                for j in range(self.k):
                    dists[j] = np.linalg.norm(X[i, :] - self.mu[j, :])
                self.inds[i] = np.argmin(dists)
            
            oldmu = self.mu.copy()
            for j in range(self.k):
                self.mu[j, :] = np.mean(X[self.inds == j, :], axis=0)
            
            if all(self.inds == oldinds) and (self.mu == oldmu).all():
                break
                
            niter += 1
            if niter % 100 == 0:
                print "niter = ", niter
                print "inds"
                print self.inds  
        
            
#        print niter
#        print self.inds
#        print self.mu        
                        
        err = 0.0
        for j in range(self.k):
            d = X[self.inds == j, :] - self.mu[j, :]
            err += sum(sum(d * d))
        return err / float(m)
    

#print(chr(27)+"[2J")
q19 = True

if q19:
    
    trainfn = 'hw4_kmeans_train.dat'
    traindata = np.loadtxt(trainfn)    
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]    
    y = traindata[:, n]
    
    k = 10
    Eins = 0.0

    for i in range(500):
        km = KMeans(k)
        Ein = km.train(X)
        print "i", i, "Ein", Ein
        Eins += Ein
        
    print "avgEin", Eins / 500.0
