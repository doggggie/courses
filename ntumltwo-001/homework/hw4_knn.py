import numpy as np
import sys

# d-M-1 NN w tanh type neurons (including the output neuron)
# use Squred error measure

DEBUG_PRT = False

def tanh(x):
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1
    
def dtanh(x):
    y = tanh(x)
    return 1 - y * y

def activate_func(x):
    return tanh(x)
    
def activate_func_deriv(x):
    return dtanh(x)
        

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
            dists[i] = np.norm(self.X[i,:] - X, 2)
            
                    
            
        x = np.concatenate((np.ones(1.0), X))
        for il in range(1, self.nl+1):
            s = np.dot(np.transpose(self.w[il]), x)
            if il < self.nl:
                # non-output layer
                x = np.concatenate((np.ones(1.0), activate_func(s)))
            else:
                # output layer
                x = activate_func(s)
                
        return x

    def compute_err(self, X, y):
        m, n = np.shape(X)
    
        err = 0.0
        for i in range(m):
            ypredict = self.predict(X[i,:])
            err += (ypredict - y[i]) ** 2
        err /= float(m)
        return err
    
Eins = {}
Eouts = {}

q15 = True    
q16 = False
q17 = False    

#print(chr(27)+"[2J")


if q15:
    
    np.random.seed(0)
    
    trainfn = 'hw4_nnet_train.dat'
    traindata = np.loadtxt(trainfn)    
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]    
    y = traindata[:, n]
    
    lrate = 0.01
    niter = 50000
    r = 0.1

    NEXP = 20    
            
    Eins = []
    Eouts = []
        
    for nexp in range(NEXP):
    
        wt_bds = (-r, r)
        nlayers = 3
        layers = {0:n, 1:8, 2:3, 3:1}
        
        nn = NeuralNetwork(nlayers, layers)
        nn.set_learning_rate(lrate)
        nn.set_num_iterations(niter)
        nn.set_weight_bounds(wt_bds)
        nn.train(X, y)
        
        Ein = nn.compute_err(X, y)
        
        testfn = 'hw4_nnet_test.dat'
        testdata = np.loadtxt(testfn)    
        m, n = np.shape(testdata)    
        n = n - 1
        Xtest = testdata[:, 0:n]    
        ytest = testdata[:, n]
        Eout = nn.compute_err(Xtest, ytest)

        Eins.append(Ein)
        Eouts.append(Eout)
            
        print "nexp", nexp, "Ein", Ein, "Eout", Eout

        # print out the average Eout up to now
        print "avgEin", sum(Eins)/float(len(Eins)), \
            "avgEout", sum(Eouts)/float(len(Eouts)), \
            "size", len(Eins)
                
        sys.stdout.flush()
 
        print "\n"
       
