import numpy as np
import sys
import multiprocessing

from datetime import datetime

# d-M-1 NN w tanh type neurons (including the output neuron)
# use Squred error measure

DEBUG_PRT = False

def sgn(x):
    s = np.sign(x)
    if len(np.shape(s)) == 0:
        if s == 0: 
            s = 1
    else:
        s[s==0] = 1
    return s
    
def tanh(x):
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1
    
def dtanh(x):
    y = tanh(x)
    return 1 - y * y

def activate_func(x):
    return tanh(x)
    
def activate_func_deriv(x):
    return dtanh(x)
        

class NeuralNetwork(object):
    
    # d-M-1 has 2 layers, and layers = {0:d, 1:M, 2:1}
    # assume the output layer has d_L = 1
    
    def __init__(self, nl, layers):
        self.nl = nl
        self.l = layers.copy()
        self.w = {}
        self.d = {}
        self.x = {}
        self.s = {}
        self.learning_rate = 0.1
        self.niter = 0
        self.weight_bds = (-1.0, 1.0)
    
        # allocate memory
        # weights is allocated in init_weights
#        for il in range(1, nl+1):
#            self.weights[il] = np.zeros(np.shape(layers[il-1] + 1, \
#                                                 layers[il]))
        for il in range(nl+1):
            self.x[il] = np.zeros(layers[il] + 1)
            if il > 0:
                self.d[il] = np.zeros(layers[il])
                self.s[il] = np.zeros(layers[il])
    
    
    def set_learning_rate(self, lrate):
        self.learning_rate = lrate
    
    def set_num_iterations(self, niter):
        self.niter = niter    
    
    def set_weight_bounds(self, bounds):
        self.weight_bds = bounds
        
    def init_weights(self):
        for il in range(1, self.nl+1):
            self.w[il] = np.reshape( \
                np.random.uniform(self.weight_bds[0], \
                                  self.weight_bds[1], \
                                  (self.l[il-1] + 1) * self.l[il]),
                (self.l[il-1] + 1, self.l[il]) )


    def train(self, X, y):    
        m, n = np.shape(X)
        
        # check if n == self.layers[0]
        # check if y's shape is consistent with self.layers[self.nlayer]
   
        self.init_weights()

        if DEBUG_PRT:
            print 'weights', self.w, "\n\n"
        
        
        for iter in range(self.niter):
            # 1. stochastic: randomly pick 0 <= n < M
            idx = np.random.randint(m)
            
            # 2. forward: compute x_i^l's, with x^0 = X
#            self.xs[0] = np.concatenate((np.ones(1.0), X[idx, :]))
            self.x[0][0] = 1.0
            self.x[0][1:] = X[idx, :]
            for il in range(1, self.nl+1):
                self.s[il] = np.dot(np.transpose(self.w[il]), \
                                    self.x[il-1])
                                         
                if il < self.nl:
                    # non-output layer
                    self.x[il][0] = 1.0
                    self.x[il][1:] = activate_func(self.s[il])
                else:
                    # output layer
                    self.x[il] = activate_func(self.s[il])
                
            # 3. backward: compute delta_j_l's, with x^0 = X
            # output layer
            self.d[self.nl] = -2.0 * (y[idx] - self.x[self.nl]) * \
                              activate_func_deriv(self.s[self.nl])
            # other layer
            for il in range(self.nl-1, 0, -1):
                self.d[il] = np.dot(self.w[il+1][1:, :], self.d[il+1]) * \
                             activate_func_deriv(self.s[il])
            
            # 4. gradient descent: update w_ij^l's
            for il in range(1, self.nl+1):
                self.w[il] -= self.learning_rate * \
                            np.outer(self.x[il-1], self.d[il])
            
            
            if DEBUG_PRT:
                print "\n\n"
                print 'iter', iter
                print 'idx', idx, 'X', X[idx, :], 'y', y[idx], '\n'
                print 'x0\n', self.x[0]
                print 'x1\n', self.x[1]
                print 'x2\n', self.x[2]
                print "\n\n"
                print 'd1\n', self.d[1]
                print 'd2\n', self.d[2]
                print "\n\n"
                print 's1\n', self.s[1]
                print 's2\n', self.s[2]
                print "\n\n"
                print 'w1\n', self.w[1]
                print 'w2\n', self.w[2]
            
        # return gNNET(x)
   
   
    def predict(self, X):
        x = np.concatenate((np.ones(1.0), X))
        for il in range(1, self.nl+1):
            s = np.dot(np.transpose(self.w[il]), x)
            if il < self.nl:
                # non-output layer
                x = np.concatenate((np.ones(1.0), activate_func(s)))
            else:
                # output layer
                x = activate_func(s)
        y = sgn(x)
        return y

    def compute_err(self, X, y):
        m, n = np.shape(X)
    
        err = 0.0
        for i in range(m):
            ypredict = self.predict(X[i,:])
            if ypredict != y[i]:
                err += 1
        err /= float(m)
        return err

    
def experiment_once(lrate, niter, r, M):
    
    seed = datetime.now().microsecond/1000
    np.random.seed(seed)

    
    trainfn = 'hw4_nnet_train.dat'
    traindata = np.loadtxt(trainfn)    
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]
    y = traindata[:, n]
    
    wt_bds = (-r, r)
    
    nlayers = 2
    layers = {0:n, 1:M, 2:1}
        
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
            
    print "seed", seed, "M", M, "lrate", lrate, "r", r, "Ein", Ein, "Eout", Eout
    sys.stdout.flush()
    
    return (lrate, niter, r, M, Ein, Eout)

                
Eins = {}
Eouts = {}

#print(chr(27)+"[2J")
#np.random.seed(5)

if __name__ == "__main__":
    
    lrate = 0.1
    niter = 50000
    
    
    #r = 0.1
    rs = (0.001, 0.1)

    NEXP = 500    

    #Ms = (6, 11)
    M = 3
    
    pool = multiprocessing.Pool(processes=4)
    results = []
    
    #for M in Ms:
    for r in rs:
        Eins[r] = []
        Eouts[r] = []

    for nexp in range(NEXP):
        for r in rs:
            res = pool.apply_async(experiment_once, (lrate, niter, r, M))
            results.append(res)
            (lrate, niter, r, M, Ein, Eout) = res.get()
            Eins[r].append(Ein)
            Eouts[r].append(Eout)
            print "nexp", nexp, "r", r, \
                "Ein/Eout", Ein, Eout, \
                "avgEin/Eout", \
                sum(Eins[r])/float(len(Eins[r])),\
                sum(Eouts[r])/float(len(Eouts[r]))
            sys.stdout.flush()
 
    pool.close()
    pool.join()
    
    for r in rs:
        print "r", r, "avgEin", sum(Eins[r])/float(len(Eins[r])),\
            "avgEout", sum(Eouts[r])/float(len(Eouts[r]))



if False: #q14
    
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
       
