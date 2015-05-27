import numpy as np
import sys

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
    
Eins = {}
Eouts = {}
            
q11 = False
q12 = True    
q13 = False
q14 = False    

#print(chr(27)+"[2J")
np.random.seed(5)

if q11:
    
    trainfn = 'hw4_nnet_train.dat'
    traindata = np.loadtxt(trainfn)    
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]    
    y = traindata[:, n]
    
#    M = 16
    lrate = 0.1
    niter = 50000
    r = 0.1

    NEXP = 50
            
    wt_bds = (-r, r)
    
#    Ms = (1, 6, 11, 16, 21)
    Ms = (6, 11)
    for M in Ms:
        Eins[M] = []
        Eouts[M] = []
        
    for nexp in range(NEXP):
    
        for M in Ms:
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

            Eins[M].append(Ein)
            Eouts[M].append(Eout)
            
            print "nexp", nexp, "M", M, "Ein", Ein, "Eout", Eout
            print "M", M, "avgEin", sum(Eins[M])/float(len(Eins[M])),\
                "avgEout", sum(Eouts[M])/float(len(Eouts[M]))

            sys.stdout.flush()
 
        print "\n"
        
    for M in Ms:
        print "M", M, "avgEin", sum(Eins[M])/float(len(Eins[M])),\
            "avgEout", sum(Eouts[M])/float(len(Eouts[M]))


if q12:
    
    trainfn = 'hw4_nnet_train.dat'
    traindata = np.loadtxt(trainfn)    
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]    
    y = traindata[:, n]
    
    M = 3
    lrate = 0.1
    niter = 50000
#    r = 0.1

    NEXP = 500    
            
    
    #rs = (0,0.001,0.1,10,1000)
    rs = (0.001, 0.1)

    for r in rs:
        Eins[r] = []
        Eouts[r] = []
        
    for nexp in range(NEXP):
    
        for r in rs:
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

            Eins[r].append(Ein)
            Eouts[r].append(Eout)
            
            print "nexp", nexp, "r", r, "Ein", Ein, "Eout", Eout

            # print out the average Eout up to now
            print "r", r, "avgEin", sum(Eins[r])/float(len(Eins[r])), \
                "avgEout", sum(Eouts[r])/float(len(Eouts[r])), \
                "size", len(Eins[r])
                
            sys.stdout.flush()
 
        print "\n"
        
    for r in rs:
        print "r", r, "avgEin", sum(Eins[r])/float(len(Eins[r])), \
            "avgEout", sum(Eouts[r])/float(len(Eouts[r])), \
            "size", len(Eins[r])


if q13:
    
    trainfn = 'hw4_nnet_train.dat'
    traindata = np.loadtxt(trainfn)    
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]    
    y = traindata[:, n]
    
    M = 3
#    lrate = 0.1
    niter = 50000
    r = 0.1

    NEXP = 20    
            
    
    lrates = (0.001, 0.01, 0.1, 1.0, 10.0)

    for lrate in lrates:
        Eins[lrate] = []
        Eouts[lrate] = []
        
    for nexp in range(NEXP):
    
        for lrate in lrates:
            
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

            Eins[lrate].append(Ein)
            Eouts[lrate].append(Eout)
            
            print "nexp", nexp, "lrate", lrate, "Ein", Ein, "Eout", Eout

            # print out the average Eout up to now
            print "lrate", lrate, \
                "avgEin", sum(Eins[lrate])/float(len(Eins[lrate])), \
                "avgEout", sum(Eouts[lrate])/float(len(Eouts[lrate])), \
                "size", len(Eins[lrate])
                
            sys.stdout.flush()
 
        print "\n"
        
    for lrate in lrates:
        print "lrate", lrate, "avgEin", sum(Eins[lrate])/float(len(Eins[lrate])), \
            "avgEout", sum(Eouts[lrate])/float(len(Eouts[lrate])), \
            "size", len(Eins[lrate])

if q14:
    
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
       
