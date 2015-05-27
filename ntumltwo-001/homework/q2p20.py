from cStringIO import StringIO
import sys
import re

from svmutil import *
import numpy as np
import random

digit = 0

C = 0.1

def kernelGaussian(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(np.array(x1) - np.array(x2), 2) ** 2)
    
                
#def p20():
if True:    
    
    fn = 'C:\\shanying\\mooc\\ntumltwo-001\\features.train'
    
    alldata = np.loadtxt(fn)
    m, n = np.shape(alldata)
    
    y = [2 * int(alldata[i, 0] == digit) - 1 for i in range(m)]
    x = [list(alldata[i, 1:]) for i in range(m)]
    
    xa = np.array(x)
    ya = np.array(y)
    
    gammas = [1., 10., 100., 10000., 10000.]
    counts = {}
    
    for k in range(100):
        
        bestEval = np.Inf
        bestGamma = 0.0
        
        for gamma in gammas:
    
            validx = random.sample(range(m), 1000)
            trainidx = [i for i in range(m) if i not in validx]

            xtrain = list(xa[trainidx])
            ytrain = list(ya[trainidx])
            xtrain = [list(xtrain[i]) for i in range(len(xtrain))]
            xval = list(xa[validx])
            yval = list(ya[validx])
            xval = [list(xval[i]) for i in range(len(xval))]
    
            param = '-s 0 -t 2 -g %f -c %f -h 0 -q' % (gamma, C)
            model = svm_train(ytrain, xtrain, param)
    
            labels, accs, vals = svm_predict(yval, xval, model)
            Eval = sum(np.array(labels) != np.array(yval)) / float(len(yval))
            
            if Eval < bestEval:
                bestGamma = gamma
                bestEval = Eval
        
        print 'k', k, 'gamma', bestGamma, 'Eval', bestEval
        
        if bestGamma not in counts:
            counts[bestGamma] = 1
        else:
            counts[bestGamma] += 1
            
    print counts
#    return x,y,mdl