import numpy as np
import matplotlib.pyplot as plot

    
def linregr_reg(X, y, reg=0.0):
    m, n = np.shape(X)
    
    Xnew = np.concatenate((X, np.eye(n) * np.sqrt(reg)), axis=0)
    ynew = np.concatenate((y, np.zeros(n)))
    
    wt = np.dot(np.linalg.pinv(Xnew), ynew)    
    return wt
    
def hw4q13(reg=10.0):
    
    testfn = 'ntumlone_hw4_hw4_test.dat'
    trainfn = 'ntumlone_hw4_hw4_train.dat'
    
    traindata = np.loadtxt(trainfn)
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]    
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    y = traindata[:, n]
    
    w = linregr_reg(X, y, reg)

    print w
    

    Ein = np.mean(np.sign(np.dot(X, w)) != y)
    print Ein
    
    testdata = np.loadtxt(testfn)
    
    m, n = np.shape(testdata)
    
    n = n - 1
    X = testdata[:, 0:n]
    
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    
    y = testdata[:, n]
    
    Eout = np.mean(np.sign(np.dot(X, w)) != y)
    print Eout
    
    return



def hw4q14():
    
    testfn = 'ntumlone_hw4_hw4_test.dat'
    trainfn = 'ntumlone_hw4_hw4_train.dat'
    
    
    minEin = 10
    minEout = 10
    minreg = 5
    

    
    
    for logreg in range(2, -11, -1):
        reg = 10 ** logreg
    
        traindata = np.loadtxt(trainfn)
    
        m, n = np.shape(traindata)    
        n = n - 1
        X = traindata[:, 0:n]    
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
        y = traindata[:, n]
    
        w = linregr_reg(X, y, reg)

    #print w
    

        Ein = np.mean(np.sign(np.dot(X, w)) != y)
    
        testdata = np.loadtxt(testfn)
    
        m, n = np.shape(testdata)
    
        n = n - 1
        X = testdata[:, 0:n]
    
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
    
        y = testdata[:, n]
    
        Eout = np.mean(np.sign(np.dot(X, w)) != y)
        print reg, Ein, Eout
    
        if Eout < minEout:
            minEin = Ein
            minreg = reg
            minEout = Eout
            
    print minreg, minEin, minEout
    
    return
    
    
    
    
def hw4q16():
    
    testfn = 'ntumlone_hw4_hw4_test.dat'
    trainfn = 'ntumlone_hw4_hw4_train.dat'
    
    
    minEin = 10
    minEout = 10
    minEcv = 10
    minreg = 5
    

    
    
    for logreg in range(2, -11, -1):
        reg = 10 ** logreg
    
        traindata = np.loadtxt(trainfn)
    
        _, n = np.shape(traindata)    
        n = n - 1
        X = traindata[0:120, 0:n]
        m, _ = np.shape(X)
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
        y = traindata[0:120, n]


    
            
        w = linregr_reg(X, y, reg)

    #print w
    

        Ein = np.mean(np.sign(np.dot(X, w)) != y)
    
    
        Xcv = traindata[120:, 0:n]
        m, _ = np.shape(Xcv)
        Xcv = np.concatenate((np.ones((m, 1)), Xcv), axis=1)
        ycv = traindata[120:, n]
        Ecv = np.mean(np.sign(np.dot(Xcv, w)) != ycv)
    
    
        testdata = np.loadtxt(testfn)
    
        m, n = np.shape(testdata)
    
        n = n - 1
        X = testdata[:, 0:n]
    
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
    
        y = testdata[:, n]
    
        Eout = np.mean(np.sign(np.dot(X, w)) != y)
        print reg, Ein, Ecv, Eout
    
        if Ecv < minEcv:
            minEin = Ein
            minreg = reg
            minEout = Eout
            minEcv = Ecv
            
    print minreg, minEin, minEcv, minEout
    
    return    
    
    
    
    
def hw4q19():
    
    trainfn = 'ntumlone_hw4_hw4_train.dat'
    traindata = np.loadtxt(trainfn)
    
    n = np.shape(traindata)[1]  
    n = n - 1
        

    minEcv = 10
    minreg = 5
    
    for logreg in range(2, -11, -1):
        reg = 10 ** logreg
    
    
        Ecv = 0.0
        for ifold in range(5):
    

            
            X = np.concatenate((traindata[0 : ifold * 40, 0:n],\
                                traindata[(ifold + 1) * 40 :, 0:n]),\
                               axis=0)
            
            m = np.shape(X)[0]
            X = np.concatenate((np.ones((m, 1)), X), axis=1)
            y = np.concatenate((traindata[0 : ifold * 40, n],\
                                  traindata[(ifold + 1) * 40 :, n]),\
                                  axis=0)
 
            w = linregr_reg(X, y, reg)
 
            #Ein = np.mean(np.sign(np.dot(X, w)) != y)
    
    
            Xcv = traindata[ifold * 40 : (ifold + 1) * 40, 0:n]
            m = np.shape(Xcv)[0]
            Xcv = np.concatenate((np.ones((m, 1)), Xcv), axis=1)
            ycv = traindata[ifold * 40 : (ifold + 1) * 40, n]
            Ecv += np.mean(np.sign(np.dot(Xcv, w)) != ycv)
    
      
        Ecv = Ecv / 5.0
   
        print reg, Ecv
    
        if Ecv < minEcv:
            minreg = reg
            minEcv = Ecv
            
    print minreg, minEcv
    
    return        