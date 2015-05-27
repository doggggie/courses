import numpy as np

def decision_stump(X, y, u):
    s, i, theta = 1, 0, 1.0
    bestEin = np.Inf
    
    m, n = np.shape(X)
    
    total_u = sum(u)
    total_plus = sum(u[y == 1])
    total_minus = total_u - total_plus
    
    for ival in range(n):
        idxsort = np.argsort(X[:, ival])
        xsorted = X[idxsort, ival]
        usorted = u[idxsort]
        ysorted = y[idxsort]
        
        count_fwd_plus = 0
        count_fwd_minus = 0
        count_bwd_plus = total_plus
        count_bwd_minus = total_minus
        
        for ithres in range(m+1):
            if ithres == 0:
                thres = -np.Inf
            elif ithres == m:
                thres = np.Inf
            else:
                thres = (xsorted[ithres] + xsorted[ithres-1]) * 0.5

            if ithres > 0:
                count_fwd_plus = count_fwd_plus + \
                                 (1 == ysorted[ithres-1]) * usorted[ithres-1]
                count_fwd_minus = count_fwd_minus + \
                                 (-1 == ysorted[ithres-1]) * usorted[ithres-1]
                count_bwd_plus = total_plus - count_fwd_plus
                count_bwd_minus = total_minus - count_fwd_minus    
            
            for sval in (-1, 1):
                if sval < 0:
                    Ein = (count_fwd_minus + count_bwd_plus) / float(total_u)
                else:
                    Ein = (count_fwd_plus + count_bwd_minus) / float(total_u)
                    
                if Ein < bestEin:
                    bestEin = Ein
                    s = sval
                    i = ival
                    theta = thres
                  
    # best Ein found, compute e and unew
    
    #print '  E^u_in', bestEin
    
    err_idx = (s * np.sign(X[:, i] - theta) != y)
    corr_idx = (s * np.sign(X[:, i] - theta) == y)
    e = sum(u[err_idx]) / total_u
    dmd = np.sqrt((1 - e) / e)
    unew = u.copy() 
    unew[err_idx] *= dmd
    unew[corr_idx] /= dmd  
                
    return s, i, theta, e, unew
    
def adaboost_stump(X, y, T=300):
    m, n = np.shape(X)
    
    slst = []
    ilst = []
    thetalst = []
    alphalst = []
    
    u = np.ones(m) / float(m)
    
    
    testfn = 'hw2_adaboost_test.dat'
    testdata = np.loadtxt(testfn)
    Xtest = testdata[:, 0:n]    
    ytest = testdata[:, n]
    mtest = np.shape(Xtest)[0]
    
    niter = 1
    while niter <= T:
        
        s, i, theta, e, u = decision_stump(X, y, u)
        
        Ein = sum(s * np.sign(X[:, i] - theta) != y) / float(m)
        
        Eout = sum(s * np.sign(Xtest[:, i] - theta) != ytest) / float(mtest)
        
        
        diamond = np.sqrt((1-e) / e)
        alpha = np.log(diamond)
        
        slst.append(s)
        ilst.append(i)
        thetalst.append(theta)
        alphalst.append(alpha)
        
        sumtmp = np.zeros(m)
        for t in range(len(slst)):
            sumtmp += alphalst[t] * slst[t] * \
                      np.sign(X[:, ilst[t]] - thetalst[t])
        Ein_G = sum(np.sign(sumtmp) != y) / float(m)
        
        sumtmp = np.zeros(mtest)
        for t in range(len(slst)):
            sumtmp += alphalst[t] * slst[t] * \
                      np.sign(Xtest[:, ilst[t]] - thetalst[t])
        Eout_G = sum(np.sign(sumtmp) != ytest) / float(mtest)
        
        print "#%d e %.2f Ein/Eout(gt) %.2f %.2f Ein/Eout(G) %.2f %.2f sum_u %.2g" \
              % (niter, e, Ein, Eout, Ein_G, Eout_G, sum(u))
        if Ein_G == 0.0:
            print 'Ein_G is zero'
        else:
            print 'Ein_G is NOT zero'
        #print 'u vec', u
        # '(s,i,th)', s, i, theta
        
        niter += 1
        
#def hw2q18():
if True:    
    testfn = 'hw2_adaboost_test.dat'
    trainfn = 'hw2_adaboost_train.dat'
    
    traindata = np.loadtxt(trainfn)
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]    
    y = traindata[:, n]
    
    adaboost_stump(X, y, 300)