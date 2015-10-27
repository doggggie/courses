import numpy as np

def nzsign(num):
    sign = np.sign(num)
    if sign != 0:
        return sign
    else:
        return -1

def pocketpla(tbl, niter=100):
    m, n = np.shape(tbl)
    
    w0 = np.zeros(n)
    wt = w0[:]
    wbest = w0[:]
    ndiffbest = m
    
    x = np.concatenate((np.ones((m, 1)), tbl[:, :n-1]), 1)
    y = tbl[:, n-1]

    totcount = 0
    nupdate = 0
    
    lrate = 1.0

    while True:
        
        j = np.random.randint(0, m)
        totcount += 1
         
        if nzsign(np.dot(wt, x[j])) != nzsign(y[j]):
            # update
            wt = wt + lrate * y[j] * x[j]
            nupdate += 1
            
            # count the updated number of opposite signs
            ndiff = 0
            for i in range(m):
                if nzsign(np.dot(wt, x[i])) != nzsign(y[i]):
                    ndiff += 1
            if ndiff < ndiffbest:
                ndiffbest = ndiff
                wbest = wt[:]
        
        if nupdate == niter or totcount >= 10000 * m:
            break
        
##    print "total count", totcount
    return (totcount, wbest, wt, ndiffbest)


def errtest(tbl, wt):
    m, n = np.shape(tbl)    
    x = np.concatenate((np.ones((m, 1)), tbl[:, :n-1]), 1)
    y = tbl[:, n-1]
    ndiff = 0
    for i in range(m):
        if nzsign(np.dot(wt, x[i])) != nzsign(y[i]):
            ndiff += 1
    errrate = ndiff / float(m)
    print ndiff, m, errrate
    return errrate
    

def hw1p18():
    datafn = 'ntumlone_hw1_hw1_18_train.dat'
    datatbl = np.loadtxt(datafn)
    
    testfn = 'ntumlone_hw1_hw1_18_test.dat'
    testtbl = np.loadtxt(testfn)
    
    np.random.seed(0)

    # Q18
    niter = 50
    sumerr = 0.0
    ntest = 0 # 50 or 2000
    for i in range(ntest):
        print i, "       ",
        ct, wbest, wt, ndiff = pocketpla(datatbl, niter=niter)
        err = errtest(testtbl, wbest)
        sumerr += err
    #print "Q18\nerr", sumerr / float(ntest)
    
    # Q19
    niter = 50
    sumerr = 0.0
    ntest = 0 #2000
    for i in range(ntest):
        print i, "       ",
        ct, wbest, wt, ndiff = pocketpla(datatbl, niter=niter)
        err = errtest(testtbl, wt)
        sumerr += err
    #print "Q19\nerr", sumerr / float(ntest)
    
    # Q20
    niter = 100
    sumerr = 0.0
    ntest = 0 # 50 or 2000
    for i in range(ntest):
        print i, "       ",
        ct, wbest, wt, ndiff = pocketpla(datatbl, niter=niter)
        err = errtest(testtbl, wbest)
        sumerr += err
    #print "Q20\nerr", sumerr / float(ntest)

hw1p18()
