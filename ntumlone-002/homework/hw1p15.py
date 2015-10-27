import numpy as np

def nzsign(num):
    sign = np.sign(num)
    if sign != 0:
        return sign
    else:
        return -1

def pla(tbl, perm=False, lrate=1.0):
    m, n = np.shape(tbl)
    w0 = np.zeros(n)
    wt = w0[:]
    x = np.concatenate((np.ones((m, 1)), tbl[:, :n-1]), 1)
    y = tbl[:, n-1]
    count = 0
    totcount = 0
    slist = []
    if perm:
        slist = np.random.permutation(range(m))
    else:
        slist = range(m)
    i = 0
    noupdate = 0
    while True:
        if np.mod(i, m) == 0:
            i = 0
        j = slist[i]
         
        if nzsign(np.dot(wt, x[j])) != nzsign(y[j]):
            # update
            wt = wt + lrate * y[j] * x[j]
            totcount += 1
            noupdate = 0
        else:
            noupdate += 1
        
        if noupdate == m:
            break
        
        i += 1

##    print "total count", totcount
    return (totcount, wt)

def hw1p15():
    fn = 'ntumlone_hw1_hw1_15_train.dat'
    datatbl = np.loadtxt(fn)
    
    # Q15
    ct, w = pla(datatbl, perm=False, lrate=1.0)
    print "Q15\ncount: ", ct
    
    np.random.seed(0)
    
    # Q16
    sumct = 0
    for i in range(2000): 
        ct, w = pla(datatbl, perm=True, lrate=1.0)  
        sumct += ct
    print "Q16\navg ct: ", float(sumct)/2000.0
            
    # Q17
    sumct = 0
    for i in range(2000): 
        ct, w = pla(datatbl, perm=True, lrate=0.5)  
        sumct += ct
    print "Q17\navg ct: ", float(sumct)/2000.0


hw1p15()
