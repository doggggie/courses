import numpy as np

def nzsign(num):
    sign = np.sign(num)
    if sign != 0:
        return sign
    else:
        return -1
        
def gendata():
    x = 2 * np.random.ranf() - 1
    y = nzsign(x)
    fliprand = np.random.ranf()
    if fliprand < 0.2:
        y = -y
 
    return x, y
    
def genNdata(N):
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        x[i], y[i] = gendata()
        
    return x, y

def decision_stump_hypo(x, s, theta):
    return s * np.sign(x - theta)    
            
def solve_ds(x, y, N):
    sortedx = np.sort(x)
    bests, besttheta, bestEin = 0, 0.0, 1.0
    
    for i in range(N+1):
        if i == 0:
            theta = (-1 + sortedx[i]) / 2.0
        elif i == N:
            theta = (1 + sortedx[N-1]) / 2.0
        else:
            theta = (sortedx[i] + sortedx[i-1]) / 2.0
            
        for s in [-1, 1]:
            h = decision_stump_hypo(x, s, theta)
            Ein = np.sum(h != y) / float(N)
            if Ein < bestEin:
                bests, besttheta, bestEin = s, theta, Ein
    
    return bests, besttheta, bestEin
    
def calcEout(s, theta):
    return 0.5 + 0.3 * s * (abs(theta) - 1)
    
def q17():

    np.random.seed(0)
    
    N = 20
    Ntest = 5000
    
    avgEin = 0.0
    avgEout = 0.0
    
    for tidx in range(Ntest):
        x, y = genNdata(N)
        
        s, theta, Ein = solve_ds(x, y, N)
        
        Eout = calcEout(s, theta)
        
#        print tidx, s, theta, Ein, Eout
        
        avgEin = avgEin + Ein
        avgEout = avgEout + Eout

    avgEin = avgEin / float(Ntest)
    avgEout = avgEout / float(Ntest)
    
    print "average Ein", avgEin, "Eout", avgEout
    
    
def solve_ds_nd(x, y, N, d):
    
    bests, besttheta, bestEin = 0, 0.0, 1.0
    bestdim = -1
    
    for dim in range(d):
    
        s, theta, Ein = solve_ds(x[:, dim], y, N)
        
        print dim, s, theta, Ein
        
        if Ein < bestEin:
            bests, besttheta, bestEin = s, theta, Ein
            bestdim = dim
    
    return bests, besttheta, bestEin, bestdim

def calEtest_nd(x, y, N, s, theta, dimidx):
    h = decision_stump_hypo(x[:, dimidx], s, theta)
    return np.sum(h != y) / float(N)
    

def q19():
#    fn = 'C:\\Users\\Administrator\\Desktop\\gg\\ntumlone\\hw1_15_train.dat'
    trainfn = 'ntumlone_hw2_hw2_train.dat'
    traindata = np.loadtxt(trainfn)
    m, n = np.shape(traindata)
    y = traindata[:, n-1]
    x = traindata[:, 0:n-1]
    N = m
    d = n-1
    s, theta, Ein, dim = solve_ds_nd(x, y, N, d)
    print 'Ein', Ein, 's', s, 'theta', theta, 'dim', dim
    
    testfn = 'ntumlone_hw2_hw2_test.dat'
    testdata = np.loadtxt(testfn)
    m, n = np.shape(testdata)
    y = testdata[:, n-1]
    x = testdata[:, 0:n-1]
    N = m
    d = n-1
    Etest = calEtest_nd(x, y, N, s, theta, dim)
    print "Etest", Etest
    