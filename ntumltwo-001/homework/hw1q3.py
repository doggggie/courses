from scipy.optimize import fmin_slsqp
import numpy as np


xdata = np.array([[1, 0], [0, 1], [0, -1], \
                  [-1, 0], [0, 2], [ 0, -2], [-2, 0]])
ydata = np.array([-1, -1, -1, 1, 1, 1, 1])
ndata = np.shape(xdata)[0]

def kernel2(x1, x2):
    return (1 + np.dot(x1, x2)) ** 2

def f(alpha):
    val = 0.0
    for i in range(ndata):
        for j in range(ndata):
            val += alpha[i] * alpha[j] * \
                   ydata[i] * ydata[j] * \
                   kernel2(xdata[i], xdata[j]) * 0.5
    for i in range(ndata):
        val -= alpha[i]                       
    return val

def fp(alpha):
    val = np.zeros(ndata)
    for i in range(ndata):
        for j in range(ndata):
            val[i] += ydata[i] * ydata[j] * \
                      kernel2(xdata[i], xdata[j]) * alpha[j]
    val -= 1
    return val

def fh(alpha):
    val = np.zeros((ndata, ndata))
    for i in range(ndata):
        for j in range(ndata):
            val[i][j] = ydata[i] * ydata[j] * \
                        kernel2(xdata[i], xdata[j])
    return val

def f_eqcons(alpha):
    val = np.dot(alpha, ydata)
    return val

def fp_eqcons(alpha):
    val = ydata.copy()
    return val


a0 = np.zeros(ndata)
bds = []
for i in range(ndata):
    bds.append((0, np.Inf))
    
a = fmin_slsqp(f, a0, f_eqcons=f_eqcons, bounds=bds, fprime=fp, \
               fprime_eqcons=fp_eqcons, iter=100, acc=1e-06, \
               iprint=1, disp=None, full_output=0)
                 
#array([ -1.94122534e-15,   7.03700540e-01,   7.03700540e-01,
#         8.88880502e-01,   2.59260289e-01,   2.59260289e-01,
#        -7.35674474e-15])
                 
s = 1 # support vector index
b = ydata[s]
for i in range(ndata):
    b -= a[i] * ydata[i] * kernel2(xdata[i], xdata[s])

