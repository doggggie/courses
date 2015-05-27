from numpy import exp
import numpy as np

def E(u, v):
    return exp(u) + exp(2*v) + exp(u * v) + u*u - 2 * u * v + 2 * v * v - 3 * u - 2 * v

def grad(u, v):
    du = exp(u) + v * exp(u * v) + 2 * u - 2 * v - 3
    dv = 2 * exp(2 * v) + u * exp(u * v) - 2 * u + 4 * v - 2
    return du, dv
    
def hessian(u, v):
    ddu = exp(u) + v*v*exp(u*v) + 2
    dudv = v*u*exp(u*v) - 2
    ddv = 4 * exp(2*v) +u*u*exp(u*v) + 4
    return np.matrix([[ddu, dudv], [dudv, ddv]])
    
def update(nupdates, u0, v0, rate=1.0):
    uold, vold = u0, v0
    for i in range(nupdates):
        du, dv = grad(uold, vold)
        unew = uold - rate * du
        vnew = vold - rate * dv
        uold, vold = unew, vnew
        print i, unew, vnew, E(unew, vnew)
    return unew, vnew
    
def update_newton(nupdates, u0, v0, rate=1.0):
    uold, vold = u0, v0
    for i in range(nupdates):
        du, dv = grad(uold, vold)
        h = hessian(uold, vold)
        hinv = np.linalg.inv(h) 
        upvec = np.dot(hinv, np.matrix([[du], [dv]]))
        unew = uold - rate * upvec[0,0]
        vnew = vold - rate * upvec[1,0]
        uold, vold = unew, vnew
        print i, unew, vnew, E(unew, vnew)
    return unew, vnew    