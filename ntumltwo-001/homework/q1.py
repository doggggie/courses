from cStringIO import StringIO
import sys
import re

digit = 0

gamma = 100.0
Q = 2
C = 0.1

from svmutil import *
import numpy as np

def kernelGaussian(x1, x2):
#    gamma = 100.0
    return np.exp(-gamma * np.linalg.norm(np.array(x1) - np.array(x2), 2) ** 2)
    
def kernelLinear(x1, x2):
    return np.dot(x1, x2)
    
def kernelPolyn(x1, x2):
#    Q = 2
    return (1 + np.dot(x1, x2)) ** Q            

#C = 0.01
q15param = '-s 0 -t 0 -c %f -h 0 -q' % (C, )
q16param = '-s 0 -t 1 -d %d -g 1 -r 1 -c %f -h 0 -q' % (Q, C)
q18param = '-s 0 -t 2 -g %f -c %f -h 0 -q' % (gamma, C)
    
kernel = kernelGaussian
param = q18param
                
#def p16():
if True:    
    
    testfn = 'C:\\shanying\\mooc\\ntumltwo-001\\features.test'
    trainfn = 'C:\\shanying\\mooc\\ntumltwo-001\\features.train'
    
    traindata = np.loadtxt(trainfn)
    m, n = np.shape(traindata)
    nf = n - 1 # num of features
    
    y = [2 * int(traindata[i, 0] == digit) - 1 for i in range(m)]
    x = [list(traindata[i, 1:]) for i in range(m)]
        
    
    
    # redirect stdout to string
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    model = svm_train(y, x, param)

    # redirect output back to stdout    
    sys.stdout = old_stdout
    
    training_output = mystdout.getvalue()
    #print training_output
    '''
OUTPUT LOOKS LIKE THIS:
---------------------------------------
optimization finished, #iter = 1607
nu = 0.327527
obj = -23.144993, rho = 0.926864
nSV = 2487, nBSV = 2290
Total nSV = 2487
Accuracy = 83.6236% (6097/7291) (classification)
Accuracy = 82.1126% (1648/2007) (classification)
    '''

    obj = float(re.search(r'obj = ([\.\d\-\+e]*),', training_output).group(1))
    # rho should equal model.rho[0]
    rho = float(re.search(r'rho = ([\.\d\-\+e]*)', training_output).group(1))

    
    svinds = model.get_sv_indices()
    svcoefs = model.get_sv_coef()                
    
    testdata = np.loadtxt(testfn)
    mtest = np.shape(testdata)[0]
    ytest = [2 * int(testdata[i, 0] == digit) - 1 for i in range(mtest)]
    xtest = [list(testdata[i, 1:]) for i in range(mtest)]
    
    labels, accs, vals = svm_predict(y, x, model) # q16
    
    Ein = sum(np.array(labels) != np.array(y)) / float(len(y))
    
    suma = 0
    for i in range(model.get_nr_sv()):
        suma = suma + svcoefs[i][0] * y[svinds[i]-1]

    #print 'digit ', digit, '  Ein ', Ein, '  sum_alpha ', suma
    #return
    

    #wnorm2 = 0.0
    #for i in range(model.get_nr_sv()):
    #    for j in range(model.get_nr_sv()):
    #        k = kernel(x[svinds[i]-1], x[svinds[j]-1])
    #        wnorm2 += svcoefs[i][0] * svcoefs[j][0] * k
    #suma2 = 0.5 * wnorm2 - obj # suma & suma2 should equal and indeed they are
    
    wnorm2 = 2 * (suma + obj)
    
    labels, accs, vals = svm_predict(ytest, xtest, model) # q18
    Eout = sum(np.array(labels) != np.array(ytest)) / float(len(ytest))
    nsv = model.get_nr_sv()
             
    margin = 1.0 / np.sqrt(wnorm2)
    sumksi = (obj - 0.5 * wnorm2) / C                     
                                           
    print 'digit', digit, 'gm', gamma, ' Ein/Eout ', Ein, Eout, \
          '  sum_a ', suma, '#SV ', nsv, 'margin ', margin, \
          '  sum_ksi ', sumksi, 'obj ', obj, 'wnorm', np.sqrt(wnorm2)
                                
#    return x,y,mdl