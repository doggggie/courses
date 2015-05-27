import numpy as np
import sys

def sgn(x):
    s = np.sign(x)
    if len(np.shape(s)) == 0:
        if s == 0: 
            s = 1
    else:
        s[s==0] = 1
    return s

def checkEqual(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return np.all(first == rest for rest in iterator)
    except StopIteration:
        return True
        
def checkEqual2(iterator):
    return len(set(iterator)) <= 1        
        
def checkEqual3(lst):
    return np.all(lst[1:] == lst[:-1])

def gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    proportion_plus = sum(y == 1) / float(n)
    proportion_minus = 1 - proportion_plus
    return 1.0 - proportion_plus ** 2 - proportion_minus ** 2


class DTree(object):
    
    def __init__(self):
        self.tree_idx = -1
        self.tree = []
            
    def __train(self, X, y, oldtree):
        s, i, theta = 1, 0, 1.0
    
        m, n = np.shape(X)
   
        no_more_branch = False
        
        if checkEqual3(y):
            no_more_branch = True
            s = y[0]
            i = 0
            theta = -np.Inf #min(X[:,0]) - 1
        elif checkEqual3(X):
            no_more_branch = True
            yplus = sum(y == 1)
            yminus = sum(y == -1)
            s = 1 if yplus > yminus else -1
            i = 0
            theta = -np.Inf #X[0, 0] - 1
        
        if no_more_branch:        
            # return g_t(x) = Ein-optimal constant
            child1 = -1
            child2 = -1
            newtree = oldtree + [(s, i, theta, child1, child2)]
            treeidx = len(oldtree)
            return (treeidx, newtree)
    
        # 1. learn branching criteria
        # b(x) = argmin_decisionstumps sum |D_c|*impurity(D_c with h)
        min_impurity = np.Inf
    
        for ival in range(n):
        
            # sort by X(:, i)
            idxsort = np.argsort(X[:, ival])
            Xsorted = X[idxsort, ival]
            ysorted = y[idxsort]

            for kval in range(1, m): # theta
                
                y1 = ysorted[:kval]
                y2 = ysorted[kval:]
                
                impurity = len(y1) * gini(y1) + \
                           len(y2) * gini(y2)
                
                if impurity < min_impurity:
                    min_impurity = impurity
                    s = 0
                    i = ival
                    theta = (Xsorted[kval] + Xsorted[kval - 1]) / 2.0
                
        # 2. split D to 2 parts Dc = {(xn, yn): b(xn) = C}
        X1 = X[ X[:,i] >= theta, : ] 
        y1 = y[ X[:,i] >= theta ] 
        X2 = X[ X[:,i] < theta, : ] 
        y2 = y[ X[:,i] < theta ] 
        
        # 3. build sub-tree Gc <- dtree(Dc)
        child1, oldtree = self.__train(X1, y1, oldtree)
        child2, oldtree = self.__train(X2, y2, oldtree)
    
        # 4. return G(x) = sum |b(x) == c| * Gc(x)
        newtree = oldtree + [(s, i, theta, child1, child2)]
        treeidx = len(oldtree)
        return (treeidx, newtree)
        
    def train(self, X, y):
        oldtree = []
        treeidx, newtree = self.__train(X, y, oldtree)
        self.tree = newtree

    def getTree(self):
        return self.tree

    def predict(self, X):
        node = self.tree[-1] # tree root
        while node[0] == 0:            
            # follow tree to find leaf            
            i, theta = node[1], node[2]
            if X[i] >= theta:
                child = node[3]
            else:
                child = node[4]
            node = self.tree[child]
        
        # we are at leaf node now
        s, i, theta = node[0], node[1], node[2]
        y = s * sgn(X[i] - theta)
        return y

    def compute_err(self, X, y):
        m, n = np.shape(X)
    
        if np.shape(self.tree)[0] < 1 or m == 0:
            return 0.0
    
        ypredict = np.zeros(m)
        for i in range(m):
            ypredict[i] = self.predict(X[i,:])
        err = sum(ypredict != y) / float(m)
        return err
    

class DTreeOneLevel(object):

    def __init__(self):
        self.tree = None
    
    def train(self, X, y):

        s, i, theta = 1, 0, 1.0
    
        m, n = np.shape(X)
       
        # 1. learn branching criteria
        # b(x) = argmin_decisionstumps sum |D_c|*impurity(D_c with h)
        min_impurity = np.Inf
    
        for ival in range(n):
        
            # sort by X(:, i)
            idxsort = np.argsort(X[:, ival])
            Xsorted = X[idxsort, ival]
            ysorted = y[idxsort]

            for kval in range(0, m+1): # theta
                
                y1 = ysorted[:kval]
                y2 = ysorted[kval:]
                
                impurity = len(y1) * gini(y1) + \
                           len(y2) * gini(y2)
                
                if impurity < min_impurity:
                    min_impurity = impurity
                    i = ival
                    if kval == 0:
                        theta = -np.Inf
                    elif kval == m:
                        theta = np.Inf
                    else:
                        theta = (Xsorted[kval] + Xsorted[kval - 1]) / 2.0
                
        # 2. split D to 2 parts Dc = {(xn, yn): b(xn) = C}
        y1 = y[ X[:,i] >= theta ]     
        yplus = sum(y1 == 1)
        yminus = sum(y1 == -1)
        s1 = 1 if yplus > yminus else -1

        y2 = y[ X[:,i] < theta ] 
        yplus = sum(y2 == 1)
        yminus = sum(y2 == -1)
        s2 = 1 if yplus > yminus else -1

        self.tree = (s1, s2, i, theta)
    
    def getTree(self):
        return self.tree
        
    def predict(self, X):
        (s1, s2, i, theta) = self.tree
        if X[i] >= theta:
            return s1
        else:
            return s2

    def compute_err(self, X, y):
        m, n = np.shape(X)    
        ypredict = np.zeros(m)
        for i in range(m):
            ypredict[i] = self.predict(X[i,:])
        err = sum(ypredict != y) / float(m)
        return err
         

class RandomForest(object):
    
    def __init__(self, N, T, dtclass):
        self.N = N 
        self.T = T
        self.DTClass = dtclass
        self.trees = []
        
    def train(self, X, y):
        self.trees = []    
        m, n = np.shape(X)
    
        for t in range(T):
            sampleidx = np.random.random_integers(0, m - 1, N)
            Xsub = X[sampleidx, :]
            ysub = y[sampleidx]
            
            tree = self.DTClass()
            tree.train(Xsub, ysub)
            self.trees.append(tree)
        
    def predict(self, X):
        m, n = np.shape(X)
        ntree = len(self.trees)
    
        y = np.zeros(m)        
        ysample = np.zeros(ntree)    
    
        for j in range(m):
            for i, tree in enumerate(self.trees):
                ysample[i] = tree.predict(X[j,:])
            # voting
            y[j] = 1 if sum(ysample == 1) > sum(ysample == -1) else -1
    
        return y

    def compute_err(self, X, y):
        m, n = np.shape(X)
        ypredict = self.predict(X)
        err = sum(ypredict != y) / float(m)
        return err

    def compute_avg_dtree_err(self, X, y):
        err = 0.0
        for tree in self.trees:
            err += tree.compute_err(X, y)
        return err / float(len(self.trees))
            
        
#def hw3q13():
if True:
    
    q13 = False
    q14 = False
    q15 = False
    q16 = False
    q19 = True
    
    trainfn = 'hw3_train.dat'
    traindata = np.loadtxt(trainfn)    
    
    m, n = np.shape(traindata)    
    n = n - 1
    X = traindata[:, 0:n]    
    y = traindata[:, n]
    
    if q13 or q14 or q15:
        tree = DTree()
        tree.train(X, y)
        Ein = tree.compute_err(X, y)
        print "Q14 Ein", Ein
    
    if q15 or q16 or q19:
        testfn = 'hw3_test.dat'
        testdata = np.loadtxt(testfn)    
    
        m, n = np.shape(testdata)    
        n = n - 1
        Xtest = testdata[:, 0:n]    
        ytest = testdata[:, n]
    
    if q15:
        Eout = tree.compute_err(Xtest, ytest)    
        print "Q15 Eout", Eout

    if q16 or q19:
                
        if q16:
            DTClass = DTree
        else:
            DTClass = DTreeOneLevel
        
        N = np.shape(traindata)[0]
        T = 300
        Eins_dt = []
        Eins_rf = []
        Eouts_rf = []
        
        for i in range(10):
            forest = RandomForest(N, T, DTClass)
            forest.train(X, y)
            Ein_dt = forest.compute_avg_dtree_err(X, y)
            Ein_rf = forest.compute_err(X, y)
            Eout_rf = forest.compute_err(Xtest, ytest)
            
            print "*** ", i, Ein_dt, Ein_rf, Eout_rf, "***"
            sys.stdout.flush()
            
            Eins_dt.append(Ein_dt)
            Eins_rf.append(Ein_rf)
            Eouts_rf.append(Eout_rf)
        
        avg_Ein_dt = sum(Eins_dt) / float(len(Eins_dt))
        avg_Ein_rf = sum(Eins_rf) / float(len(Eins_rf))
        avg_Eout_rf = sum(Eouts_rf) / float(len(Eouts_rf))
        
        print "Q16/Q19 avg Ein_DT", avg_Ein_dt, \
              "avg Ein_RF", avg_Ein_rf, \
              "avg Eout_RF", avg_Eout_rf

        sys.stdout.flush()
        