import numpy as np

if True:
    A = np.eye(10)
    b = np.array([-2.0, -6.0, -9.0, 1.0, 8.0, 10.0, 1.0, 
                -9.0, -4.0, -3.0])
    for i in range(10):
        for j in range(10):
            A[i][j] += np.sin((i+1) + (j+1))
    normA = np.linalg.norm(A, ord=2, axis=0)
    for i in range(10):
        A[:, i] /= normA[i]
            
    s = 3
    x = np.zeros(10)
    r = b.copy()
    inds = []
    
    for t in range(1, s+1):
        maxidx = -1
        maxval = -1.0
        
        for j in range(10):
            if j not in inds:
                xj = abs(np.dot(A[:, j], r))
                if xj > maxval:
                    maxidx = j
                    maxval = xj
        inds.append(maxidx)
        Anew = np.zeros((10, t))
        for j in range(t):
            Anew[:,j] = A[:,inds[j]]
        xnew = np.dot(np.linalg.pinv(Anew), b)
        
        Axnew = np.dot(Anew, xnew)
        r = b.copy()
        for j in range(t):
            r[inds[j]] -= Axnew[j]
        
        x = np.zeros(10)
        for j in range(t):
            x[inds[j]] = xnew[j] 
        