import math

def eb(N, dvc, eps):
    logres = math.log(4.0) + dvc * math.log(2*N) + (- eps**2 * N / 8.0)
    return math.exp(logres)
    
def eps_origVC(N, dvc, delta):
    return math.sqrt(8.0 / N * 
        (math.log(4.0) + 
         dvc * math.log(2*N) - 
         math.log(delta)) )
         
def eps_rademacher(N, dvc, delta):
    term1 = 2.0 / N *        (
        math.log(2.0 * N) + 
        dvc * math.log(N)    )
    term2 = 2 * math.log(1.0 / delta) / N
    return math.sqrt(term1) + math.sqrt(term2) + 1.0 / N

def vdb_helper(N, dvc, delta, eps):
    return math.sqrt(1.0 / N * (
        2 * eps + 
        math.log(6.0) + 
        dvc * math.log(2*N) - 
        math.log(delta)         ) )        
            
def eps_vdb(N, dvc, delta):
    eps_old = 1.0
    eps_new = vdb_helper(N, dvc, delta, eps_old)
    num_iter = 0
    while abs(eps_new - eps_old) > 1.0e-5 and num_iter < 1000:
        #print "eps", eps_old
        eps_old = eps_new
        eps_new = vdb_helper(N, dvc, delta, eps_old)
        num_iter += 1
    if num_iter >= 1000:
        return float("inf")
    #print "num_iter", num_iter
    return eps_new
    
def devroye_helper(N, dvc, delta, eps):
     return math.sqrt(1.0 / (2.0 * N) * (
        4 * eps * (1+eps) + 
        math.log(4.0) + 
        2 * dvc * math.log(N) - 
        math.log(delta)                  ) )
        
def eps_devroye(N, dvc, delta):
    eps_old = 1.0
    eps_new = devroye_helper(N, dvc, delta, eps_old)
    num_iter = 0
    while abs(eps_new - eps_old) > 1.0e-5 and num_iter < 1000:
        #print "eps", eps_old
        eps_old = eps_new
        eps_new = devroye_helper(N, dvc, delta, eps_old)
        num_iter += 1
    if num_iter >= 1000:
        return float("inf")
    #print "num_iter", num_iter
    return eps_new
    
def eps_variantvc(N, dvc, delta):
    return math.sqrt(16.0 / N * 
        (math.log(2.0) + 
         dvc * math.log(N) - 
         0.5 * math.log(delta)) )    