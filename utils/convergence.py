import numpy as np

#Function to calculate order of convergence
def konvord(e):
    h = e[2:] / e[1:-1]
    n = np.log(h)
    d = np.log(e[1:-1]/e[:-2])
    p =  n/d
    print(np.sum(p))
    mw =np.sum(p)/len(p)
    return mw #

#Function to calculate rate of convergence (for linear convergence)
def konvrate(e):
    n = len(e)
    k = np.arange(0,n) #array mit 0,1,2,3,...,n-1 (Iterationsschritte)
    fit = np.polyfit(k,np.log(e),1)
    L = np.exp(fit[0])
    print(L)
    return L