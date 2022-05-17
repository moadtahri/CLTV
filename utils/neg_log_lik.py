import pandas as pd
import numpy as np
def _loglikelihood(params, n, active, return_s=False):
    """
    params: alpha and beta initial values
    n: total number of people starting at t0
    actives: # of people stayed active at each time point starting at t1
    """
    t = list(range(1, len(active)+1))  # t: [ 1, 2, ...]
    alpha, beta = params
    p = []
    s = []
    lost = []
    ll = []
    for i in t:
        if i == 1:
            p.append(alpha/(alpha+beta))
            s.append(1- p[0])
            lost.append(n-active[0])
        else:
            p.append((beta+i-2)/(alpha+beta+i-1) * p[-1])
            s.append(s[-1] - p[i-1])
            lost.append(active[i-2] - active[i-1])

        ll.append(lost[i-1] * np.log(p[i-1]))
        #print('p: ', p[i-1])
    ll.append(active[-1] * np.log(s[-1]))

    #print('s: ', s[-1])
    if return_s== True:
        return s
    else:
        return ll

def _negative_log_likelihood(params, n, active):
    return -(np.sum(_loglikelihood(params, n, active)))
