# Data science project for Business ====
# Project: Customer LifeTime Value ====
#=======================================
# Conda ENV: 

# Libraries ====
from scipy.optimize import minimize
from autograd import value_and_grad, hessian
import numpy as np
import pandas as pd
import hvplot.pandas
from utils.expected_clv import e_clv
from utils.neg_log_lik import _negative_log_likelihood, _loglikelihood
#Data Preparation stage =====
params = np.array((1,1)) #init params
n = 1000
active = [631, 482, 382, 326]
#active = [869, 743, 653, 593, 551, 517, 491]
# Data exploration Stage ====
_negative_log_likelihood(params, n, active)
#Machine Learning Stage =====
res = minimize(
        _negative_log_likelihood,
        args = (n, active), #parameters we do not optimize on
        tol=1e-13,
        x0= np.array((1,1)), #starting value of params
        bounds=[(0, None), (0, None)],
        options={'ftol' : 1e-100000000},
    )
res
def model_fit_survival(params, n, active):
    t = list(range(1, len(active)+1))
    df_plot = pd.DataFrame({'t': t,
                            'observed': [x/n for x in active],
                            'model': _loglikelihood(params, n, active, return_s=True)})
    return df_plot.hvplot('t',['observed', 'model'], title = 'Survival rate')
    # using init params (1,1)
print(model_fit_survival(params, n, active))
# using optimized params res.x
model_fit_survival(res.x, n, active)
alpha, beta = res.x
d = 0.1
net_cf = 100
clv = e_clv(alpha, beta, d, net_cf)
print("clv is ", clv)

