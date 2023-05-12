# Trend and cycle decomposition

# standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io                      # load data
from scipy.optimize import minimize  # minimize a function
from scipy.stats import chi2
import statsmodels.api as sm

# my functions
from my_functions import *

#Load quarterly US data for the period 1940:1 to 2000:4 (T = 244)
# y = scipy.io.loadmat('lfac_usgdp.mat')['RGDP'] 
data = pd.read_csv('ibc.csv', decimal=',')

# print(y.head())
# y = data['ibc'].values.reshape((len(data),1))
y = pd.read_csv("selic.txt").values.reshape((len(pd.read_csv("selic.txt")),1))
# y = np.log(y) - np.mean(np.log(y))
# y = 100*y

b_hpfilter = [1,1/40]

(Λ, Φ, R, Q)=state_space(b_hpfilter)
s11, ss, lnl = kalman(Λ, Φ, R, Q, y.T)

start_values = [1,1]
bhat = minimize(neglog, start_values, args=y, method='BFGS').x

print('\nParameter estimates')
print(pd.DataFrame(bhat, index=['sigma_c', 'sigma_z'], columns=['bhat']))

bhat_const = minimize(neglog_const, [1,1], args=y, method='BFGS').x

print('\nParameter estimates')
print(pd.DataFrame(bhat_const, index=['sigma_c', 'sigma_z'], columns=['bhat_const']))

print('loglikilihood function (unconstrained) = ', round(-neglog(bhat,y), 5))
print('loglikilihood function (constrained)   = ', round(-neglog(bhat_const,y), 5))
print('loglikilihood function (HP filter)     = ', round(-neglog(b_hpfilter,y), 5))

# loglikilihood ratio

lf0 = -neglog(b_hpfilter,y) # restricted model
lf1 = -neglog(bhat_const,y) # unrestricted model

LR = -2*(lf0-lf1)
print('LR statistic    = ', round(LR, 4))
print('p-value         = ', round(chi2.sf(x=LR, df=1), 4))
# print('p-value         = ', 1-chi2.cdf(x=LR, df=1))

(Λ, Φ, R, Q)=state_space(b_hpfilter)
s11, ss, lnl = kalman(Λ, Φ, R, Q, y.T)

# obtain the trend via standard hp_filter
cycle, trend = sm.tsa.filters.hpfilter(y, 1600)

# plot some figures
fig, ax = plt.subplots(figsize=(10,8))

ax.plot(y[0:], '--', c='k', label="US real GDP x 100")
ax.plot(ss[0,0:], c='k', label="Trend via kalman filter representation of HP fiter")
ax.plot(trend[0:], c='r', alpha=0.3, lw=3, label="Trend via standard HP filter")
ax.legend(loc='best')
plt.show() 

fig, ax = plt.subplots(nrows= 2, figsize=(15,5))

# cycle
ciclo_onesided = np.squeeze(y[0:]) - s11[0,0:] # ciclo y observado - estado filtrado (one-sided filter)
ciclo_twosided = np.squeeze(y[0:]) - ss[0,0:] # ciclo y observado - estado suavizado (two-sided filter)

ax[0].plot(ciclo_onesided, c='k', label="IBC-BR x 100")
ax[1].plot(ciclo_twosided, c='k', label="IBC-BR x 100")

# plt.show()

data2export = pd.DataFrame(ciclo_onesided, index=data['Data'], columns=['one-sided-hp'])

data2export['two-sided-hp'] = ciclo_twosided

data2export.to_csv('output-gap.csv')
print(data2export)