#!
"""
"""

import numpy as np
from scipy.special import comb

from .geocoordinate import earth_radius, haversine


def exp_model(s, d, l):
    return s**2 * (1 - np.exp(-d/l))


def semivariance_lonlat(x, y, z):
    dis = np.zeros(int(comb(np.size(z), 2))) * np.nan
    zdif = np.zeros(int(comb(np.size(z), 2))) * np.nan

    k = 0
    for i, zi in enumerate(z):
        for j, zj in enumerate(z):
            if i <= j:
                continue
            dis[k] = haversine(x[i], x[j], y[i], y[j])
            zdif[k] = 0.5 * (zi - zj)**2
            k += 1

    zdif = [zdifi for _, zdifi in sorted(zip(dis, zdif))]
    dis = sorted(dis)
    zdif = np.array(zdif)
    dis = np.array(dis)

    step = (max(dis) /2 / 20)
    #step = (max(dis) / 20)
    dis_ma = []
    zdif_ma = []
    for i in range(20):
        dis_inzone = dis[(dis > i*step)&(dis < (i+1)*step)]
        zdif_inzone = zdif[(dis > i*step)&(dis < (i+1)*step)]
        if len(zdif_inzone) == 0:
            continue
        zdif_ave_inzone = np.mean(zdif_inzone)
        dis_ma.append( (i+0.5) * step )
        zdif_ma.append( zdif_ave_inzone )

    return dis, zdif, dis_ma, zdif_ma


def fit_semivariance(x, y, fit_model):

    param = []

    if fit_model == 'exponential':
  
        fit_err = np.ones((20)) * 999
        var_list = np.ones((20)) * np.nan
        for i in range(20):
            di = i+1
            li = np.array(30)**(0.05 * di)

            X = np.asmatrix(1 - np.exp(-1 * (x / li))).T
            Y = np.asmatrix(y).T
       
            A = (X.T * X).I * X.T * Y

            var = A[[0]]
            var_list[i] = var
            fit_err[i] = np.sum(np.abs(y - var * (1 - np.exp(-1 * (x / li)))))

    min_idx = np.argmin(fit_err)
    di = min_idx+1
    lengh = np.array(30)**(0.05 * di)
    var = var_list[min_idx]

    param.append(np.sqrt(var))
    param.append(lengh)

    return param

def ordinary_kriging(lon, lat, z, ilon, ilat, fit_model='exponential'):
    
    dis, zdif, dis_ma, zdif_ma = semivariance_lonlat(lon, lat, z)
   
    param = fit_semivariance(dis_ma, zdif_ma, fit_model)

    #import matplotlib.pyplot as plt 
    #fig, ax = plt.subplots()
    #ax.scatter(dis, zdif, alpha=0.5)
    #ax.plot(dis_ma, zdif_ma, 'r-')
    #ax.plot(
    #    np.arange(0, np.max(dis_ma)), 
    #    exp_model(param[0], 
    #    np.arange(0, np.max(dis_ma)), param[1]), 
    #    'b-'
    #)
    #plt.show()

    
    
    
    

