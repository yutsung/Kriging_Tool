#!/usr/local/bin/python3
"""
"""
import datetime
from functools import wraps

import numpy as np
from scipy.special import comb

from .geocoordinate import earth_radius, haversine


def count_runtime(func):
    @wraps(func)
    def with_runtime(*args, **kwargs):
        starttime = datetime.datetime.now()
        out_func = func(*args, **kwargs)
        endtime = datetime.datetime.now()
        print(
            '{}, runtime: {}'.format(
                func.__name__,
                endtime - starttime
            )
        )
        return out_func
    return with_runtime


def exp_model(s, d, l):
    return np.power(s, 2) * (1 - np.exp(-d/l))


@count_runtime
def semivariance_lonlat(x, y, z):
    dis = np.zeros(int(comb(np.size(z), 2))) * np.nan
    zdif = np.zeros(int(comb(np.size(z), 2))) * np.nan

    k = 0
    for i ,zi in enumerate(z[1:], 1):
        dis[k:k+i] = haversine(np.repeat(x[i], i), x[:i], np.repeat(y[i], i), y[:i])
        zdif[k:k+i] = 0.5 * np.power(zi - z[:i], 2)
        k += i


    zdif = [zdifi for _, zdifi in sorted(zip(dis, zdif))]
    dis = sorted(dis)
    zdif = np.array(zdif)
    dis = np.array(dis)

    step = (max(dis) /2 / 20)
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


@count_runtime
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


@count_runtime
def do_blue(lon, lat, z, ilon_list, ilat_list, param, fit_model):
    """Best Linear Unbiased Estimation
    """

    if np.size(ilon_list) == 1:
        ilon_list = [ilon_list]
        ilat_list = [ilat_list]
    ivalue = np.zeros((np.size(ilon_list))) * np.nan

    A = np.asmatrix(np.zeros((len(z)+1, len(z)+1)))
    A[:, -1] = 1
    A[-1, :] = 1
    A[-1, -1] = 0

    if fit_model == 'exponential':
        
        for i in range(len(z)):
            d_ij = haversine(
                np.repeat(lon[i], len(z)), 
                lon[:], 
                np.repeat(lat[i], len(z)), 
                lat[:]
            )
            gamma_d_ij = exp_model(param[0], d_ij, param[1])
            A[i, :-1] = gamma_d_ij
    
    for ipoint, (ilon, ilat) in enumerate(zip(ilon_list, ilat_list)):
        print(ilon)
        print(ilat)
        Gamma = np.asmatrix(np.zeros((len(z)+1, 1)))
        Gamma[-1, 0] = 1
        for i in range(len(z)):
            d_i0 = haversine(lon[i], ilon, lat[i], ilat)
            gamma_d_i0 = exp_model(param[0], d_i0 ,param[1])
            Gamma[i, 0] = gamma_d_i0

        Lambda = A.I * Gamma
        ivalue[ipoint] = np.sum( np.asarray(Lambda.T) * np.append(z, 1) )

    return ivalue


def ordinary_kriging(
        lon, 
        lat, 
        z, 
        ilon_list, 
        ilat_list, 
        fit_model='exponential'
    ):
    
    dis, zdif, dis_ma, zdif_ma = semivariance_lonlat(lon, lat, z)
   
    param = fit_semivariance(dis_ma, zdif_ma, fit_model)

    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots()
    ax.scatter(dis, zdif, alpha=0.5)
    ax.plot(dis_ma, zdif_ma, 'r-')
    ax.plot(
        np.arange(0, np.max(dis_ma)), 
        exp_model(param[0], np.arange(0, np.max(dis_ma)), param[1]), 
        'b-'
    )
    plt.savefig('semivar.png')

    ivalue = do_blue(
        lon, 
        lat, 
        z, 
        ilon_list, 
        ilat_list, 
        param, 
        fit_model=fit_model
    )
    
    return ivalue
    

