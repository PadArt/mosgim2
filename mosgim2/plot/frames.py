import numpy as np
import datetime
import scipy.special as sp
from datetime import timedelta
from mosgim2.coords.coords import geo2mag, geo2lt, geo2modip


def make_matrix(nbig, mbig, theta, phi):
    n_ind = np.arange(0, nbig + 1, 1)
    m_ind = np.arange(-mbig, mbig + 1, 1)
    M, N = np.meshgrid(m_ind, n_ind)
    Y = sp.sph_harm(np.abs(M), N, 0, 0)
    idx = np.isfinite(Y)
    M = M[idx]
    N = N[idx]
    n_coefs = len(M)
    matrix = np.zeros((len(theta), n_coefs))
    for i in range(0, len(theta), 1):
        Ymn = sp.sph_harm(np.abs(M), N, theta[i], phi[i])
        a = np.zeros(len(Ymn))
        a[M < 0] = Ymn[M < 0].imag * np.sqrt(2) * (-1.) ** M[M < 0]
        a[M > 0] = Ymn[M > 0].real * np.sqrt(2) * (-1.) ** M[M > 0]
        a[M == 0] = Ymn[M == 0].real
        matrix[i, :] = a[:]
    return matrix


def makeframes(lon, colat, coord, nbig, mbig, coefs, ts):

    time = np.array([datetime.datetime.utcfromtimestamp(float(t)) for t in ts])

    lon_m, colat_m = np.meshgrid(lon, colat)

    Zl = []

    for k in np.arange(0,len(coefs),1): # consecutive tec map number
        print(time[k])
        if coord == 'geo':
            mcolat, mt = geo2lt(np.deg2rad(colat_m.flatten()), np.deg2rad(lon_m.flatten()), time[k]) 
        elif coord == 'mag':        
            mcolat, mt = geo2mag(np.deg2rad(colat_m.flatten()), np.deg2rad(lon_m.flatten()), time[k]) 
        elif coord == 'modip':        
            mcolat, mt = geo2modip(np.deg2rad(colat_m.flatten()), np.deg2rad(lon_m.flatten()), time[k]) 

        Atest = make_matrix(nbig, mbig, mt, mcolat)
        Z = np.dot(Atest, coefs[k]).reshape(len(colat), len(lon))
        Zl.append(Z)

    return Zl


