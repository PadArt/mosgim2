import numpy as np
from scipy.interpolate import RectSphereBivariateSpline
from mosgim2.utils.time_utils import sec_of_day
from mosgim2.consts.phys_consts import POLE_PHI, POLE_THETA, RE  


# Geodetic to Geomagnetic transform: http://www.nerc-bas.ac.uk/uasd/instrums/magnet/gmrot.html
GEOGRAPHIC_TRANSFORM = np.array([
    [np.cos(POLE_THETA)*np.cos(POLE_PHI), np.cos(POLE_THETA)*np.sin(POLE_PHI), -np.sin(POLE_THETA)],
    [-np.sin(POLE_PHI), np.cos(POLE_PHI), 0],
    [np.sin(POLE_THETA)*np.cos(POLE_PHI), np.sin(POLE_THETA)*np.sin(POLE_PHI), np.cos(POLE_THETA)]
])

def xyz2spher(x,y,z):
    '''
    ECEF X,Y,Z [m] to spherical L(longitude), B(latitude) [rad], H (height above sphere) [m]
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    
    Q = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # H - height
    H = Q - RE

    # L - longitude
    L = np.arctan2(y, x)
    L[L < 0] = L[L < 0] + 2. * np.pi


    # B - spherical latitude
    B =  np.pi / 2. - np.arccos(z / Q)
    return L, B, H


def geo2modip(theta, phi, date):
    '''
    Spherical theta (colatitude) , longitude (phi) [rad] to modip colatitude (theta_m) and local time (lt) [rad] for datetime    
    '''
    data = np.load('modip.npz')
    colats = data['colats']
    lons = data['lons']
    md = data['modip']

    FACT = np.pi /180.
    lut = RectSphereBivariateSpline(colats * FACT, lons  * FACT, md  * FACT)

    theta = np.asarray(theta)
    phi = np.asarray(phi)
    date = np.asarray(date)

    phi[phi < 0] = phi[phi < 0] + 2. * np.pi

    modip = lut.ev(theta, phi)

    theta_m = np.pi/2 - modip
    ut = sec_of_day(date)

    phi_sbs = np.deg2rad(180. - ut*15./3600)
    phi_sbs = np.asarray(phi_sbs)
    phi_sbs[phi_sbs < 0] = phi_sbs[phi_sbs < 0] + 2. * np.pi

    lt = phi - phi_sbs + np.pi
    lt[lt < 0] = lt[lt < 0] + 2. * np.pi
    lt[lt > 2. * np.pi] = lt[lt > 2. * np.pi] - 2. * np.pi

    return theta_m, lt



def geo2mag(theta, phi, date):
    '''
    Spherical theta (colatitude) , longitude (phi) [rad] to geomagnetic spherical colatitude (theta_m) and magnetic local time (lt) [rad] for datetime    
    '''
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    date = np.asarray(date)

    ut = sec_of_day(date)

    phi_sbs_m = np.pi - ut * np.deg2rad(15.) / 3600. - POLE_PHI 
    phi_sbs_m = np.asarray(phi_sbs_m)
    phi_sbs_m[phi_sbs_m < 0] = phi_sbs_m[phi_sbs_m < 0] + 2. * np.pi

    r = np.vstack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))

    r_mag = GEOGRAPHIC_TRANSFORM @ r
    theta_m = np.arccos(r_mag[2])
    phi_m = np.arctan2(r_mag[1], r_mag[0])

    phi_m[phi_m < 0.] = phi_m[phi_m < 0.] + 2. * np.pi 


    mlt = phi_m - phi_sbs_m + np.pi   
    mlt[mlt < 0.] = mlt[mlt < 0.] + 2. * np.pi 
    mlt[mlt > 2. * np.pi] = mlt[mlt > 2. * np.pi] - 2. * np.pi

    return theta_m, mlt


def geo2lt(theta, phi, date):
    '''
    Spherical theta (colatitude) , longitude (phi) [rad] to spherical colatitude (theta) and local time (lt) [rad] for datetime    
    '''
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    date = np.asarray(date)

    ut = sec_of_day(date)

    phi_sbs = np.deg2rad(180. - ut*15./3600)
    phi_sbs = np.asarray(phi_sbs)
    phi_sbs[phi_sbs < 0] = phi_sbs[phi_sbs < 0] + 2. * np.pi

    lt = phi - phi_sbs + np.pi
    lt[lt < 0] = lt[lt < 0] + 2. * np.pi
    lt[lt > 2. * np.pi] = lt[lt > 2. * np.pi] - 2. * np.pi

    return theta, lt
