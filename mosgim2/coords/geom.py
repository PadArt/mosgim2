import numpy as np
from mosgim2.consts.phys_consts import RE
from mosgim2.coords.coords import xyz2spher


def MF(el, IPPh):
    """
    :param el: elevation angle in [rad]
    :param IPPh: height of IPP in [m]
    """
    return 1./np.sqrt(1 - (RE * np.cos(el) / (RE + IPPh)) ** 2)


def ipp(obs_x, obs_y, obs_z, sat_x, sat_y, sat_z, IPPh):
    """
    :param obs_x: observer x coord in ECEF [m]
    :param obs_y: observer y coord in ECEF [m]
    :param obs_z: observer z coord in ECEF [m]
    :param sat_x: satellite x coord in ECEF [m]
    :param sat_y: satellite y coord in ECEF [m]
    :param sat_z: satellite z coord in ECEF [m]
    :param IPPh: height of IPP in meters
    """
    A = (sat_x - obs_x) ** 2 + (sat_y - obs_y) ** 2 + (sat_z - obs_z) ** 2 
    obs2 = obs_x **2 + obs_y **2 + obs_z **2    
    B = 2 * (obs_x * sat_x + obs_y * sat_y + obs_z * sat_z - obs2)       
    C = obs2 - (RE + IPPh)**2       
    D = B**2 - 4*A*C

    t1 = (- B + np.sqrt(D)) / (2. * A)
    t2 = (- B - np.sqrt(D)) / (2. * A)

    idx1 = np.where((t1<=1)&(t1>=0))
    idx2 = np.where((t2<=1)&(t2>=0))

    t = np.empty(len(sat_x))

    t[idx1] = t1[idx1]
    t[idx2] = t2[idx2]


    ipp_x = obs_x * (1 - t) + sat_x * t
    ipp_y = obs_y * (1 - t) + sat_y * t
    ipp_z = obs_z * (1 - t) + sat_z * t
    
    ipp_l, ipp_b, ipp_h = xyz2spher(ipp_x,ipp_y,ipp_z)

    return ipp_l, ipp_b, ipp_h


def elevation(obs_x, obs_y, obs_z, sat_x, sat_y, sat_z):
    '''
    :param obs_x: observer x coord in ECEF [m]
    :param obs_y: observer y coord in ECEF [m]
    :param obs_z: observer z coord in ECEF [m]
    :param sat_x: satellite x coord in ECEF [m]
    :param sat_y: satellite y coord in ECEF [m]
    :param sat_z: satellite z coord in ECEF [m]
    '''   
    A = (sat_x - obs_x) ** 2 + (sat_y - obs_y) ** 2 + (sat_z - obs_z) ** 2 
    C = obs_x **2 + obs_y **2 + obs_z **2    
    B = obs_x * sat_x + obs_y * sat_y + obs_z * sat_z - C

    el = np.arcsin(B / (np.sqrt (A * C)))
    return el



