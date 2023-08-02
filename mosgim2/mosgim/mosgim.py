import numpy as np
from lemkelcp.lemkelcp import lemkelcp
from scipy.special import sph_harm
from scipy.sparse import lil_matrix, csr_matrix
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from mosgim2.coords.geom import MF
from mosgim2.mosgim.common import split_data, vcoefs
from mosgim2.consts.phys_consts import secs_in_day


# configuration
GB_CHUNK = 30000 
#################################


 
def construct_normal_system(nbig, mbig, IPPh, nT, ndays, sigma0, input_data, linear):
    """
    :param nbig: maximum order of spherical harmonic
    :param mbig: maximum degree of spherical harmonic
    :param IPPh: height of shell [m]
    :param nT: number of time intervals
    :param ndays: number of days in analysis
    :param sigma0: standard deviation of TEC estimate at zenith

    : in input_data
    :param time': array of times of IPPs in secs
    :param theta: array of LTs of IPPs in rads
    :param phi: array of co latitudes of IPPs in rads
    :param el: array of elevation angles in rads
    :param time_ref: array of ref times of IPPs in sec
    :param theta_ref: array of ref longitudes (LTs) of IPPs in rads
    :param phi_ref: array of ref co latitudes of IPPs in rads
    :param el_ref: array of ref elevation angles in rads
    :param rhs: array of rhs (measurements TEC difference on current and ref rays)

    :param linear: bool defines const or linear
    """
    print('constructing normal system for series')

    time =      input_data['time']
    theta =     input_data['mlt1']
    phi =       input_data['mcolat1']
    el =        input_data['el']
    time_ref =  input_data['time_ref']
    theta_ref = input_data['mlt1_ref']
    phi_ref =   input_data['mcolat1_ref']
    el_ref =    input_data['el_ref']
    rhs =       input_data['rhs']

    tmc = time
    tmr = time_ref
    SF = MF(el, IPPh)
    SF_ref = MF(el_ref, IPPh)
 
    # Construct weight matrix for the observations
    len_rhs = len(rhs)
    P = lil_matrix((len_rhs, len_rhs))
    el_sin = np.sin(el)
    elr_sin = np.sin(el_ref)
    diagP = (1 / sigma0**2) * (el_sin ** 2) * (elr_sin ** 2) / (el_sin ** 2 + elr_sin ** 2) 
    P.setdiag(diagP)
    P = P.tocsr()
 
    # Construct matrix of the problem (A)
    n_ind = np.arange(0, nbig + 1, 1)
    m_ind = np.arange(-mbig, mbig + 1, 1)
    M, N = np.meshgrid(m_ind, n_ind)
    Y = sph_harm(np.abs(M), N, 0, 0)
    idx = np.isfinite(Y)
    M = M[idx]
    N = N[idx]
    n_coefs = len(M)
 
    tic = (tmc * nT / (ndays * secs_in_day)).astype('int16')
    tir = (tmr * nT / (ndays * secs_in_day)).astype('int16')

    ac = vcoefs(M=M, N=N, theta=theta, phi=phi, sf=SF)
    ar = vcoefs(M=M, N=N, theta=theta_ref, phi=phi_ref, sf=SF_ref)
    print('coefs done', n_coefs, nT, ndays, len_rhs)

    #prepare (A) in csr sparse format
    dims = 4 if linear else 2
    nT_add = 1 if linear else 0
    data = np.empty(len_rhs * n_coefs * dims)
    rowi = np.empty(len_rhs * n_coefs * dims)
    coli = np.empty(len_rhs * n_coefs * dims)
    
    if linear:
        hour_cc = (ndays * secs_in_day) * tic / nT    
        hour_cn = (ndays * secs_in_day) * (tic + 1) / nT    
        hour_rc = (ndays * secs_in_day) * tir / nT    
        hour_rn = (ndays * secs_in_day) * (tir + 1) / nT  
        dt = (ndays * secs_in_day) / nT 
        for i in range(0, len_rhs, 1): 
            st  = [i * n_coefs * dims + j * n_coefs for j in range(0, 4)]
            end = [i * n_coefs * dims + j * n_coefs for j in range(1, 5)]
            data[st[0]: end[0]] =   (  tmc[i] - hour_cc[i]) * ac[i] / dt
            data[st[1]: end[1]] =   ( -tmc[i] + hour_cn[i]) * ac[i] / dt
            data[st[2]: end[2]] = - (  tmr[i] - hour_rc[i]) * ar[i] / dt
            data[st[3]: end[3]] = - ( -tmr[i] + hour_rn[i]) * ar[i] / dt

            rowi[st[0]: end[-1]] = i 
            
            coli[st[0]: end[0]] = \
                np.arange((tic[i] + 0) * n_coefs, (tic[i] + 1) * n_coefs, 1).astype('int32')
            coli[st[1]: end[1]] = \
                np.arange((tic[i] + 1) * n_coefs, (tic[i] + 2) * n_coefs, 1).astype('int32')
            coli[st[2]: end[2]] = \
                np.arange((tir[i] + 0) * n_coefs, (tir[i] + 1) * n_coefs, 1).astype('int32')
            coli[st[3]: end[3]] = \
                np.arange((tir[i] + 1) * n_coefs, (tir[i] + 2) * n_coefs, 1).astype('int32')
    else:
        for i in range(0, len_rhs, 1): 
            st  = [i * n_coefs * dims + j * n_coefs for j in range(0, 2)]
            end = [i * n_coefs * dims + j * n_coefs for j in range(1, 3)]
            data[st[0]: end[0]] =  ac[i]
            data[st[1]: end[1]] = -ar[i]
            
            rowi[st[0]: end[-1]] = i 
            
            coli[st[0]: end[0]] = \
                np.arange(tic[i] * n_coefs, (tic[i] + 1) * n_coefs, 1).astype('int32')
            coli[st[1]: end[1]] = \
                np.arange(tir[i] * n_coefs, (tir[i] + 1) * n_coefs, 1).astype('int32')

    A = csr_matrix((data, (rowi, coli)), 
                   shape=(len_rhs, (nT + nT_add) * n_coefs))
    print('matrix (A) for subset done')
 
    # define normal system
    AP = A.transpose().dot(P)
    N = AP.dot(A).todense()    
    b = AP.dot(rhs)
    rhsT_P_rhs = np.dot(diagP, np.square(rhs))
    print('normal matrix (N) for subset done')

    return N, b, rhsT_P_rhs, len_rhs


def stack_constrain_solve_ns(nbig, mbig, IPPh, nT, ndays, sigma0, sigma_v, data_chunks, nworkers=3, linear=True):


    nT_add = 1 if linear else 0
    n_coefs = (nbig + 1)**2 - (nbig - mbig) * (nbig - mbig + 1)

    N = np.zeros((n_coefs * (nT + nT_add), n_coefs * (nT + nT_add)))
    b = np.zeros(n_coefs * (nT + nT_add))
    rhsT_P_rhs = 0.
    DOF = --n_coefs * (nT + nT_add) # degrees of freedom    

    # stacking normal matrix in paralell mode
    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        queue = []
        for chunk in data_chunks:
            params = (nbig, mbig, IPPh, nT, ndays, sigma0, chunk, linear, )
            query = executor.submit(construct_normal_system, *params)
            queue.append(query)
        for v in as_completed(queue):
            NN, bb, rPr, lrhs = v.result()
            N += NN
            b += bb
            rhsT_P_rhs += rPr
            DOF += lrhs
    print('normal matrix (N) stacked')


    # imposing frozen conditions on consecuitive maps coeffs
    for ii in range(0, nT - 1 + nT_add, 1): 
        for kk in range(0, n_coefs):
            N[ii*n_coefs + kk, ii*n_coefs + kk] += 1. / sigma_v**2
            N[(ii + 1) * n_coefs + kk, (ii+1) * n_coefs + kk] += 1. / sigma_v**2
            N[(ii + 1) * n_coefs + kk, ii * n_coefs + kk] += -1. / sigma_v**2
            N[ii * n_coefs + kk, (ii + 1) * n_coefs + kk] += -1. / sigma_v**2
            DOF += 1
    print('normal matrix (N) constraints added')


    # # solve normal system
    Ninv = np.linalg.inv(N)     
    res = np.dot(Ninv, b)  
    print('normal system solved')
    
    disp_scale = (rhsT_P_rhs - np.dot(res, b)) / DOF  

    print('aposteriory standard deviation of unit weight', np.sqrt(disp_scale))

    return res, disp_scale, Ninv


# non paralel version, keept for large memory problems
def stack_constrain_solve_ns_np(nbig, mbig, IPPh, nT, ndays, sigma0, sigma_v, data_chunks, linear=True):


    nT_add = 1 if linear else 0
    n_coefs = (nbig + 1)**2 - (nbig - mbig) * (nbig - mbig + 1)

    N = np.zeros((n_coefs * (nT + nT_add), n_coefs * (nT + nT_add)))
    b = np.zeros(n_coefs * (nT + nT_add))
    rhsT_P_rhs = 0.
    DOF = -n_coefs * (nT + nT_add) # degrees of freedom    


    # stacking normal matrix in consequitive mode 
    for chunk in data_chunks:
        params = (nbig, mbig, IPPh, nT, ndays, sigma0, chunk, linear, )
        NN, bb, rPr, lrhs = construct_normal_system(*params)
        N += NN
        b += bb
        rhsT_P_rhs += rPr
        DOF += lrhs
    print('normal matrix (N) stacked')


    # imposing frozen conditions on consecuitive maps coeffs
    for ii in range(0, nT - 1 + nT_add, 1): 
        for kk in range(0, n_coefs):
            N[ii*n_coefs + kk, ii*n_coefs + kk] += 1. / sigma_v**2
            N[(ii + 1) * n_coefs + kk, (ii+1) * n_coefs + kk] += 1. / sigma_v**2
            N[(ii + 1) * n_coefs + kk, ii * n_coefs + kk] += -1. / sigma_v**2
            N[ii * n_coefs + kk, (ii + 1) * n_coefs + kk] += -1. / sigma_v**2
            DOF += 1
    print('normal matrix (N) constraints added')


    # # solve normal system
    Ninv = np.linalg.inv(N)     
    res = np.dot(Ninv, b)  
    print('normal system solved')
    
    disp_scale = (rhsT_P_rhs - np.dot(res, b)) / DOF  

    print('aposteriory standard deviation of unit weight', np.sqrt(disp_scale))

    return res, disp_scale, Ninv


# constructing observation matrix to use in LCP problem
def constructG(nbig, mbig, nT, theta, phi, timeindex, linear):
    """
    :param nbig: maximum order of spherical harmonic
    :param mbig: maximum degree of spherical harmonic
    :param nT: number of time intervals
    :param theta: array of LTs of IPPs in rads
    :param phi: array of co latitudes of IPPs in rads
    :param timeindex: array of timeindex
    :param linear: boolean for linear or const interpolation

    """
    nT_add = 1 if linear else 0
 
    # Construct observation matrix (G)
    n_ind = np.arange(0, nbig + 1, 1)
    m_ind = np.arange(-mbig, mbig + 1, 1)
    M, N = np.meshgrid(m_ind, n_ind)
    Y = sph_harm(np.abs(M), N, 0, 0)
    idx = np.isfinite(Y)
    M = M[idx]
    N = N[idx]
    n_coefs = len(M)
    len_rhs = len(phi)

    a = vcoefs(M=M, N=N, theta=theta, phi=phi, sf=np.ones(len(phi)))

    print('coefs done', n_coefs)

    #prepare (G) in csr sparse format
    data = np.empty(len_rhs * n_coefs)
    rowi = np.empty(len_rhs * n_coefs)
    coli = np.empty(len_rhs * n_coefs)

    for i in range(0, len_rhs,1): 
        data[i * n_coefs: (i + 1) * n_coefs] = a[i]
        rowi[i * n_coefs: (i + 1) * n_coefs] = i * np.ones(n_coefs).astype('int32')
        coli[i * n_coefs: (i + 1) * n_coefs] = np.arange(timeindex[i] * n_coefs, (timeindex[i] + 1) * n_coefs, 1).astype('int32')

    
    G = csr_matrix((data, (rowi, coli)), shape=(len_rhs, (nT + nT_add) * n_coefs))
    print('matrix (G) done')

    return G


def LCPCorrection(res, Ninv, nbig, mbig, nT, ndays, linear):

    tint = int(nT / ndays)


    colat = np.arange(2.5, 180, 2.5)
    mlt = np.arange(0., 360., 5.)
    mlt_m, colat_m  = np.meshgrid(mlt, colat)

    nT_add = 1 if linear else 0

    if ndays==3:
        mlt_m = np.tile(mlt_m.flatten(), tint + nT_add)
        colat_m = np.tile(colat_m.flatten(), tint  + nT_add)
        time_m = np.array([int(_ / (len(colat) * len(mlt))) for _ in range(len(colat) * len(mlt) * tint, len(colat) * len(mlt) * (2 * tint + nT_add),1)])

    if ndays==1:
        mlt_m = np.tile(mlt_m.flatten(), nT + nT_add)
        colat_m = np.tile(colat_m.flatten(), nT + nT_add)
        time_m = np.array([int(_ / (len(colat) * len(mlt))) for _ in range(len(colat) * len(mlt) * (nT + nT_add))])

    G = constructG(nbig, mbig, nT, np.deg2rad(mlt_m), np.deg2rad(colat_m), time_m, linear)
    
    w = G.dot(res)
    idx = (w<0)
    print(idx)
 
    if np.any(idx):
                            
        Gnew = G[idx,:]
        wnew = w[idx]
   
        print('constructing M')

        NGT = Ninv * Gnew.transpose()
        M = Gnew.dot(NGT)

        print('solving LCP')

        sol = lemkelcp(M,wnew,1000000)
        
        try:        
            res = res + NGT.dot(sol[0])
            print ('lcp adjustment done')
        except:
            print('no lcp solution found')

    return res


def solve_all(nbig, mbig, IPPh, tint, sigma0, sigma_v, data, gigs=2, lcp=True, nworkers=1, linear=True):
    chunk_size = GB_CHUNK * gigs


    ndays = np.ceil((np.max(data['time']) - np.min(data['time'])) / secs_in_day).astype('int') # number of days in data
    nT = tint * ndays  # number of intervals for all time period      

    nchunks = np.int(len(data['rhs']) / chunk_size) # set chuncks size to fit in memory 
    nchunks = 1 if nchunks < 1 else nchunks

    print('start, nbig=%s, mbig=%s, nT=%s, ndays=%s, sigma0=%s, sigma_v=%s, number of observations=%s, number of chuncks=%s' % (nbig, mbig, nT, ndays, sigma0, sigma_v, len(data['rhs']), nchunks))


    # split data into chunks
    data_chunks = split_data(data, nchunks)

    if (nworkers > 1):
        res, disp_scale, Ninv = stack_constrain_solve_ns(nbig, mbig, IPPh, nT, ndays, sigma0, sigma_v, data_chunks, nworkers, linear)


    if (nworkers == 1):
        res, disp_scale, Ninv = stack_constrain_solve_ns_np(nbig, mbig, IPPh, nT, ndays, sigma0, sigma_v, data_chunks, linear)



    if lcp:
        res = LCPCorrection(res, Ninv, nbig, mbig, nT, ndays, linear) 


    return res, disp_scale, Ninv


