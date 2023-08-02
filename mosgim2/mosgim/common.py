import numpy as np
from scipy.special import sph_harm

def calc_coefs(M, N, theta, phi, sf):
    """
    :param M: meshgrid of harmonics degrees
    :param N: meshgrid of harmonics orders
    :param theta: LT of IPP in rad
    :param phi: co latitude of IPP in rad
    :param sf: slant factor
    """
    n_coefs = len(M)
    a = np.zeros(n_coefs)
    Ymn = sph_harm(np.abs(M), N, theta, phi)  # complex harmonics on meshgrid
    #  introducing real basis according to scipy normalization
    a[M < 0] = Ymn[M < 0].imag * np.sqrt(2) * (-1.) ** M[M < 0]
    a[M > 0] = Ymn[M > 0].real * np.sqrt(2) * (-1.) ** M[M > 0]
    a[M == 0] = Ymn[M == 0].real
    del Ymn
    return a*sf
vcoefs = np.vectorize(calc_coefs, excluded=['M','N'], otypes=[np.ndarray])


def get_chunk_indexes(size, nchunks):
    if nchunks > 1:
        step = int(size / nchunks)
        ichunks = [(i-step, i) for i in range(step, size, step)]
        if (size - ichunks[-1][1]) / size > 0.1:
            ichunks += [(ichunks[-1][1], size)]
        else:
            ichunks[-1] = (ichunks[-1][0], size)
        return ichunks
    elif nchunks <= 1:
        return [(0, size)]


def split_data(data, nchunks=1):

    arr = np.empty((len(data['time']),), list(zip(data.keys(), [float for _ in range(len(data.keys()))]) ))

    for key in data.keys():
        arr[key] = data[key]

    size = len(data['time'])
    combs = []
    ichunks = get_chunk_indexes(size, nchunks)
    for i, (start, fin) in enumerate(ichunks):
        comb = arr[start:fin]
        combs.append(comb)
    return combs

