import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import h5py

# imports from project
from mosgim2.data.loader import LoaderTxt
from mosgim2.data.tec_prepare import(process_data, combine_data,
                                     calc_coordinates,
                                     get_data,
                                     sites)

import config
from mosgim2.mosgim.mosgim import solve_all as solve1
from mosgim2.mosgim.mosgim2 import solve_all as solve2
from mosgim2.consts.phys_consts import secs_in_day, POLE_THETA, POLE_PHI
from mosgim2.data.writer import writer

if __name__ == '__main__':

    res_path = config.res_path
    data_path = config.data_path
    coords = config.coords
    IPPh_layer1 = config.IPPh_layer1
    IPPh_layer2 = config.IPPh_layer2
    nbig_layer1 = config.nbig_layer1
    mbig_layer1 = config.mbig_layer1  
    nbig_layer2 = config.nbig_layer2
    mbig_layer2 = config.mbig_layer2
    tint = config.tint 
    sigma0 = config.sigma0
    sigma_v = config.sigma_v
    linear = config.linear
    lcp = config.lcp 
    nlayers = config.nlayers
    nworkers = config.nworkers
    maxgap = config.maxgap
    maxjump = config.maxjump 
    el_cutoff = config.el_cutoff
    derivative = config.derivative 
    short = config.short 
    sparse = config.sparse

    n_coefs_layer1 = (nbig_layer1 + 1)**2 - (nbig_layer1 - mbig_layer1) * (nbig_layer1 - mbig_layer1 + 1)
    n_coefs_layer2 = (nbig_layer2 + 1)**2 - (nbig_layer2 - mbig_layer2) * (nbig_layer2 - mbig_layer2 + 1)
    n_coefs = n_coefs_layer1 + n_coefs_layer2

    st = time.time()

    loader = LoaderTxt(root_dir=data_path, IPPh1 = IPPh_layer1, IPPh2 = IPPh_layer2)
    data_generator = loader.generate_data(sites=sites)

    data = process_data(data_generator, maxgap = maxgap, maxjump = maxjump, el_cutoff = el_cutoff,
                        derivative = derivative, short = short, sparse = sparse)
    print(sorted(set(sites) - set(loader.not_found_sites)))
    data_combined = combine_data(data)
    result = calc_coordinates(data_combined, coords)
    data, time0 = get_data(result)

    print(f'Preprocessing done, took {time.time() - st}')

    st = time.time()

    ndays = np.ceil((np.max(data['time']) - np.min(data['time'])) / secs_in_day).astype('int') # number of days in input file

    nT_add = 1 if linear else 0

    if not (ndays == 1 or ndays == 3):
        print('procedure only works with 1 or 3 consecutive days data')
        exit(1)

    if nlayers == 2:
        res, disp_scale, Ninv = solve2(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, IPPh_layer1, IPPh_layer2, tint, sigma0, sigma_v, data, gigs=2, lcp=lcp, nworkers=nworkers, linear=linear)
    if nlayers == 1:
        res, disp_scale, Ninv = solve1(nbig_layer1, mbig_layer1, IPPh_layer1, tint, sigma0, sigma_v, data, gigs=2, lcp=lcp, nworkers=nworkers, linear=linear)

    if ndays == 3:
        res = res[n_coefs * (tint):n_coefs * (2 * tint + nT_add)] # select central day from 3day interval
        time0 = time0 + timedelta(days=1) 

    print(f'Computation done, took {time.time() - st}')

    res_file = Path(res_path, time0.strftime("%Y-%m-%d")+'.hdf5')

    if nlayers == 1:
        if coords == 'mag':
            writer(filename=res_file, res=res, time0=time0, nmaps=tint + nT_add, linear=linear, coord=coords, 
                   nlayers=nlayers, layer1_dims=[nbig_layer1, mbig_layer1], layer1_height=IPPh_layer1, 
                   sites=sorted(set(sites) - set(loader.not_found_sites)), pole_colat = POLE_THETA, pole_long = POLE_PHI)
        else:
            writer(filename=res_file, res=res, time0=time0, nmaps=tint + nT_add, linear=linear, coord=coords, 
                   nlayers=nlayers, layer1_dims=[nbig_layer1, mbig_layer1], layer1_height=IPPh_layer1, 
                   sites=sorted(set(sites) - set(loader.not_found_sites)))
  
    if nlayers == 2:
        if coords == 'mag':        
            writer(filename=res_file, res=res, time0=time0, nmaps=tint + nT_add, linear=linear, coord=coords, 
                   nlayers=nlayers, layer1_dims=[nbig_layer1, mbig_layer1], layer1_height=IPPh_layer1, layer2_dims = [nbig_layer2, mbig_layer2], layer2_height = IPPh_layer2, 
                   sites=sorted(set(sites) - set(loader.not_found_sites)), pole_colat = POLE_THETA, pole_long = POLE_PHI)  
        else:
            writer(filename=res_file, res=res, time0=time0, nmaps=tint + nT_add, linear=linear, coord=coords, 
                   nlayers=nlayers, layer1_dims=[nbig_layer1, mbig_layer1], layer1_height=IPPh_layer1, layer2_dims = [nbig_layer2, mbig_layer2], layer2_height = IPPh_layer2, 
                   sites=sorted(set(sites) - set(loader.not_found_sites)))  

