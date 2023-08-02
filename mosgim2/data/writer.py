import h5py
from datetime import datetime, timedelta, timezone
import numpy as np
from mosgim2.consts.phys_consts import secs_in_day

def writer(filename, res, time0, nmaps, linear, coord, nlayers, layer1_dims, layer1_height,  **kwargs):    
    hf = h5py.File(filename, 'w')
    hf.attrs['nmaps'] = nmaps   #tint + nT_add
    hf.attrs['linear'] = linear
    hf.attrs['coord'] = coord
    hf.attrs['nlayers'] = nlayers
    hf.attrs['layer1_dims'] = layer1_dims #[nbig_layer1, mbig_layer1]
    hf.attrs['layer1_height'] = layer1_height    
    nbig_layer1, mbig_layer1 = layer1_dims
    n_coefs_layer1 = (nbig_layer1 + 1)**2 - (nbig_layer1 - mbig_layer1) * (nbig_layer1 - mbig_layer1 + 1)
    
    if nlayers == 2:
        hf.attrs['layer2_dims'] = kwargs.get('layer2_dims') # [nbig_layer2, mbig_layer2]
        hf.attrs['layer2_height'] = kwargs.get('layer2_height')
        nbig_layer2, mbig_layer2 = kwargs.get('layer2_dims')
        n_coefs_layer2 = (nbig_layer2 + 1)**2 - (nbig_layer2 - mbig_layer2) * (nbig_layer2 - mbig_layer2 + 1)
    elif nlayers == 1:
        n_coefs_layer2 = 0
 
    res_frames = np.array_split(res, nmaps)
    ts0 = time0.replace(tzinfo=timezone.utc).timestamp()        

    n_coefs = n_coefs_layer1 + n_coefs_layer2

    if nlayers == 1:
        hf.create_dataset('layer1_SHcoefs', data=res_frames)
    if nlayers == 2:
        l1 = [f[0:n_coefs_layer1] for f in res_frames]
        l2 = [f[n_coefs_layer1:n_coefs] for f in res_frames]
        hf.create_dataset('layer1_SHcoefs', data=l1)
        hf.create_dataset('layer2_SHcoefs', data=l2)

    tint = nmaps - 1 if linear else nmaps     
    times = np.array([ts0 + i * secs_in_day/ tint for i in range(nmaps)])
    hf.create_dataset('timestamps', data=times)

    hf.attrs['sites'] = kwargs.get('sites')
    if coord == 'mag':
        hf.attrs['pole_colat'] = kwargs.get('pole_colat')
        hf.attrs['pole_long'] = kwargs.get('pole_long')
    hf.close()

