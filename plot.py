import numpy as np
from pathlib import Path
from mosgim2.coords.coords import geo2mag, geo2lt, geo2modip
from mosgim2.plot.plot import plot1l, plot2l
from mosgim2.plot.frames import makeframes

import h5py



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot GIMs ')
    parser.add_argument('--in_file',
                        type=Path,
                        help='Path to data, map creation')
    parser.add_argument('--out_file',
                        type=Path,
                        help='Path to video', default='animation.gif')
    args = parser.parse_args()


    data = h5py.File(args.in_file, 'r')

    nmaps = data.attrs['nmaps']
    nlayers = data.attrs['nlayers']
    coord =  data.attrs['coord']
    nbig_layer1, mbig_layer1 = data.attrs['layer1_dims'] 
    IPPh_layer1 = data.attrs['layer1_height']
    res_layer1 = data['layer1_SHcoefs'] 
    if nlayers == 2:
        nbig_layer2, mbig_layer2 = data.attrs['layer2_dims'] 
        IPPh_layer2 = data.attrs['layer2_height']
        res_layer2 = data['layer2_SHcoefs']

    ts = data['timestamps']

    # prepare net to estimate TEC on it
    colat = np.arange(2.5, 180, 2.5)
    lon = np.arange(-180, 180, 5.)

    frames1 = makeframes(lon, colat, coord, nbig_layer1, mbig_layer1, res_layer1, ts)    
    if nlayers == 2:
        frames2 = makeframes(lon, colat, coord, nbig_layer2, mbig_layer2, res_layer2, ts)    
        plot2l(str(args.out_file), colat, lon, ts, frames1, frames2)
    else:
        plot1l(str(args.out_file), colat, lon, ts, frames1)    
    


