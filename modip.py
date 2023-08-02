import numpy as np
from scipy.interpolate import RectSphereBivariateSpline
import pyIGRF.calculate as calculate


def inclination(lat, lon, alt=300., year=2017.):
    """
    :return
         I is inclination (+ve down)
    """
    if lon < 0:
        lon = lon + 360.      
    FACT = 180./np.pi
    RE = 6371
    x, y, z, f = calculate.igrf12syn(year, 2, RE + alt, lat, lon)
    h = np.sqrt(x * x + y * y)
    i = FACT * np.arctan2(z, h)
    return i


def modip(lat, lon, alt=300, year=2017):
 
    I = inclination(lat=lat, lon=lon, alt=alt, year=year)
    modip = np.rad2deg(np.arctan2(np.deg2rad(I), np.sqrt(np.cos(np.deg2rad(lat)))))
    return modip


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create modip coordinates grid')
    parser.add_argument('--year',
                        type=float,
                        help='desired year')
    parser.add_argument('--alt',
                        type=float,
                        help='reference altitude in km')
    args = parser.parse_args()

    colats = np.arange(1, 180., 1)
    lons = np.arange(0, 360.,2)

    year = args.year
    alt = args.alt

    filename = 'modip.npz'
    a = np.zeros((len(colats),len(lons)))
    i = 0
    for colat in colats:
        j = 0
        for lon in lons:
            a[i,j] = modip(90 - colat, lon, alt, year) 
            j=j+1
        i=i+1
    np.savez(filename, year=year, colats=colats, lons=lons, modip = a)
    print(filename, year, alt)





