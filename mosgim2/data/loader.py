import os
import time
import numpy as np
import concurrent.futures
from datetime import datetime
from warnings import warn
from collections import defaultdict
from pathlib import Path


from mosgim2.coords.coords import xyz2spher
from mosgim2.coords.geom import ipp, elevation



tec_suite_FIELDS = ['datetime', 'sat_x', 'sat_y', 'sat_z', 'tec']
tec_suite_DTYPE = (object, float, float, float, float)

class Loader():
    
    def __init__(self, IPPh1, IPPh2):
        self.FIELDS = ['datetime', 'el', 'ipp_lat1', 'ipp_lon1', 'ipp_lat2', 'ipp_lon2', 'tec']
        self.DTYPE = (object, float, float, float, float, float, float)
        self.not_found_sites = []
        self.IPPh1 = IPPh1
        self.IPPh2 = IPPh2


class LoaderTxt(Loader):
    
    def __init__(self, root_dir, IPPh1, IPPh2):
        super().__init__(IPPh1, IPPh2)
        self.dformat = "%Y-%m-%dT%H:%M:%S"
        self.root_dir = root_dir


    def get_files(self, rootdir):
        """
        Root directroy must contain folders with site name 
        Inside subfolders are *.dat files for every satellite
        """
        result = defaultdict(list)
        for subdir, _, files in os.walk(rootdir):
            for filename in files:
                filepath = Path(subdir) / filename
                if str(filepath).endswith(".dat"):
                    site = filename[:4]
                    if site != subdir[-4:]:
                        raise ValueError(f'{site} in {subdir}. wrong site name')
                    result[site].append(filepath)
                else:
                    warn(f'{filepath} in {subdir} is not data file')
        for site in result:
            result[site].sort()
        return result


    def load_data(self, filepath):

        convert = lambda x: datetime.strptime(x.decode("utf-8"), self.dformat)
        with open(filepath, 'r') as fp:
            for l_no, line in enumerate(fp):
                if '(X, Y, Z)' in line:
                    obs_x, obs_y, obs_z  = map(float, line.split(":")[1].strip().split(','))
                    break

        data = np.genfromtxt(filepath, 
                             comments='#', 
                             names=tec_suite_FIELDS, 
                             dtype=tec_suite_DTYPE,
                             converters={"datetime": convert},  
                             )

        ip1 = ipp(obs_x, obs_y, obs_z, data['sat_x'], data['sat_y'], data['sat_z'], self.IPPh1)
        ip2 = ipp(obs_x, obs_y, obs_z, data['sat_x'], data['sat_y'], data['sat_z'], self.IPPh2)


        el = elevation(obs_x, obs_y, obs_z, data['sat_x'], data['sat_y'], data['sat_z'])


        arr = np.empty((len(data['tec']),), 
                       list(zip(self.FIELDS,self.DTYPE)))

        arr['datetime'] = data['datetime']
        arr['el'] = el

        arr['ipp_lat1'] = ip1[1]
        arr['ipp_lon1'] = ip1[0]

        arr['ipp_lat2'] = ip2[1]
        arr['ipp_lon2'] = ip2[0]

        arr['tec'] = data['tec']


        return arr, filepath
    
    def generate_data(self, sites=[]):
        files = self.get_files(self.root_dir)
        print(f'Collected {len(files)} sites')
        self.not_found_sites = sites[:]
        for site, site_files in files.items():
            if sites and not site in sites:
                continue
            self.not_found_sites.remove(site)
            count = 0
            st = time.time()
            for sat_file in site_files:
                try:
                    data, _ = self.load_data(sat_file)
                    count += 1
                    yield data, sat_file
                except Exception as e:
                    print(f'{sat_file} not processed. Reason: {e}')
            print(f'{site} contribute {count} files, takes {time.time() - st}')
            
