import numpy as np
from numpy import deg2rad as rad
from datetime import datetime
from scipy.signal import savgol_filter
from collections import defaultdict

from mosgim2.coords.coords import geo2mag, geo2modip, geo2lt
from mosgim2.utils.time_utils import sec_of_day, sec_of_interval


sites = ['019b', '7odm', 'ab02', 'ab06', 'ab07', 'ab09', 'ab11', 'ymer', 'yssk', 'zamb'
         'ab12', 'ab13', 'ab15', 'ab17', 'ab21', 'ab27', 'ab33', 'ab35', 'ab37', 'ab41',
         'ab44', 'ab45', 'ab48', 'ab49', 'ac03', 'ac21', 'ac61', 'ac65', 'acor', 'acsb',
         'acso', 'adis', 'ahid', 'aira', 'ajac', 'albh', 'alg3', 'alic', 'alme', 'alon',
         'alrt', 'alth', 'amc2', 'ankr', 'antc', 'aqui', 'areq', 'artu', 'aruc', 'asky',
         'aspa', 'auck', 'badg', 'baie', 'bake', 'bald', 'bamo', 'bako', 'bara', 'barh',
         'bcyi', 'bell', 'benn', 'berp', 'bilb', 'bjco', 'bjfs', 'bjnm', 'bla1', 'blyn',
         'bluf', 'bogi', 'bogt', 'braz', 'brip', 'brst', 'brft', 'brux', 'bshm', 'bucu',
         'budp', 'bums', 'buri', 'burn', 'bzrg', 'cand', 'cant', 'capf', 'cas1', 'casc',
         'ccj2', 'cedu', 'chan', 'chiz', 'chpi', 'chti', 'chum', 'chur', 'cihl', 'cjtr',
         'ckis', 'clrk', 'cmbl', 'cn00', 'cn04', 'cn09', 'cn13', 'cn20', 'cn22', 'cn23',
         'cn40', 'cnmr', 'coco', 'con2', 'cord', 'coyq', 'crao', 'cusv', 'daej', 'dakr',
         'dane', 'darw', 'dav1', 'ddsn', 'devi', 'dgar', 'dgjg', 'dond', 'drao', 'dubo',
         'ecsd', 'ela2', 'eur2', 'faa1', 'falk', 'fall', 'ffmj', 'flin', 'flrs', 'func',
         'g101', 'g107', 'g117', 'g124', 'g201', 'g202', 'ganp', 'gisb', 'glps', 'gls1',
         'gls2', 'gls3', 'glsv', 'gmma', 'gode', 'guat', 'guax', 'harb', 'hces', 'hdil',
         'helg', 'hlfx', 'hmbg', 'hnlc', 'hob2', 'hofn', 'holm', 'howe', 'hrao', 'hsmn',
         'hueg', 'hyde', 'ibiz', 'ifr1', 'iisc', 'ilsg', 'inmn', 'invk', 'iqal', 'iqqe',
         'irkj', 'isba', 'isco', 'ista', 'joen', 'karr', 'kely', 'khar', 'khlr', 'kir0',
         'kiri', 'kiru', 'kour', 'ksnb', 'ksu1', 'kuaq', 'kulu', 'kuuj', 'kvtx', 'lamp',
         'laut', 'lcsb', 'lhaz', 'lkwy', 'lovj', 'lply', 'lthw', 'lyns', 'mac1', 'mag0',
         'maju', 'mal2', 'mall', 'mana', 'mar6', 'marg', 'mas1', 'mat1', 'maw1', 'mcar',
         'mcm4', 'mdvj', 'mgue', 'mizu', 'mkea', 'mobs', 'moiu', 'morp', 'nain', 'naur',
         'newl', 'nium', 'nnor', 'noa1', 'not1', 'novm', 'nril', 'nya1', 'ohi2', 'ons1',
         'ous2', 'p008', 'p014', 'p038', 'p050', 'p776', 'p778', 'p803', 'palm', 'parc',
         'park', 'pbri', 'pece', 'penc', 'pets', 'pimo', 'pirt', 'pngm', 'pol2', 'pove',
         'qaar', 'qaq1', 'qiki', 'recf', 'reso', 'reyk', 'riop', 'rmbo', 'salu', 'sask',
         'savo', 'sch2', 'scor', 'sg27', 'sgoc', 'shao', 'soda', 'stas', 'stew', 'sthl',
         'stj2', 'stk2', 'sumk', 'suth', 'syog', 'tash', 'tcms', 'tehn', 'tetn', 'tixg',
         'tomo', 'tor2', 'tow2', 'trds', 'tro1', 'tsk2', 'tuc2', 'tuva', 'udec', 'ufpr',
         'ulab', 'unbj', 'unpm', 'urum', 'usmx', 'vaas', 'vacs', 'vars', 'vis0', 'vlns',
         'whit', 'whng', 'whtm', 'will', 'wind', 'wway', 'xmis', 'yakt', 'yell', 'ykro']




def getContInt(data,  maxgap=35., maxjump=3., el_cutoff=rad(10.)):
    time = sec_of_day(data['datetime'])  
    r = np.array(range(len(time)))
    idx = np.isfinite(data['tec']) & np.isfinite(data['ipp_lon1']) & np.isfinite(data['ipp_lat1']) & np.isfinite(data['ipp_lon2']) & np.isfinite(data['ipp_lat2']) & np.isfinite(data['el']) & (data['el'] > el_cutoff) & (data['tec'] != 0.) 
    r = r[idx]
    intervals = []
    if len(r) == 0:
        return intervals
    beginning = r[0]
    last = r[0]
    last_time = time[last]
    for i in r[1:]:
        if abs(time[i] - last_time) > maxgap or abs(data['tec'][i] - data['tec'][last]) > maxjump:
            intervals.append((beginning, last))
            beginning = i
        last = i
        last_time = time[last]
        if i == r[-1]:
            intervals.append((beginning, last))
    return idx, intervals


def process_data(data_generator, maxgap=35.,
                 maxjump=3., el_cutoff=rad(10.),
                 derivative=False, short = 3600, sparse = 600):
    all_data = defaultdict(list)
    count = 0
    for data, data_id in data_generator:
        if data.shape==():
            print(f'No data for {data_id}')
            continue
        times = data['datetime'][:]
        data_days = [datetime(d.year, d.month, d.day) for d in times]
        if len(set(data_days)) != 1:
            msg = f'{data_id} is not processed: multiple days presented '
            msg += f'{set(data_days)}. Skip.'
            print(msg)
            continue
        try:
            prepared = process_intervals(data, maxgap=maxgap,
                                         maxjump=maxjump, el_cutoff=el_cutoff,
                                         derivative=derivative, short = short, sparse = sparse)
            count += len(prepared['dtec'])
            for k in prepared:
                all_data[k].extend(prepared[k])
        except Exception as e:
            print(f'{data_id} not processed. Reason: {e}')
    return all_data


def process_intervals(data, maxgap=35., maxjump=3., el_cutoff=rad(10.), derivative=False,
                      short = 3600, sparse = 600):
    result = defaultdict(list)
    tt = sec_of_day(data['datetime'])
    idx, intervals = getContInt(data,  maxgap=maxgap, maxjump=maxjump, el_cutoff=el_cutoff)

    for start, fin in intervals:

        if (tt[fin] - tt[start]) < short:    # disgard all the arcs shorter than 1 hour
            #print('too short interval')
            continue
        ind_sparse = (tt[start:fin] % sparse == 0)
        data_sample = data[start:fin]
#       TODO need some reasonable filtering here to remove small scale fluctuations
        data_sample = data_sample[ind_sparse]

        if derivative == True:
            dtec = data_sample['tec'][1:] - data_sample['tec'][0:-1]
            data_out = data_sample[1:]
            data_ref = data_sample[0:-1]

        if derivative == False:
            idx_min = np.argmin(data_sample['tec'])
            data0 = data_sample[idx_min]
            data_out = np.delete(data_sample, idx_min)
            dtec = data_out['tec'][:] - data0['tec']
            data_ref = np.zeros_like(data_out)
            data_ref[:] = data0
        result['dtec'].append(dtec)
        result['out'].append(data_out)
        result['ref'].append(data_ref)
    return result

def combine_data(all_data):

    tec_data = np.concatenate(tuple(o for o in all_data['dtec']))
    out_data = np.concatenate(tuple(o for o in all_data['out']))
    ref_data = np.concatenate(tuple(r for r in all_data['ref']))
    fields = ['dtec', 'time', 'lon1', 'lat1', 'lon2', 'lat2',  'el', 'rtime', 'rlon1', 'rlat1', 'rlon2', 'rlat2', 'rel',
              'colat1', 'mlt1', 'colat2', 'mlt2', 
              'rcolat1', 'rmlt1', 'rcolat2', 'rmlt2']

    comb = dict()
    comb['dtec'] = tec_data
    _out_data = out_data
    comb['time'] = _out_data['datetime']
    comb['lon1'] = _out_data['ipp_lon1']
    comb['lat1'] = _out_data['ipp_lat1']
    comb['lon2'] = _out_data['ipp_lon2']
    comb['lat2'] = _out_data['ipp_lat2']
    comb['el'] = _out_data['el']
    _ref_data = ref_data
    comb['rtime'] = _ref_data['datetime']
    comb['rlon1'] = _ref_data['ipp_lon1']
    comb['rlat1'] = _ref_data['ipp_lat1']
    comb['rlon2'] = _ref_data['ipp_lon2']
    comb['rlat2'] = _ref_data['ipp_lat2']
    comb['rel'] = _ref_data['el']
    for f in fields:
        if f in comb:
            continue
        comb[f] = np.zeros(comb['dtec'].shape)
        
    return comb


def calc_cur(comb, c2c):
    colat1, mlt1 = c2c(np.pi/2 - comb['lat1'], comb['lon1'], comb['time'])
    colat2, mlt2 = c2c(np.pi/2 - comb['lat2'], comb['lon2'], comb['time'])
    return colat1, mlt1, colat2, mlt2 

def calc_ref(comb, c2c):
    rcolat1, rmlt1 = c2c(np.pi/2 - comb['rlat1'], comb['rlon1'], comb['rtime'])
    rcolat2, rmlt2 = c2c(np.pi/2 - comb['rlat2'], comb['rlon2'], comb['rtime'])
    return  rcolat1, rmlt1, rcolat2, rmlt2 


def calc_coordinates(comb, coord_type):
    
    if coord_type == 'mag':
        comb['colat1'], comb['mlt1'], comb['colat2'], comb['mlt2'] = calc_cur(comb, geo2mag)
        comb['rcolat1'], comb['rmlt1'], comb['rcolat2'], comb['rmlt2'] = calc_ref(comb, geo2mag)
    elif coord_type == 'geo':
        comb['colat1'], comb['mlt1'], comb['colat2'], comb['mlt2'] = calc_cur(comb, geo2lt)
        comb['rcolat1'], comb['rmlt1'], comb['rcolat2'], comb['rmlt2'] = calc_ref(comb, geo2lt)
    elif coord_type == 'modip':
        comb['colat1'], comb['mlt1'], comb['colat2'], comb['mlt2'] = calc_cur(comb, geo2modip)
        comb['rcolat1'], comb['rmlt1'], comb['rcolat2'], comb['rmlt2'] = calc_ref(comb, geo2modip)

    return comb


def get_data(comb):
    t = np.min(comb['time'])
    time0 = t.replace(hour=0, minute=0, second=0, microsecond=0)
    data = dict(time = sec_of_interval(comb['time'], time0),
                mlt1 = comb['mlt1'],
                mcolat1 = comb['colat1'],
                mlt2 = comb['mlt2'],
                mcolat2 = comb['colat2'],
                el = comb['el'],
                time_ref = sec_of_interval(comb['rtime'], time0),
                mlt1_ref = comb['rmlt1'],
                mcolat1_ref = comb['rcolat1'],
                mlt2_ref = comb['rmlt2'],
                mcolat2_ref = comb['rcolat2'],
                el_ref = comb['rel'],
                rhs = comb['dtec'])
    return data, time0
