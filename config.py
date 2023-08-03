from numpy import deg2rad as rad

# MosGIM configuration
#[PATHS]
data_path = <INPUT DATA DIR>
res_path = <RES DIRECTORY>

#[GENERAL]
nworkers = 1 # number of cpu cores you are able to use


#[PREPARE]
coords = 'mag'  # type of coordinates to use, can be [mag, geo, modip]
nlayers = 2 # number of layers in model 1 or 2, nevertheless provide both shell heights 
IPPh_layer1 = 300000. # schell height of first layer [m]
IPPh_layer2 = 750000. # schell height of second layer [m]
el_cutoff = rad(10.) # elevation cutoff for GNSS satellites [rad]
maxgap=35. # maximum gap in data to start new continous TEC arch [sec]
maxjump=1. # maximum TEC jump to start new continous TEC arch [TECu]
derivative = False # derivative (True) or relative (False) variant of difference approach for DCBs mitigation  
short = 3600 # minimum length of TEC continous arch taken into processing [sec]
sparse = 600 # sparse TEC data timestep [sec]

#[SOLVER]
nbig_layer1 = 15  # max order of spherical harmonic expansion in first layer
mbig_layer1 = 15  # max degree of spherical harmonic expansion in first layer (0 <= mbig <= nbig)
nbig_layer2 = 10  # max order of spherical harmonic expansion in second layer
mbig_layer2 = 10  # max degree of spherical harmonic expansion in second layer (0 <= mbig <= nbig)
tint = 24 # number of time intervals per day 
sigma0 = 0.1  # TECu - measurement noise at zenith 
sigma_v = 0.015  # TECu - allowed variability for each coef between two consecutive maps (0.03 TECu by Shaer for dt=2h and 149 coeffs)
linear = True # assumes piecewise linear interpolation between time nodes, if false piecewise constant interpolation
lcp = True # impose positivity constrains for TEC in each layer by solving LCP 

#[TEST CORRECTNESS]

if coords not in ['mag', 'geo', 'modip']:
    print('wrong coordinate system, setting to default mag ')
    coords = 'mag'

if nlayers not in [1, 2]:
    print('wrong number of layers, setting to default 2 ')
    nlayers = 2

if (nbig_layer1 < mbig_layer1): 
    print('wrong degree and order of SH for first layer relation, setting to default N=M ')
    mbig_layer1 = nbig_layer1 

if (nbig_layer2 < mbig_layer2): 
    print('wrong degree and order of SH for first layer relation, setting to default N=M ')
    mbig_layer2 = nbig_layer2 


