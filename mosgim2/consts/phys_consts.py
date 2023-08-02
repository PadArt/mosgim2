from numpy import pi, radians

# physicsal constants


RE = 6371000. # Earth radius in [m]
secs_in_day = 86400 # number of seconds in day


# North magnetic pole coordinates, for 2017 in [rad]
# Taken from here: http://wdc.kugi.kyoto-u.ac.jp/poles/polesexp.html
POLE_THETA = pi/2 - radians(80.5)
POLE_PHI = radians(-72.6)

