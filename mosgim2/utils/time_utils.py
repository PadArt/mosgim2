from datetime import datetime
import numpy as np


def sec_of_day(time):
    day_start = time.replace(hour=0, minute=0, second=0, microsecond=0)
    return (time - day_start).total_seconds()
sec_of_day = np.vectorize(sec_of_day)


def sec_of_interval(time, time0):
    return (time - time0).total_seconds()
sec_of_interval = np.vectorize(sec_of_interval, excluded='time0')

