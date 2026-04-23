import re
import numpy as N
from astropy.time import Time
def file_name_to_datetime(fname):
    """
    Read date from file name. assume integrated over dt seconds starting at file print time
    """
    res=re.search(".*(....)_(..)_(..)_(..)_(..)_(..)_(...)_(......).mp4.*",fname)
    # '2010-01-01T00:00:00'
    year=res.group(1)
    month=res.group(2)
    day=res.group(3)
    hour=res.group(4)
    mins=res.group(5)
    sec=res.group(6)    
    # add half a file length to be center of file
#    dt = TimeDelta(dt/2.0,format="sec")
    t0 = Time("%s-%s-%sT%s:%s:%s"%(year,month,day,hour,mins,sec), format='isot', scale='utc') #+ dt
    return(t0)