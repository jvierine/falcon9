
import numpy as np
import pandas as pd
import os
import gzip

def readYBC(filename='bsc5.dat.gz', path=None):
    """Read in the Yale Bright Star Catalog. Originally downloaded from:
    http://tdc-www.harvard.edu/catalogs/bsc5.html

    Adapted from code by Peter Joachim.
    https://github.com/lsst/all_sky_phot/blob/main/python/lsst/all_sky_phot/read_ybc.py
    with one minor fix on the names array
    use gzipped file to save space
    """
    names = ['HR', 'Name', 'DM', 'HD', 'SAO', 'FK5', 'IRflag', 'r_IRflag', 'Multiple','ADS', 'ADScomp', 'VarID',
             'RAh1900', 'RAm1900', 'RAs1900', 'DE-1900', 'DEd1900', 'DEm1900', 'DEs1900', 'RAh', 'RAm', 'RAs',
             'DE-', 'DEd', 'DEm', 'DEs', 'GLON', 'GLAT', 'Vmag', 'n_Vmag', 'u_Vmag', 'B-V', 'u_B-v', 'U-B',
             'u_U-B', 'R-I', 'n_R-I', 'SpType', 'n_SpType', 'pmRA', 'pmDE', 'pmDE2', 'Parallax', 'RadVel',
             'n_RadVel', 'l_RotVel', 'RotVel', 'u_RotVel', 'Dmag', 'Sep', 'MultID', 'MultCnt',
             'NoteFlag']

    colspecs = [(0, 3), (4, 13), (14, 24), (25, 30), (31, 36), (37, 40), (41, 41), (42, 42),
                (43, 43), (44, 48), (49, 50), (51, 59), (60, 61), (62, 63), (64, 67), (68, 68),
                (69, 70), (71, 72), (73, 74), (75, 76), (77, 78), (79, 82), (83, 83), (84, 85),
                (86, 87), (88, 89), (90, 95), (96, 101), (102, 106), (107, 107), (108, 108),
                (109, 113), (114, 114), (115, 119), (120, 120), (121, 125), (126, 126), (127, 146),
                (147, 147), (148, 153), (154, 159), (160, 160), (161, 165), (166, 169), (170, 173),
                (174, 175), (176, 178), (179, 179), (180, 183), (184, 189), (190, 193), (194, 195),
                (196, 196)]
    fix_colspecs = [(spec[0], spec[1]+1) for spec in colspecs]

    data = pd.read_fwf(gzip.open(filename), names=names, colspecs=fix_colspecs)
    data['RA'] = (data['RAh'] + data['RAm']/60. + data['RAs']/3600.)*360./24.
    data['Dec'] = data['DEd'] + data['DEm']/60. + data['DEs']/3600.
    sign = np.ones(data['RAh'].size)
    sign[np.where(data['DE-'] == '-')] *= -1
    data['Dec'] = data['Dec'] * sign

    return(data)

class bright_stars:
    def __init__(self,fname="bsc5.dat.gz"):
        self.data=readYBC(fname)
        self.Vmags=np.copy(self.data["Vmag"]*1.0)
        # sort by visual magnitude
        self.idxs=np.argsort(self.Vmags)
        #print(self.idxs)

    def get_ra_dec_vmag(self,i):
        return(self.data["RA"][self.idxs[i]],self.data["Dec"][self.idxs[i]],self.data["Vmag"][self.idxs[i]] )

if __name__ == "__main__":
    bs=bright_stars()
    for i in range(100):
        d=bs.get_ra_dec_vmag(i)
        print(d)



