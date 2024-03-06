# This script contains a set of utilities for the CESM2-CMIP runs

import sys
import xarray as xr
# sys.path.append('/home/miyawaki/scripts/common/CASanalysis/CASutils')

def casename(fo):
    d={ 
        'historical':   'b.e21.BHIST.f09_g17.CMIP6-historical.011',
            'ssp370':   'b.e21.BSSP370cmip6.f09_g17.CMIP6-SSP3-7.0.102',
            }
    return d[fo]

def load_raw(odir,varn,byr,se):
    # load raw data
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (odir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
    print('\n Loading data to composite...')
    ds = xr.open_dataset(fn)
    print('\n Done.')
    return ds

def rename_vn(varn):
    d={
        'fsm':      'snm',
        'fsno':     'snc',
            }
    try:
        return d[varn]
    except:
        return varn
