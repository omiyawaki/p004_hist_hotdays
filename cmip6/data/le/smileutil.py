# This script contains a set of utilities for the CESM2-SF (single forcing) runs

import sys
import xarray as xr
# sys.path.append('/home/miyawaki/scripts/common/CASanalysis/CASutils')

def mods():
    return sorted(['ACCESS-ESM1-5','CanESM5','EC-Earth3','MIROC6','MPI-ESM1-2-LR','UKESM1-0-LL'])
