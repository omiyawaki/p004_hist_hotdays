import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from cmip6util import mods,simu,emem
from glade_utils import grid
from metpy.calc import saturation_mixing_ratio,specific_humidity_from_mixing_ratio
from metpy.units import units

# collect warmings across the ensembles

varn='predtm'
ivars=['m500s','orog']

# lfo = ['historical'] # forcing (e.g., ssp245)
# byr=[1980,2000]

lfo = ['ssp370'] # forcing (e.g., ssp245)
byr=[2080,2100]

freq='day'
lse = ['jja'] # season (ann, djf, mam, jja, son)

for fo in lfo:
    for se in lse:
        lmd=mods(fo) # create list of ensemble members

        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            ens=emem(md)
            grd=grid(fo,cl,md)

            idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,cl,fo,md)
            odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
            if not os.path.exists(odir):
                os.makedirs(odir)

            c0=0 # first loop counter
            for iv in range(len(ivars)):
                ivar=ivars[iv]
                print(md)
                print(ivar)
                if ivar=='orog':
                    idir0='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % ('fx',cl,fo,md)
                    fn = '%s/%s/cl%s.%s.nc' % (idir0,ivar,ivar,'fx')
                else:
                    fn = '%s/%s/cl%s_%g-%g.%s.nc' % (idir,ivar,ivar,byr[0],byr[1],se)
                ds = xr.open_mfdataset(fn)
                svar = ds[ivar].load()

                if c0==0:
                    inpv={}
                    c0=1

                inpv[ivar]=svar

            # compute predicted maximum temp
            predtm=1/c.cpd*(inpv['m500s']*units.joule/units.kg-c.g*inpv['orog']*units.m)

            # save grid info
            gr = {}
            gr['lon'] = ds['lon']
            gr['lat'] = ds['lat']

            predtm=predtm.rename(varn)
            predtm.to_netcdf('%s/cl%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se))
