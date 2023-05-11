import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
from tqdm import tqdm
from cmip6util import mods,simu,emem
from glade_utils import grid

# collect warmings across the ensembles

varn='crd'
ivar0='dsm'
ivar2='cpe'
ty='2d'

lnt=[30]
lfo = ['historical'] # forcing (e.g., ssp245)
byr=[1980,2000]

# lfo = ['ssp370'] # forcing (e.g., ssp245)
# byr=[2080,2100]

freq='day'
lse = ['jja'] # season (ann, djf, mam, jja, son)

for nt in lnt:
    ivar1='%s%03d'%(ivar0,nt)
    ivar3='%s%03d'%(ivar2,nt)
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
                fn1 = '%s/%s/cl%s_%g-%g.%s.nc' % (idir,ivar0,ivar1,byr[0],byr[1],se)
                fn2 = '%s/%s/cl%s_%g-%g.%s.nc' % (idir,ivar2,ivar3,byr[0],byr[1],se)
                ds = xr.open_mfdataset(fn1)
                var1 = ds[ivar0].load()
                ds = xr.open_mfdataset(fn2)
                var2 = ds[ivar2].load()

                crd=var2-var1
                print(np.all(crd>0))

                # save grid info
                gr = {}
                gr['lon'] = ds['lon']
                gr['lat'] = ds['lat']

                crd=crd.rename(varn)
                crd.to_netcdf('%s/cl%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se))
