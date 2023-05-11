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

# collect warmings across the ensembles

varni='hfls' # input2
varn='ev' # output
ty='2d'

# lfo = ['historical'] # forcing (e.g., ssp245)
# byr=[1980,2000]

lfo = ['ssp370'] # forcing (e.g., ssp245)
byr=[2080,2100]

freq='day'
lse = ['jja','djf','mam','son'] # season (ann, djf, mam, jja, son)

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

for fo in lfo:
    for se in lse:
        lmd=mods(fo) # create list of ensemble members

        c0=0 # first loop counter
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            ens=emem(md)
            grd=grid(fo,cl,md)

            idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varni,md,ens,grd)

            odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
            if not os.path.exists(odir):
                os.makedirs(odir)

            fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir,varni,freq,md,fo,ens,grd)

            ds = xr.open_mfdataset(fn)
            # save grid info
            gr = {}
            gr['lon'] = ds['lon']
            gr['lat'] = ds['lat']
            lh = ds[varni].load()
            # convert lh to ~evap
            ev=lh/c.Lv.magnitude

            # select data within time of interest
            ev=ev.sel(time=ev['time.year']>=byr[0])
            ev=ev.sel(time=ev['time.year']<=byr[1])

            # select seasonal data if applicable
            if se != 'ann':
                ev=ev.sel(time=ev['time.season']==se.upper())

            if md!='CESM2':
                # path to weight file
                wf='%s/wgt.cmip6.%s.%s.cesm2.nc'%(rgdir,md,ty)
                # build regridder with existing weights
                rgd = xe.Regridder(ev,ogr, 'bilinear', periodic=True, reuse_weights=True, filename=wf)
                # regrid
                ev=rgd(ev)

            ev=ev.rename(varn)
            ev.to_netcdf('%s/cl%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se))
