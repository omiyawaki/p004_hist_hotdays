import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
from tqdm import tqdm
from scipy.interpolate import interp1d
from cmip6util import mods,simu,emem
from glade_utils import grid

# collect warmings across the ensembles

varn='orog'
ty='2d'

fo = 'historical' # forcing (e.g., ssp245)

# fo = 'ssp370' # forcing (e.g., ssp245)

freq='fx'
se = 'fx' # season (ann, djf, mam, jja, son)

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

lmd=mods(fo) # create list of ensemble members

def rg_orog(md):
    ens=emem(md)
    grd=grid(fo,cl,md)

    idir='/project/amp/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)

    odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    fn = '%s/%s_%s_%s_%s_%s_%s.nc' % (idir,varn,freq,md,fo,ens,grd)

    try:
        ds = xr.open_mfdataset(fn)
        orog = ds[varn].load()
    except:
        print('orog not available. Inter/extrapolating using zg and ps...')
        idir0='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,'day','zg',md,ens,grd)
        fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir0,'zg','day',md,fo,ens,grd)
        ds = xr.open_mfdataset(fn)
        zg = ds['zg'].load()
        pl=ds['plev'].load()
        zg=zg.mean(dim='time')
        idir0='/project/amp/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,'Amon','ps',md,ens,grd)
        fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir0,'ps','Amon',md,fo,ens,grd)
        ds = xr.open_mfdataset(fn)
        ps = ds['ps'].load()
        ps=ps.mean(dim='time')
        orog=ps.copy()
        for ilo in tqdm(range(ps.shape[1])):
            for ila in range(ps.shape[0]):
                lzg=zg[:,ila,ilo]
                lps=ps[ila,ilo]
                lzg=lzg[pl<lps]
                lpl=pl[pl<lps]
                fint=interp1d(lpl,lzg,bounds_error=False,fill_value='extrapolate')
                orog.data[ila,ilo]=fint(lps)
        if not os.path.exists(idir):
            os.makedirs(idir)
        orog=orog.rename(varn)
        orog.to_netcdf('%s/%s_%s_%s_%s_%s_%s.nc' % (idir,varn,freq,md,fo,ens,grd))

    # save grid info
    gr = {}
    gr['lon'] = ds['lon']
    gr['lat'] = ds['lat']

    if md!='CESM2':
        # path to weight file
        wf='%s/wgt.cmip6.%s.%s.cesm2.nc'%(rgdir,md,ty)
        # build regridder with existing weights
        rgd = xe.Regridder(orog,ogr, 'bilinear', periodic=True, reuse_weights=True, filename=wf)
        # regrid
        orog=rgd(orog)

    orog=orog.rename(varn)
    orog.to_netcdf('%s/cl%s.%s.nc' % (odir,varn,se))

rg_orog('IITM-ESM')

# if __name__=='__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(rg_orog)(md) for md in lmd]
#         dask.compute(*tasks,scheduler='processes')
