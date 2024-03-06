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

# lnt=[30,60,90,7,14] # number of days to accumulate data
lnt=[7,14] # number of days to accumulate data
varni='pr' # input
varn='cpr' # output
ty='2d'

lfo = ['historical'] # forcing (e.g., ssp245)
byr=[1980,2000]

# lfo = ['ssp370'] # forcing (e.g., ssp245)
# byr=[2080,2100]

freq='day'
lse = ['jja','djf','mam','son'] # season (ann, djf, mam, jja, son)

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

for nt in lnt:
    for fo in lfo:
        for se in lse:
            lmd=mods(fo) # create list of ensemble members

            c0=0 # first loop counter
            for imd in tqdm(range(len(lmd))):
                md=lmd[imd]
                ens=emem(md)
                grd=grid(fo,cl,md)

                if md in ['CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'FGOALS-g3','INM-CM4-8', 'INM-CM5-0','KACE-1-0-G', 'MIROC6','MIROC-ES2L','MPI-ESM-1-2-HAM','MRI-ESM2-0','NorESM2-LM','NorESM2-MM','UKESM1-0-LL']:
                    idir='/project/amp/miyawaki/temp/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varni,md,ens,grd)
                else:
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
                pr = ds[varni].load()
                ds=None
                # convert to mm/d
                pr=pr*86400
                spr=np.cumsum(pr,axis=0)
                sspr=np.concatenate([np.transpose(np.tile(np.array([np.NaN]*(nt-1)),(pr.shape[1],pr.shape[2],1)),[2,0,1]), np.tile(np.array([0]),(1,pr.shape[1],pr.shape[2])), spr[:-nt,:,:]],axis=0)
                # compute precip accumulated over previous nt days
                cpr=pr.copy(data=spr-sspr)
                # normalize back to mm/d
                cpr=cpr/nt

                # select data within time of interest
                cpr=cpr.sel(time=cpr['time.year']>=byr[0])
                cpr=cpr.sel(time=cpr['time.year']<=byr[1])

                # select seasonal data if applicable
                if se != 'ann':
                    cpr=cpr.sel(time=cpr['time.season']==se.upper())

                if md!='CESM2':
                    # path to weight file
                    wf='%s/wgt.cmip6.%s.%s.cesm2.nc'%(rgdir,md,ty)
                    # build regridder with existing weights
                    rgd = xe.Regridder(cpr,ogr, 'bilinear', periodic=True, reuse_weights=True, filename=wf)
                    # regrid
                    cpr=rgd(cpr)

                cpr=cpr.rename(varn)
                cpr.to_netcdf('%s/cl%s%03d_%g-%g.%s.nc' % (odir,varn,nt,byr[0],byr[1],se))
