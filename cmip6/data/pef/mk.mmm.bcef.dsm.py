import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
import dask.multiprocessing
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from etregimes import bestfit_hd22

# collect warmings across the ensembles

varn='bcef'
vn1='ef'
vn2='mrsos'
se='sc'

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_fn(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def load_vn(vn,md,mon,time0):
    varn=xr.open_dataset(get_fn(vn,md,fo,byr))[vn]
    if vn=='mrsos':
        if fo=='historical':
            vn0=varn.copy()
        else:
            vn0=xr.open_dataset(get_fn(vn,md,fo0,byr0))[vn]
        print('\n Computing soil moisture anomaly...')
        varn=varn-vn0.mean('time')
    print('\n Done.')

    varn=varn.interp_calendar(time0)
    varn=varn.sel(time=varn['time.month']==mon)
    return varn

def calc_bc(lmd):
    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mmm',varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    ds=xr.open_dataset(get_fn(vn1,'CESM2',fo,byr))
    time0=ds['time']
    gpi=ds['gpi']

    print('\n Computing budyko curve...')
    def bcmon(mon,vn1,vn2):
        svn1=[load_vn(vn1,md,mon,time0) for md in lmd]
        svn1=xr.concat(svn1,'model')
        svn2=[load_vn(vn2,md,mon,time0) for md in lmd]
        svn2=xr.concat(svn2,'model')
        bc=[]
        csm=np.nan*np.ones(len(gpi))
        mtr=np.nan*np.ones(len(gpi))
        for igpi in tqdm(range(len(gpi))):
            # remove nans 
            nvn1=svn1.data[...,igpi].flatten()
            nvn2=svn2.data[...,igpi].flatten()
            nans=np.logical_or(np.isnan(nvn1),np.isnan(nvn2))
            nvn1=nvn1[~nans]
            nvn2=nvn2[~nans]
            # remove where ef>1
            efg1=nvn1>1
            nvn1=nvn1[~efg1]
            nvn2=nvn2[~efg1]
            try:
                f1,f2=bestfit_hd22(nvn2,nvn1)
                bc.append(f2['line'])
                csm[igpi]=f2['xc']
                mtr[igpi]=f2['mt']
            except:
                bc.append(None)
        return bc,csm,mtr

    with Client(n_workers=12):
        tasks=[dask.delayed(bcmon)(mon,vn1,vn2) for mon in np.arange(1,13,1)]
        l=dask.compute(*tasks)
        bc=[il[0] for il in l]
        csm=[il[1] for il in l]
        mtr=[il[2] for il in l]
        csm=np.stack(csm)
        mtr=np.stack(mtr)

    # save bc
    if 'gwl' in byr:
        oname='%s/%s.%s.%s.pickle' % (odir,varn,byr,se)
    else:
        oname='%s/%s.%g-%g.%s.pickle' % (odir,varn,byr[0],byr[1],se)
    pickle.dump(bc,open(oname,'wb'),protocol=5)

    def save_vn(vn,svn):
        odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mmm',svn)
        if not os.path.exists(odir):
            os.makedirs(odir)
        vn=xr.DataArray(vn,coords={'month':np.arange(1,13,1),'gpi':np.arange(len(gpi))},dims=('month','gpi'))
        vn=vn.rename(svn)
        if 'gwl' in byr:
            oname='%s/%s.%s.%s.nc' % (odir,svn,byr,se)
        else:
            oname='%s/%s.%g-%g.%s.nc' % (odir,svn,byr[0],byr[1],se)
        vn.to_netcdf(oname,format='NETCDF4')

    save_vn(csm,'csm')
    save_vn(mtr,'mtr')

if __name__=='__main__':
    calc_bc(lmd)
