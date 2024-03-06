import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.distributed import Client
import dask.multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from etregimes import bestfit

qvar='qsoil'

ld=np.concatenate(([10],np.arange(20,100,20),np.arange(100,850,50)))

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]
se='sc'

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'

freq='day'

md='CESM2'
# lmd=mods(fo) # create list of ensemble members

def get_fn(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def calc_bc(depth):
    print(md)
    ens=emem(md)
    grd=grid(md)
    varn='bc%g.%s'%(depth,qvar)
    vmrs='mrso%g'%depth

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn(qvar,md,fo,byr))
    gpi=ds['gpi']
    time=ds['time']
    vn1=ds[qvar]
    vn2=xr.open_dataset(get_fn(vmrs,md,fo,byr))[vmrs]
    if fo=='historical':
        vn0=vn2.copy()
    else:
        vn0=xr.open_dataset(get_fn(vmrs,md,fo0,byr0))[vmrs]
    print('\n Done.')

    print('\n Computing soil moisture anomaly...')
    # vn2=vn2.groupby('time.month')-vn0.groupby('time.month').mean('time')
    vn2=vn2-vn0.mean('time')
    print('\n Done.')

    print('\n Computing budyko curve...')
    def bcmon(mon,vn1,vn2):
        svn1=vn1.sel(time=vn1['time.month']==mon)
        svn2=vn2.sel(time=vn2['time.month']==mon)
        bc=[]
        csm=np.nan*np.ones(len(gpi))
        mtr=np.nan*np.ones(len(gpi))
        for igpi in range(len(gpi)):
            nvn1=svn1.data[...,igpi].flatten()
            nvn2=svn2.data[...,igpi].flatten()
            nans=np.logical_or(np.isnan(nvn1),np.isnan(nvn2))
            nvn1=nvn1[~nans]
            nvn2=nvn2[~nans]
            try:
                f1,f2=bestfit(nvn2,nvn1)
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
        odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,svn)
        if not os.path.exists(odir):
            os.makedirs(odir)
        vn=xr.DataArray(vn,coords={'month':np.arange(1,13,1),'gpi':np.arange(len(gpi))},dims=('month','gpi'))
        vn=vn.rename(svn)
        if 'gwl' in byr:
            oname='%s/%s.%s.%s.nc' % (odir,svn,byr,se)
        else:
            oname='%s/%s.%g-%g.%s.nc' % (odir,svn,byr[0],byr[1],se)
        vn.to_netcdf(oname,format='NETCDF4')

    save_vn(csm,'csm%g.%s'%(depth,qvar))
    save_vn(mtr,'mtr%g.%s'%(depth,qvar))

if __name__=='__main__':
    # calc_bc(400)
    [calc_bc(depth) for depth in tqdm(ld)]

# if __name__=='__main__':
#     with Pool(max_workers=len(ld)) as p:
#         p.map(calc_bc,ld)
