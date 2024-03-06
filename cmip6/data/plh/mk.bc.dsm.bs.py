import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
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
from sklearn.utils import resample

# number of times to resample for bootstrap
nbs=500

# collect warmings across the ensembles

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]
varn='bc'
se='sc'

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'

freq='day'

lmd=mods(fo) # create list of ensemble members
# md='CESM2'

def get_fn(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def calc_bc(md):
    print(md)
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn('hfls',md,fo,byr))
    gpi=ds['gpi']
    time=ds['time']
    vn1=ds['hfls']
    vn2=xr.open_dataset(get_fn('mrsos',md,fo,byr))['mrsos']
    if fo=='historical':
        vn0=vn2.copy()
    else:
        vn0=xr.open_dataset(get_fn('mrsos',md,fo0,byr0))['mrsos']
    print('\n Done.')

    print('\n Computing soil moisture anomaly...')
    # vn2=vn2.groupby('time.month')-vn0.groupby('time.month').mean('time')
    vn2=vn2-vn0.mean('time')
    print('\n Done.')

    print('\n Computing budyko curve...')

    def bcmon(mon,vn1,vn2):
        svn1=vn1.sel(time=vn1['time.month']==mon)
        svn2=vn2.sel(time=vn2['time.month']==mon)
        bc=[[] for _ in range(len(gpi))]
        csm=np.nan*np.ones([len(gpi),nbs])

        # for igpi in range(len(gpi)):
        for igpi in range(len(gpi)):
            nvn1=svn1.data[...,igpi].flatten()
            nvn2=svn2.data[...,igpi].flatten()
            nans=np.logical_or(np.isnan(nvn1),np.isnan(nvn2))
            nvn1=nvn1[~nans]
            nvn2=nvn2[~nans]

            nvn=np.stack((nvn1,nvn2),axis=-1)
            for ibs in range(nbs):
                bvn=resample(nvn)
                try:
                    f1,f2=bestfit(bvn[:,1],bvn[:,0])
                    bc[igpi].append(f2['line'])
                    csm[igpi,ibs]=f2['xc']
                except:
                    bc[igpi].append(None)
        return bc,csm

    with Client(n_workers=12):
        tasks=[dask.delayed(bcmon)(mon,vn1,vn2) for mon in np.arange(1,13,1)]
        l=dask.compute(*tasks)
        bc=[il[0] for il in l]
        csm=[il[1] for il in l]
        csm=np.stack(csm,axis=0)

    # save bc
    if 'gwl' in byr:
        oname='%s/%s_bs.%s.%s.pickle' % (odir,varn,byr,se)
    else:
        oname='%s/%s_bs.%g-%g.%s.pickle' % (odir,varn,byr[0],byr[1],se)
    pickle.dump(bc,open(oname,'wb'),protocol=5)

    # save csm
    csm=xr.DataArray(csm,coords={'month':np.arange(1,13,1),'gpi':np.arange(len(gpi)),'sample':range(nbs)},dims=('month','gpi','sample'))
    csm=csm.rename('csm')
    if 'gwl' in byr:
        oname='%s/%s_bs.%s.%s.nc' % (odir,varn,byr,se)
    else:
        oname='%s/%s_bs.%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
    csm.to_netcdf(oname,format='NETCDF4')

if __name__=='__main__':
    [calc_bc(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_bc,lmd)

# if __name__ == '__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_bc)(md) for md in lmd]
#         dask.compute(*tasks,scheduler='processes')
#         # dask.compute(*tasks,scheduler='single-threaded')
