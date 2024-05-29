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
nbs=100

# collect warmings across the ensembles

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]
varn='bc'
se='sc'

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'

freq='day'

seas='summer'
# lreg=['saf','sea','shl','ind','ca']
lreg=['sa','saf','sea','shl','ind','ca']

# lmd=mods(fo) # create list of ensemble members
md='CESM2'

# load gpi lat data
llat,llon=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lmilatlon.pickle','rb'))

def get_fn(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def calc_bc(reg):
    print(md)
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if 'gwl' in byr:
        oname='%s/%s_raw.%s.%s.%s.pickle' % (odir,varn,byr,se,reg)
    else:
        oname='%s/%s_raw.%g-%g.%s.%s.pickle' % (odir,varn,byr[0],byr[1],se,reg)

    if os.path.exists(oname):
        print('\n Reloading data...')
        svn1,svn2=pickle.load(open(oname,'rb'))
    else:
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
        mgpi=gpi.copy()
        msk=pickle.load(open('/project/amp02/miyawaki/data/p004/cmip6/hotspots/%s.pickle'%reg,'rb'))
        mgpi=mgpi[msk==1] # gridpoints to keep (i.e., inside mask)

        def xsel(xvn,seas):
            slat=gpi.isel(gpi=mgpi)
            sllat=llat[slat]
            xvn=xvn.isel(gpi=mgpi)
            latnh=slat[sllat>=0]
            latsh=slat[sllat<0]
            if seas=='summer':
                xvnnh=xvn.sel(time=xvn['time.month'].isin([6,7,8])).sel(gpi=latnh)
                xvnsh=xvn.sel(time=xvn['time.month'].isin([12,1,2])).sel(gpi=latsh)
            if xvnnh.shape[1]==0:
                return xvnsh.data.flatten()
            elif xvnsh.shape[1]==0:
                return xvnnh.data.flatten()
            else:
                return np.concatenate((xvnnh.data.flatten(),xvnsh.data.flatten()))

        svn1=xsel(vn1,seas)
        svn2=xsel(vn2,seas)

        # save raw data
        pickle.dump([svn1,svn2],open(oname,'wb'),protocol=5)

    bc=[]
    csm=np.nan*np.ones(nbs)
    mt=np.nan*np.ones(nbs)

    nans=np.logical_or(np.isnan(svn1),np.isnan(svn2))
    nvn1=svn1[~nans]
    nvn2=svn2[~nans]

    nvn=np.stack((nvn1,nvn2),axis=-1)
    for ibs in tqdm(range(nbs)):
        bvn=resample(nvn)
        try:
            f1,f2=bestfit(bvn[:,1],bvn[:,0])
            bc.append(f2['line'])
            csm[ibs]=f2['xc']
            mt[ibs]=f2['mt']
        except:
            bc.append(None)

    # save bc
    if 'gwl' in byr:
        oname='%s/%s_bs.%s.%s.%s.pickle' % (odir,varn,byr,se,reg)
    else:
        oname='%s/%s_bs.%g-%g.%s.%s.pickle' % (odir,varn,byr[0],byr[1],se,reg)
    pickle.dump(bc,open(oname,'wb'),protocol=5)

    # save csm
    csm=xr.DataArray(csm,coords={'sample':range(nbs)},dims=('sample'))
    csm=csm.rename('csm')
    if 'gwl' in byr:
        oname='%s/%s_bs.%s.%s.%s.nc' % (odir,varn,byr,se,reg)
    else:
        oname='%s/%s_bs.%g-%g.%s.%s.nc' % (odir,varn,byr[0],byr[1],se,reg)
    csm.to_netcdf(oname,format='NETCDF4')

    # save mt
    mt=xr.DataArray(mt,coords={'sample':range(nbs)},dims=('sample'))
    mt=mt.rename('mt')
    if 'gwl' in byr:
        oname='%s/%s_bs.%s.%s.%s.nc' % (odir,varn,byr,se,reg)
    else:
        oname='%s/%s_bs.%g-%g.%s.%s.nc' % (odir,varn,byr[0],byr[1],se,reg)
    mt.to_netcdf(oname,format='NETCDF4')

if __name__=='__main__':
    [calc_bc(reg) for reg in tqdm(lreg)]
    # calc_bc(lreg[0])

# if __name__=='__main__':
#     with Pool(max_workers=len(lreg)) as p:
#         p.map(calc_bc,lreg)

