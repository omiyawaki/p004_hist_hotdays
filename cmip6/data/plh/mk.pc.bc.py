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
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from etregimes import bestfit

# collect warmings across the ensembles

varn='bc'
se='sc'
p=95
nt=7 # number of days for window (nt days before and after)

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_fn(varn,md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def get_fnhd(varn,md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/p.%s%03d_%s.%s.nc' % (idir,varn,nt,byr,se)
    else:
        fn='%s/p.%s%03d_%g-%g.%s.nc' % (idir,varn,nt,byr[0],byr[1],se)
    return fn

def calc_bc(md):
    print(md)
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn('hfls',md))
    gpi=ds['gpi']
    time=ds['time']
    vn1=ds['hfls']
    vn2=xr.open_dataset(get_fn('mrsos',md))['mrsos']
    vn3=xr.open_dataset(get_fn('tas',md))['tas']
    print('\n Done.')

    print('\n Loading hot day data...')
    ptas=xr.open_dataarray(get_fnhd('tas',md))
    ptas=ptas.sel(percentile=p)
    print('\n Done.')

    print('\n Computing budyko curve...')
    # create list to store bcs
    bc = [ ([0] * len(gpi)) for im in range(12) ]

    def bcmon(mon,vn1,vn2,vn3):
        svn1=vn1.sel(time=vn1['time.month']==mon)
        svn2=vn2.sel(time=vn2['time.month']==mon)
        svn3=vn3.sel(time=vn3['time.month']==mon)
        iday=list(set(svn1['time.dayofyear'].data))
        stas=ptas.sel(doy=iday).mean(dim='doy')
        ihd=svn3.data-np.transpose(stas.data[:,None],[1,0])
        svn1.data[ihd<0]=np.nan
        svn2.data[ihd<0]=np.nan
        svn3.data[ihd<0]=np.nan
        bc=[]
        for igpi in tqdm(range(len(gpi))):
            # remove nans
            nvn1=svn1.data[:,igpi].flatten()
            nvn2=svn2.data[:,igpi].flatten()
            nans=np.logical_or(np.isnan(nvn1),np.isnan(nvn2))
            nvn1=nvn1[~nans]
            nvn2=nvn2[~nans]
            try:
                f1,f2=bestfit(nvn1,nvn2)
                bc.append(f2['line'])
            except:
                bc.append(None)
        return bc

    with ProgressBar():
        tasks=[dask.delayed(bcmon)(mon,vn1,vn2,vn3) for mon in np.arange(1,13,1)]
        bc=dask.compute(*tasks,scheduler='threads')

    # save bc
    oname='%s/pc%03d.%s.%g-%g.%s.pickle' % (odir,p,varn,byr[0],byr[1],se)
    pickle.dump(bc,open(oname,'wb'),protocol=5)

if __name__=='__main__':
    # [calc_bc(md) for md in lmd]
    calc_bc('CESM2')

# if __name__ == '__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_bc)(md) for md in lmd]
#         dask.compute(*tasks,scheduler='processes')
#         # dask.compute(*tasks,scheduler='single-threaded')
