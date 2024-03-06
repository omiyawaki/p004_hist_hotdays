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

def calc_bc(md):
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
    print('\n Done.')

    print('\n Computing budyko curve...')
    # create list to store bcs
    bc = [ ([0] * len(gpi)) for im in range(12) ]

    def bcmon(mon,vn1,vn2):
        svn1=vn1.sel(time=vn1['time.month']==mon)
        svn2=vn2.sel(time=vn2['time.month']==mon)
        bc=[]
        for igpi in tqdm(range(len(gpi))):
            try:
                f1,f2=bestfit(svn2.data[:,igpi].flatten(),svn1.data[:,igpi].flatten())
                bc.append(f2['line'])
            except:
                bc.append(None)
        return bc

    with ProgressBar():
        tasks=[dask.delayed(bcmon)(mon,vn1,vn2) for mon in np.arange(1,13,1)]
        bc=dask.compute(*tasks,scheduler='threads',num_workers=12)

    # save bc
    oname='%s/%s.%g-%g.%s.pickle' % (odir,varn,byr[0],byr[1],se)
    pickle.dump(bc,open(oname,'wb'),protocol=5)

if __name__=='__main__':
    calc_bc('CanESM5')

# if __name__ == '__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_bc)(md) for md in lmd]
#         dask.compute(*tasks,scheduler='processes')
#         # dask.compute(*tasks,scheduler='single-threaded')
