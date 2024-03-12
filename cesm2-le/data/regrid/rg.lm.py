#!/glade/work/miyawaki/conda-envs/g/bin/python
#PBS -l select=1:ncpus=10:mpiprocs=10
#PBS -l walltime=06:00:00
#PBS -q casper
#PBS -A P54048000
#PBS -N rg.lm.py

import os
import sys
sys.path.append('../')
sys.path.append('/glade/u/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import numpy as np
import xarray as xr
import constants as c
from scipy.ndimage import generic_filter1d
from tqdm import tqdm
from util import emem,simu,casename,rename_vn
from cesmutils import realm,history
np.set_printoptions(threshold=sys.maxsize)

# comdsect warmings across the ensembles

varn='trefht' # input1

ty='2d'
checkexist=False

fo='lens2' # forcing (e.g., ssp245)
lsim=np.arange(100)
byr=[1950,2020]

freq='day'
se='ts'

# load ocean indices
_,omi=pickle.load(open('../lomi.pickle','rb'))

md='CESM2'


def lm(isim):
    mem=emem(fo)[isim]
    sim=simu(fo)[isim]
    rlm=realm(varn)
    hst=history(varn)
    ovn=rename_vn(varn)

    idir='/glade/campaign/cesm/collections/CESM2-LE/%s/proc/tseries/%s_1/%s' % (rlm,freq,varn.upper())

    odir='/glade/campaign/cgd/cas/miyawaki/data/share/cesm2-le/%s/%s/%s.%s/%s' % (se,fo,md,mem,ovn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if checkexist:
        if 'gwl' in byr:
            if os.path.isfile('%s/lm.%s_%s.%s.nc' % (odir,vn,byr,se)):
                print('Output file already exists, skipping...')
                return
        else:
            if os.path.isfile('%s/lm.%s_%g-%g.%s.nc' % (odir,vn,byr[0],byr[1],se)):
                print('Output file already exists, skipping...')
                return

    # load raw data
    files=list(os.walk(idir))[0][2]
    files=[f for f in files if sim in f]
    fn=['%s/%s'%(idir,f) for f in files]
    print('\n Loading data to composite...')
    vn=xr.open_mfdataset(fn)[varn.upper()]

    # select time
    print('\n Selecting data within range of interest...')
    vn=vn.sel(time=vn['time.year']>=byr[0])
    vn=vn.sel(time=vn['time.year']<byr[1])
    print('\n Done.')

    # remove ocean points
    time=vn['time']
    vn=vn.data
    vn=np.reshape(vn,(vn.shape[0],vn.shape[1]*vn.shape[2]))
    vn=np.delete(vn,omi,axis=1)
    ngp=vn.shape[1]

    vn=xr.DataArray(vn,coords={'time':time,'gpi':np.arange(ngp)},dims=('time','gpi'))

    vn=vn.rename(ovn)
    if 'gwl' in byr:
        vn.to_netcdf('%s/lm.%s_%s.%s.nc' % (odir,ovn,byr,se),format='NETCDF4')
    else:
        vn.to_netcdf('%s/lm.%s_%g-%g.%s.nc' % (odir,ovn,byr[0],byr[1],se),format='NETCDF4')

# lm(0)

if __name__=='__main__':
    with Pool(max_workers=len(lsim)) as p:
        p.map(lm,lsim)
