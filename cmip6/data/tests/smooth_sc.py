import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from scipy.signal import butter,sosfiltfilt
from cmip6util import mods,simu,emem
from glade_utils import grid
from regions import pointlocs
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

# collect warmings across the ensembles
lre=['swus']

nt=30 # half-window size (days)
ft=np.ones(2*nt+1)
lday=np.arange(-nt,nt+1)
p=95
varn='tas' # input1
ty='2d'

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

freq='day'
se='sc'

# low pass butterworth filter
nf=10 # order
wn=1/10 # critical frequency [1/day]
lpf=butter(nf,wn,output='sos')

lmd=mods(fo) # create list of ensemble members

def calc_llp(md):
    print(md)
    ens=emem(md)
    grd=grid(md)

    for re in lre:
        print(re)
        iloc=pointlocs(re)

        idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        odir='/project/amp/miyawaki/plots/p004/cmip6/tests/%s'%(re)
        if not os.path.exists(odir):
            os.makedirs(odir)

        # load raw data
        if 'gwl' in byr:
            fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
        else:
            fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
        print('\n Loading data to composite...')
        ds = xr.open_mfdataset(fn)
        ds=ds.isel(lat=iloc[0],lon=iloc[1])
        vn = ds[varn].load()
        print('\n Done.')

        # day of year mean
        doy=list(set(vn['time.dayofyear'].data))
        svn=vn.groupby('time.dayofyear').mean('time')

        # low-pass filtered day of year mean
        fsvn=sosfiltfilt(lpf,svn.data)

        fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
        ax.plot(doy,svn.data,label='No smoothing')
        ax.plot(doy,fsvn,label='%g-day low-pass filtered'%(1/wn))
        ax.set_xlabel('Day of year')
        ax.set_ylabel(r'$T_{2\,m}$ (K)')
        ax.legend(loc='lower center',frameon=False)
        fig.savefig('%s/smooth_sc.pdf'%(odir),dpi=300,format='pdf')

calc_llp('CESM2')
