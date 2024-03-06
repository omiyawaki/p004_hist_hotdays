import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import pickle
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import monname
from glade_utils import smile

lvn=['tas']
se = 'sc' # season (ann, djf, mam, jja, son)

fo='historical' # forcings 
yr='1980-2000'

# fo='ssp370' # forcings 
# yr='2080-2100'

# fut='gwl2.0'

lmd=['CanESM5']

def calc_em(md,varn):
    lme=smile(md,fo,varn)

    for i,me in enumerate(tqdm(lme)):
        print(fo,varn,me)
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)

        # load data
        evn=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir,varn,yr,se))[varn]

        # save individual model data
        if i==0:
            ievn=np.empty(np.insert(np.asarray(evn.shape),0,len(lme)))

        ievn[i,...]=evn

    # compute em and std
    mevn=evn.copy()
    mevn.data=np.nanmean(ievn,axis=0)

    sevn=mevn.copy()
    sevn.data=np.nanstd(ievn,axis=0)

    cevn=mevn.copy()
    cevn.data=1.96*sevn.data/np.sqrt(len(lme))

    # save em and std
    odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    mevn.to_netcdf('%s/lm.%s_%s.em.%s.nc' % (odir,varn,yr,se))
    sevn.to_netcdf('%s/std.lm.%s_%s.em.%s.nc' % (odir,varn,yr,se))
    cevn.to_netcdf('%s/ci.lm.%s_%s.em.%s.nc' % (odir,varn,yr,se))


for md in tqdm(lmd):
    if __name__=='__main__':
        with ProgressBar():
            tasks=[dask.delayed(calc_em)(md,vn) for vn in lvn]
            dask.compute(*tasks,scheduler='processes')
            # dask.compute(*tasks,scheduler='single-threaded')
