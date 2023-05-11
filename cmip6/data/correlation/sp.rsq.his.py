import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm
from cmip6util import mods
from utils import corr2d

nt=7 # window size in days
p=95
pref1='ddp'
varn1='tas'
pref2='dsp'
varn2='hfls'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc'
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

troplat=20    # latitudinal bound of tropics

largs=[
    {
    'landonly':False, # only use land grid points for rsq
    'troponly':False, # only look at tropics
    },
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':False, # only look at tropics
    },
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':True, # only look at tropics
    },
]

def calc_corr(flags):
    # grid
    rgdir='/project/amp/miyawaki/data/share/regrid'
    # open CESM data to get output grid
    cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
    cdat=xr.open_dataset(cfil)
    gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

    # load land mask
    lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

    if flags['troponly']:
        lats=np.transpose(np.tile(gr['lat'].data,(len(gr['lon']),1)),[1,0])
        lm[np.abs(lats)>troplat]=np.nan

    md='mi'
    idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
    idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn2)
    odir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    i1=pickle.load(open('%s/%s%s_%s_%s.%s.pickle' % (idir1,pref1,varn1,his,fut,se), 'rb'))
    i2=pickle.load(open('%s/%s%s_%s.%s.%s.pickle' % (idir2,pref2,varn2,his,p,se), 'rb'))

    if i2.shape != i1.shape:
        i2=np.transpose(i2[...,None],[0,1,4,2,3])

    if flags['landonly']:
        i1=i1*lm
        i2=i2*lm
        r=corr2d(i1,i2,gr,(3,4),lm=lm)
    else:
        r=corr2d(i1,i2,gr,(3,4))

    oname='%s/sp.rsq.%s_%s_%s.%s' % (odir,varn,his,fut,se)
    if flags['landonly']:
        oname='%s.land'%oname
    if flags['troponly']:
        oname='%s.trop'%oname

    pickle.dump(r,open('%s.pickle'%oname, 'wb'),protocol=5)

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_corr)(args) for args in largs]
        dask.compute(*tasks,scheduler='processes')
