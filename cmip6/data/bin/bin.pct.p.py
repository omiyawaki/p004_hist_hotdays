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
from scipy.stats import linregress, rankdata
from tqdm import tqdm
from cmip6util import mods
from utils import monname

nt=7 # window size in days
varn='tas'
vnpre='ddp' # prefix of variable to compute
bnpre='dp' # prefix of variable to bin
lpc=np.arange(5,95+5,5) # percentiles to compute
lpc=np.insert(lpc,0,1)
lpc=np.append(lpc,99)
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

troplat=20    # latitudinal bound of tropics

lflag=[
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':True, # only look at tropics
    },
    {
    'landonly':False, # only use land grid points for rsq
    'troponly':False, # only look at tropics
    },
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':False, # only look at tropics
    },
]

# load data
idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn)
_,_,gr=pickle.load(open('%s/d%s_%s_%s.%s.pickle' % (idir,varn,his,fut,se), 'rb'))	

idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)
ivn=pickle.load(open('%s/%s%s_%s_%s.%s.pickle' % (idir,vnpre,varn,his,fut,se), 'rb'))	
ibvn=pickle.load(open('%s/%s%s_%s_%s.%s.pickle' % (idir,bnpre,varn,his,fut,se), 'rb'))	
if len(ivn.shape) > len(ibvn.shape):
    ibvn=np.tile(ibvn,(ibvn.shape[2],1,1,1,1))
    ibvn=np.transpose(ibvn,[1,2,0,3,4])
elif len(ibvn.shape) > len(ivn.shape):
    ivn=np.tile(ivn,(ibvn.shape[2],1,1,1,1))
    ivn=np.transpose(ivn,[1,2,0,3,4])

def binp(flag):
    # load land mask
    lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

    # cos weight
    w=np.cos(np.deg2rad(gr['lat']))
    w=np.tile(w,(ivn.shape[1],ivn.shape[4],1))
    w=np.transpose(w,[0,2,1])

    if flag['troponly']:
        lats=np.transpose(np.tile(gr['lat'].data,(len(gr['lon']),1)),[1,0])
        lm[np.abs(lats)>troplat]=np.nan
    lm=np.tile(lm,(ivn.shape[1],1,1))

    b=np.empty([ivn.shape[0],ivn.shape[2],len(lpc)])
    for imd in tqdm(range(ivn.shape[0])):
        for ip in range(ivn.shape[2]):
            svn=ivn[imd,:,ip,...]
            sbvn=ibvn[imd,:,ip,...]
            sw=w.copy()
            svn=sw*svn # cos weight
            sbvn=sw*sbvn
            if flag['landonly']:
                svn=lm*svn
                sbvn=lm*sbvn
                sw=lm*sw
            svn=svn.flatten()
            sbvn=sbvn.flatten()
            sw=sw.flatten()
            svn=svn[~np.isnan(svn)]
            sbvn=sbvn[~np.isnan(sbvn)]
            sw=sw[~np.isnan(sw)]
            # compute percentiles of sbvn
            psbvn=100/len(sbvn)*rankdata(sbvn,'average')
            # bin
            dg=np.digitize(psbvn,lpc)
            b[imd,ip,:]=[np.nansum(sw[dg==i]*svn[dg==i])/np.nansum(sw[dg==i]) for i in range(len(lpc))]

    odir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)

    oname='%s/b.pct.%s.%s%s_%s_%s.%s' % (odir,bnpre,vnpre,varn,his,fut,se)
    if flag['landonly']:
        oname='%s.land'%oname
    if flag['troponly']:
        oname='%s.trop'%oname

    pickle.dump([b,gr['pct'],lpc], open('%s.pickle'%oname, 'wb'), protocol=5)	

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(binp)(flag) for flag in lflag]
        dask.compute(*tasks,scheduler='processes')
        # dask.compute(*tasks,scheduler='single-threaded')
