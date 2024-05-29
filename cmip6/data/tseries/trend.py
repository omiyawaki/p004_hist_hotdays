import os,sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from scipy.stats import linregress
from tqdm import tqdm
from util import mods

se='ts'
tavg='jja'
varn='tas'
fo1='historical'
fo2='ssp370'
fo='%s+%s'%(fo1,fo2)
selp=True
p=95 # percentile to select
yr='1950-2020'

lmd=mods(fo1)

def regress(yvn,lyr,gpi):
    # regress in time
    shvn = np.empty_like(gpi ,dtype=float) # slope of regression
    ihvn = np.empty_like(shvn,dtype=float) # intercept of regression
    rhvn = np.empty_like(shvn,dtype=float) # r value of regression
    phvn = np.empty_like(shvn,dtype=float) # p value of regression
    eshvn= np.empty_like(shvn,dtype=float) # standard error of slope
    eihvn= np.empty_like(shvn,dtype=float) # standard error of intercept 
    for igpi in range(len(gpi)):
        lrr = linregress(lyr,yvn[:,igpi])
        shvn[igpi] = lrr.slope
        ihvn[igpi] = lrr.intercept
        rhvn[igpi] = lrr.rvalue
        phvn[igpi] = lrr.pvalue
        eshvn[igpi] = lrr.stderr
        eihvn[igpi] = lrr.intercept_stderr

    stats={}
    stats['slope'] = shvn
    stats['intercept'] = ihvn
    stats['rvalue'] = rhvn
    stats['pvalue'] = phvn
    stats['stderr'] = eshvn
    stats['intercept_stderr'] = eihvn

    return stats

def trend(md):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
    odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)

    # load data
    mvn=xr.open_dataarray('%s/m.%s.%s.%s.nc'%(odir,varn,yr,tavg))
    pvn=xr.open_dataarray('%s/pc.%s.%s.%s.nc'%(odir,varn,yr,tavg))
    if selp:
        pvn=pvn.sel(percentile=p)
    else:
        pct=pvn['percentile']
    gpi=pvn['gpi'].data
    lyr=pvn['year'].data

    # regress
    mstats=regress(mvn.data,lyr,gpi)
    if selp:
        pstats=regress(pvn.data,lyr,gpi)
    else:
        pstats=[regress(pvn.sel(percentile=p).data,lyr,gpi) for p in tqdm(pct)]

    pickle.dump(mstats, open('%s/trend.m.%s.%s.%s.pickle' %      (odir,varn,yr,tavg), 'wb'), protocol=5)	
    if selp:
        pickle.dump(pstats, open('%s/trend.pc.%02d.%s.%s.%s.pickle' %(odir,p,varn,yr,tavg), 'wb'), protocol=5)	
    else:
        pickle.dump(pstats, open('%s/trend.pc.%s.%s.%s.pickle' %(odir,varn,yr,tavg), 'wb'), protocol=5)	

# trend('CESM2')
[trend(md) for md in tqdm(lmd)]
