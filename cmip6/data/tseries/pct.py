import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import numpy as np
import xarray as xr
from tqdm import tqdm
from util import mods

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

se='ts'
tavg='jja'
varn='tas'
yr='1950-2020'
fo1='historical'
fo2='ssp370'
fo='%s+%s'%(fo1,fo2)
pc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # percentiles to compute

lmd=mods(fo1)

def calc_pct(md):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
    odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    fn='%s/lm.%s.%s.%s.nc' % (idir,varn,yr,se)
    vn=xr.open_dataset(fn)[varn]
    lyr=list(set(vn['time.year'].data))
    gpi=vn['gpi'].data
    if tavg != 'ann':
        vn=vn.sel(time=vn['time.season']==tavg.upper())

    mvn=np.empty([len(lyr),len(gpi)])
    pvn=np.empty([len(lyr),len(pc),len(gpi)])
    # loop through year and gridpoints and calculate mean and percentiles
    for iyr,syr in enumerate(tqdm(lyr)):
        svn=vn.sel(time=vn['time.year']==syr).data
        mvn[iyr,:]=np.mean(svn,axis=0)
        pvn[iyr,...]=np.percentile(svn,pc,axis=0)

    mvn=xr.DataArray(mvn,coords={'year':lyr,'gpi':gpi},dims=('year','gpi'))
    pvn=xr.DataArray(pvn,coords={'year':lyr,'percentile':pc,'gpi':gpi},dims=('year','percentile','gpi'))
    mvn=mvn.rename(varn)
    pvn=pvn.rename(varn)
    mvn.to_netcdf('%s/m.%s.%s.%s.nc'% (odir,varn,yr,tavg))
    pvn.to_netcdf('%s/pc.%s.%s.%s.nc'%(odir,varn,yr,tavg))

# calc_pct('CESM2')
[calc_pct(md) for md in tqdm(lmd)]
