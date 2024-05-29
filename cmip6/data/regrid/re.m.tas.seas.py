import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor as Pool
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from regions import masklev0,masklev1,settype,retname,regionsets

# collect warmings across the ensembles

q=1
varn='tas'
ty='2d'
se='ts'
relb='tr_lnd'
tlat=30
seas='summer'

fo1='historical' # forcing (e.g., ssp245)

fo2='ssp370' # forcing (e.g., ssp245)

fo='%s+%s'%(fo1,fo2)

freq='day'

lmd=mods(fo1) # create list of ensemble members

# land/ocean mask
lm,om=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

def selmon(se):
    if se=='summer':
        nhmon=[6,7,8]
        shmon=[12,1,2]
    elif se=='winter':
        nhmon=[12,1,2]
        shmon=[6,7,8]
    return nhmon,shmon

def calc_re(md):
    ens=emem(md)
    grd=grid(md)
    if not relb=='tr_lnd':
        mtype=settype(relb)
        retn=retname(relb)
        re=regionsets(relb)

    idir1='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo1,freq,varn,md,ens,grd)
    idir2='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo2,freq,varn,md,ens,grd)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # historical temp
    fn1 = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir1,varn,freq,md,fo1,ens,grd)
    tas1 = xr.open_mfdataset(fn1)[varn]

    # sel seas
    def selseas(xvn):
        nhmon,shmon=selmon(seas)
        xvnnh=xvn.sel(time=xvn['time.month'].isin(nhmon))
        xvnsh=xvn.sel(time=xvn['time.month'].isin(shmon))
        mvnnh=xvnnh.groupby('time.year').quantile(0.5,'time')
        mvnsh=xvnsh.groupby('time.year').quantile(0.5,'time')
        pvnnh=xvnnh.groupby('time.year').quantile(q,'time')
        pvnsh=xvnsh.groupby('time.year').quantile(q,'time')
        ctime=np.intersect1d(mvnnh['year'].data,mvnsh['year'].data)
        mvnnh=mvnnh.sel(year=ctime)
        mvnsh=mvnsh.sel(year=ctime)
        pvnnh=pvnnh.sel(year=ctime)
        pvnsh=pvnsh.sel(year=ctime)
        mvn=mvnsh.copy()
        mvn.data[:,mvn['lat']>0,:]=mvnnh.data[:,mvnnh['lat']>0,:]
        pvn=pvnsh.copy()
        pvn.data[:,pvn['lat']>0,:]=pvnnh.data[:,pvnnh['lat']>0,:]
        return mvn,pvn

    mtas1,ptas1=selseas(tas1)

    # mask gridpoints outside region of interest
    if relb=='us':
        mask=masklev0(re,mtas1,mtype).data
    elif relb=='tr_lnd':
        mask=np.ones_like(mtas1.data)
        mask[:,np.abs(mtas1['lat'])>tlat,:]=np.nan
        mask=mask[0,...]
    else:
        mask=masklev1(None,mtas1,re,mtype).data

    mtas1.data=mask*mtas1.data
    ptas1.data=mask*ptas1.data

    # create area weights
    cosw=np.cos(np.deg2rad(mtas1['lat']))
    # take mean
    mtas1=mtas1.weighted(cosw).mean(('lon','lat'))
    ptas1=ptas1.weighted(cosw).mean(('lon','lat'))

    # future temp
    fn2 = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir2,varn,freq,md,fo2,ens,grd)
    tas2 = xr.open_mfdataset(fn2)[varn]
    mtas2,ptas2=selseas(tas2)

    mtas2.data=mask*mtas2.data
    ptas2.data=mask*ptas2.data
    # create area weights
    cosw=np.cos(np.deg2rad(mtas2['lat']))
    # take global mean
    mtas2=mtas2.weighted(cosw).mean(('lon','lat'))
    ptas2=ptas2.weighted(cosw).mean(('lon','lat'))
    print(mtas2.shape)

    # merge timeseries
    mtas=xr.concat([mtas1,mtas2],dim='year')
    ptas=xr.concat([ptas1,ptas2],dim='year')
    print(mtas.shape)
    print(ptas.shape)

    mtas=mtas.rename('m%s'%varn)
    ptas=ptas.rename('p%s'%varn)
    otas=xr.merge([mtas,ptas])
    otas.to_netcdf('%s/re.%s.%s.%s.%s.nc' % (odir,varn,se,relb,seas))

# calc_re('ACCESS-CM2')
# [calc_re(md) for md in tqdm(lmd)]

if __name__=='__main__':
    with Pool(max_workers=len(lmd)) as p:
        p.map(calc_re,lmd)

