import os,sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
sys.path.append('../')
from util import mods,simu,emem
sys.path.append('/home/miyawaki/scripts/common')
from gflx import bestfit
from concurrent.futures import ProcessPoolExecutor as Pool

se='sc'

# fo='historical'
# yr='1980-2000'

fo='ssp370'
yr='gwl2.0'

vn1='tas'
vn2='gflx'
vn='%s+%s'%(vn1,vn2)

def rgr(md):
    # load data
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s'%(se,fo,md)
    pv1=xr.open_dataarray('%s/%s/p.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se))
    pv2=xr.open_dataarray('%s/%s/pc.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se))
    # v1=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se))
    # v2=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se))
    gpi=pv1['gpi']

    # calculate piecewise regression
    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    x1,x2,x3=np.empty([12,len(gpi)]),np.empty([12,len(gpi)]),np.empty([12,len(gpi)])
    y1,y2,y3=np.empty([12,len(gpi)]),np.empty([12,len(gpi)]),np.empty([12,len(gpi)])
    m=np.arange(1,13,1)
    for i,mn in enumerate(m):
        sv1,sv2=(pv1.sel(month=mn).data, pv2.sel(month=mn).data)
        sv2=1/2*(sv2[1:,:]+sv2[:-1,:])
        for igpi in tqdm(gpi):
            ssv1,ssv2=sv1[:,igpi],sv2[:,igpi]
            try:
                nmsk=np.logical_or(np.isnan(ssv1),np.isnan(ssv2))
                X,Y=bestfit(ssv2,ssv1)['line']
                x1[i,igpi],x2[i,igpi],x3[i,igpi]=X
                y1[i,igpi],y2[i,igpi],y3[i,igpi]=Y
            except:
                x1[i,igpi],x2[i,igpi],x3[i,igpi]=np.nan,np.nan,np.nan
                y1[i,igpi],y2[i,igpi],y3[i,igpi]=np.nan,np.nan,np.nan

    def savenc(a,s):
        a=xr.DataArray(a,coords={'month':m,'gpi':gpi},dims=('month','gpi'))
        a=a.rename('%s regressed on %s, %s'%(vn1,vn2,s))
        a.to_netcdf('%s/pw.rgr.%s.%s.nc'%(odir,vn,s))

    savenc(x1,'x1')
    savenc(x2,'x2')
    savenc(x3,'x3')
    savenc(y1,'y1')
    savenc(y2,'y2')
    savenc(y3,'y3')

lmd=mods(fo) # create list of ensemble members
# [rgr(md) for md in tqdm(lmd)]
rgr('KACE-1-0-G')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(rgr,lmd)
