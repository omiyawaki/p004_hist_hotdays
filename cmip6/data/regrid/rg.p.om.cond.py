import os,sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from glade_utils import grid
from util import mods,simu,emem
from dask.distributed import Client
from concurrent.futures import ProcessPoolExecutor as Pool

# ld=np.concatenate((np.arange(20,80+20,20),np.arange(150,850+100,100)))
# lvn=['mrso%g'%d for d in ld] # input1
lvn=['gflx']
ty='2d'
checkexist=False
only95=False

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'
# dyr=10

se='sc'

lmd=mods(fo) # create list of ensemble members
lmd=lmd[6:]

def calc_p(md):
    ens=emem(md)
    grd=grid(md)

    for varn in lvn:
        print(md)
        print(varn)

        idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        tdir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'tas')
        odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        if 'gwl' in byr:
            oname='%s/pc.om.%s_%s.%s.nc' % (odir,varn,byr,se)
        else:
            oname='%s/pc.om.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)

        if checkexist:
            if os.path.isfile(oname):
                print('Output file already exists, skipping...')
                continue

        # load raw data
        if 'gwl' in byr:
            fn='%s/om.%s_%s.%s.nc' % (idir,varn,byr,se)
        else:
            fn='%s/om.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
        print('\n Loading data to composite...')
        ds = xr.open_mfdataset(fn)
        try:
            vn = ds[varn]
        except:
            vn = ds['plh']
        print('\n Done.')

        # select data within time of interest
        if 'gwl' in byr:
            idirg='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % ('ts','historical+%s'%fo,md,'tas')
            [ygwl,gwl]=pickle.load(open('%s/gwl%s.%s.pickle' % (idirg,'tas','ts'),'rb'))
            print(ygwl)
            idx=np.where(gwl==float(byr[-3:]))
            print(idx)
            if ygwl[idx]==1850:
                print('\n %s does not warm to %s K. Skipping...'%(md,byr[-3:]))
                return

            print('\n Selecting data within range of interest...')
            otime=vn['time'].sel(time=vn['time.year']==2080)
            vn=vn.sel(time=vn['time.year']>=ygwl[idx].data-dyr)
            vn=vn.sel(time=vn['time.year']<ygwl[idx].data+dyr)
            print('\n Done.')

        else:
            print('\n Selecting data within range of interest...')
            vn=vn.sel(time=vn['time.year']>=byr[0])
            vn=vn.sel(time=vn['time.year']<byr[1])
            otime=vn['time'].sel(time=vn['time.year']==byr[0])
            print('\n Done.')

        # compute hot days
        # load temp data
        if 'gwl' in byr:
            fn='%s/om.%s_%s.%s.nc' % (tdir,'tas',byr,se)
        else:
            fn='%s/om.%s_%g-%g.%s.nc' % (tdir,'tas',byr[0],byr[1],se)
        print('\n Loading data to composite...')
        ds = xr.open_mfdataset(fn)
        tvn = ds['tas']
        print('\n Done.')

        # load percentile data
        if 'gwl' in byr:
            pvn=xr.open_dataarray('%s/p.om.%s_%s.%s.nc' % (tdir,'tas',byr,se))
        else:
            pvn=xr.open_dataarray('%s/p.om.%s_%g-%g.%s.nc' % (tdir,'tas',byr[0],byr[1],se))

        if only95:
            pct=[95]
            opct=pct
            pvn=pvn.sel(percentile=pct)
        else:
            pct=pvn['percentile']
            fpct=np.insert(pct.data,0,0)
            fpct=np.append(fpct,100)
            opct=1/2*(fpct[1:]+fpct[:-1])
            lmn=pvn['month']

        s=pvn.shape
        gpi=pvn['gpi']

        def bin_below(v,t,p):
            return v.where(t<p).mean('time',skipna=True)

        def bin_betwn(v,t,p1,p2):
            v=v.where(t>=p1)
            return v.where(t<p2).mean('time',skipna=True)

        def bin_above(v,t,p):
            return v.where(t>=p).mean('time',skipna=True)

        # take conditional mean
        def cmean(m,vn,tvn,pvn,s,opct):
            ovn=np.empty([len(opct),s[2]])
            spvn=pvn.sel(month=m) # t(m), pct x gpi
            stvn=tvn.sel(time=tvn['time.month']==m) # dy x gpi
            svn=vn.sel(time=vn['time.month']==m) # dy x gpi
            if only95:
                spvn1=spvn.sel(percentile=pct[0])
                ovn[0,:]=bin_above(svn,stvn,spvn1)
            else:
                for ip,p in enumerate(tqdm(pct)):
                    spvn1=spvn.sel(percentile=pct[ip]) # t(p), gpi
                    if not ip==len(pct)-1: spvn2=spvn.sel(percentile=pct[ip+1]) # t(p), gpi
                    if ip==0:
                        ovn[ip,:]  =bin_below(svn,stvn,spvn1)        # 0 to first p
                        ovn[ip+1,:]=bin_betwn(svn,stvn,spvn1,spvn2)  # p1 to p2
                    elif ip==len(pct)-1:
                        ovn[ip+1,:]=bin_above(svn,stvn,spvn1)        # last p to 100  
                    else:
                        ovn[ip+1,:]=bin_betwn(svn,stvn,spvn1,spvn2)
            return ovn

        with Client(n_workers=len(lmn)):
            tasks=[dask.delayed(cmean)(m,vn,tvn,pvn,s,opct) for m in lmn]
            ovn=dask.compute(*tasks)

        ovn=np.stack(ovn,axis=0)
        ovn=xr.DataArray(ovn,coords={'month':lmn,'percentile':opct,'gpi':gpi},dims=('month','percentile','gpi'))
        print('\n Saving data...')
        ovn=ovn.rename(varn)
        ovn.to_netcdf(oname,format='NETCDF4')
        print('\n Done.')

if __name__=='__main__':
    calc_p('KACE-1-0-G')
    # [calc_p(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_p,lmd)
