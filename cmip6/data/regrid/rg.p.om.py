import os,sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from glade_utils import grid
from util import mods,simu,emem
from concurrent.futures import ProcessPoolExecutor as Pool

lmn=1+np.arange(12)
lpc=np.arange(5,95+5,5)
varn='tas' # input1
ty='2d'
checkexist=False

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'
dyr=10

freq='day'
se='sc'

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

lmd=mods(fo) # create list of ensemble members

def load_raw(odir,varn,byr,se):
    # load raw data
    if 'gwl' in byr:
        fn='%s/om.%s_%s.%s.nc' % (odir,varn,byr,se)
    else:
        fn='%s/om.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
    print('\n Loading data to composite...')
    ds = xr.open_dataset(fn)
    print('\n Done.')
    return ds

def calc_pvn(md):
    ens=emem(md)
    grd=grid(md)

    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if 'gwl' in byr:
        oname='%s/p.om.%s_%s.%s.nc' % (odir,varn,byr,se)
    else:
        oname='%s/p.om.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)

    if checkexist:
        if os.path.isfile(oname):
            print('Output file already exists, skipping...')
            return

    ds=load_raw(odir,varn,byr,se)
    vn=ds[varn]

    if 'gwl' in byr:
        idirg='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % ('ts','historical+%s'%fo,md,'tas')
        [ygwl,gwl]=pickle.load(open('%s/gwl%s.%s.pickle' % (idirg,'tas','ts'),'rb'))
        idx=np.where(gwl==float(byr[-3:]))
        if ygwl[idx]==1850:
            print('\n %s does not warm to %s K. Skipping...'%(md,byr[-3:]))
            return

        print('\n Selecting data within range of interest...')
        otime=vn['time'].sel(time=vn['time.year']==2080)
        vn=vn.sel(time=vn['time.year']>=ygwl[idx].data-dyr)
        vn=vn.sel(time=vn['time.year']<ygwl[idx].data+dyr)
        print('\n Done.')

    else:
        # select data within time of interest
        print('\n Selecting data within range of interest...')
        vn=vn.sel(time=vn['time.year']>=byr[0])
        vn=vn.sel(time=vn['time.year']<byr[1])
        otime=vn['time'].sel(time=vn['time.year']==byr[0])
        print('\n Done.')

    time=vn['time']
    ngp=vn.shape[1]
    print(ngp)

    ovn=np.empty([len(lmn),len(lpc),vn.shape[1]])
    # compute monthly climatology
    print('\n Computing percentiles...')

    for i,mn in enumerate(tqdm(lmn)):
        svn=vn.sel(time=vn['time.month']==mn)
        ovn[i,...]=np.nanpercentile(svn.data,lpc,axis=0)

    ovn=xr.DataArray(ovn,coords={'month':lmn,'percentile':lpc,'gpi':np.arange(ngp)},dims=('month','percentile','gpi'))

    ovn.to_netcdf(oname,format='NETCDF4')

# calc_pvn('CESM2')
[calc_pvn(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_pvn,lmd)
