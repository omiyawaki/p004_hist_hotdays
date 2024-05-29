import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid

# comdsect warmings across the ensembles

# lvn=['zg850'] # input1
# mycmip=False

lvn=['gradt_mon850']
mycmip=True

ty='2d'
checkexist=False

# fo='historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo='ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'
dyr=10

se='sc'

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

lmd=mods(fo) # create list of ensemble members

def calc_lm(md):
    print(md)
    ens=emem(md)
    grd=grid(md)

    for varn in lvn:
        print(varn)

        odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        if 'gwl' in byr:
            ifn='%s/mon.%s_%s.%s.nc' % (odir,varn,byr,se)
            ofn='%s/lm.%s_%s.%s.nc' % (odir,varn,byr,se)
        else:
            ifn='%s/mon.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
            ofn='%s/lm.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)

        if checkexist:
            if os.path.isfile():
                print('Output file already exists, skipping...')
                continue

        # load mon data
        print('\n Loading data...')
        ds=xr.open_dataset(ifn)
        print('\n Done.')

        # save grid info
        gr={}
        gr['lon']=ds['lon']
        gr['lat']=ds['lat']

        # regrid data
        if md!='CESM2':
            print('\n Regridding...')
            # path to weight file
            wf='%s/wgt.cmip6.%s.%s.cesm2.nc'%(rgdir,md,ty)
            # build regridder with existing weights
            rgd=xe.Regridder(ds,ogr, 'bilinear', periodic=True, reuse_weights=True, filename=wf)
            ds=rgd(ds)
            print('\n Done.')

        # remove ocean points
        mon=ds['month']
        def rmocn(vn):
            vn=vn.data
            vn=np.reshape(vn,(vn.shape[0],vn.shape[1]*vn.shape[2]))
            return np.delete(vn,omi,axis=1)
        dx=rmocn(ds['dx'])
        dy=rmocn(ds['dy'])
        ngp=dx.shape[1]

        ds=xr.Dataset(
                    data_vars={'dx':(['month','gpi'],dx),'dy':(['month','gpi'],dy)},
                    coords={'month':mon,'gpi':np.arange(ngp)}
                    )

        ds.to_netcdf(ofn,format='NETCDF4')

# calc_lm('UKESM1-0-LL')
[calc_lm(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_lm,lmd)
