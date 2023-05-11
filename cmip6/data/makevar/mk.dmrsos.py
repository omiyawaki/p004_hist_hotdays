import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
from tqdm import tqdm
from cmip6util import mods,simu,emem
from glade_utils import grid

# collect warmings across the ensembles

varn='dmrsos'
ivar='mrsos'
ty='2d'

lfo = ['historical'] # forcing (e.g., ssp245)
byr=[1980,2000]

# lfo = ['ssp370'] # forcing (e.g., ssp245)
# byr=[2080,2100]

freq='day'
# lse = ['jja'] # season (ann, djf, mam, jja, son)
lse = ['djf','mam','son'] # season (ann, djf, mam, jja, son)

for fo in lfo:
    for se in lse:
        lmd=mods(fo) # create list of ensemble members

        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            ens=emem(md)
            grd=grid(fo,cl,md)

            idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,cl,fo,md)
            odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
            if not os.path.exists(odir):
                os.makedirs(odir)

            # load mrsos data
            fn = '%s/%s/cl%s_%g-%g.%s.nc' % (idir,ivar,ivar,byr[0],byr[1],se)
            ds = xr.open_mfdataset(fn)
            mrsos = ds[ivar].load()

            # load mrsos percentile values (from historical)
            idir0='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,'his','historical',md)
            cmrsos,_ = pickle.load(open('%s/%s/c%s_%g-%g.%s.rg.pickle' % (idir0,ivar,ivar,1980,2000,se),'rb'))

            # take anomaly from mean
            dmrsos=mrsos-cmrsos[0,:,:]

            dmrsos=dmrsos.rename(varn)
            dmrsos.to_netcdf('%s/cl%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se))
