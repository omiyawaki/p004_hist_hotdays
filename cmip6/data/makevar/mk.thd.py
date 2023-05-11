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

varn='thd'
ivar='tas'
ty='2d'

lpc=[95,99]

# lfo = ['historical'] # forcing (e.g., ssp245)
# byr=[1980,2000]

lfo = ['ssp370'] # forcing (e.g., ssp245)
byr=[2080,2100]

freq='day'
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['djf','mam','son'] # season (ann, djf, mam, jja, son)

for pc in lpc:
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

                # load tas data
                fn = '%s/%s/cl%s_%g-%g.%s.nc' % (idir,ivar,ivar,byr[0],byr[1],se)
                ds = xr.open_mfdataset(fn)
                tas = ds[ivar].load()
                # load temp pct
                ptas,_ = pickle.load(open('%s/%s/ht2m_%g-%g.%s.pickle' % (idir,ivar,byr[0],byr[1],se),'rb'))
                if pc==95:
                    ptas=ptas[-2,...]
                elif pc==99:
                    ptas=ptas[-1,...]

                # identify timestamp of hot days
                thd=[([0] * tas.shape[2]) for ila in range(tas.shape[1])]
                for ilo in tqdm(range(tas.shape[2])):
                    for ila in range(tas.shape[1]):
                        ltas=tas[:,ila,ilo]
                        lptas=ptas[ila,ilo]
                        ihd=np.nonzero(ltas.data>lptas.data)
                        thd[ila][ilo]=ltas[ihd]['time']

                pickle.dump(thd,open('%s/%s_%g-%g.%g.%s.pickle' % (odir,varn,byr[0],byr[1],pc,se),'wb'),protocol=5)
