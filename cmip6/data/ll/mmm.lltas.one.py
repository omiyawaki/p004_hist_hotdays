import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint as td2q
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from regions import rinfo
from glade_utils import grid
from cmip6util import mods,emem,simu,year

# this script aggregates the histogram of daily temperature for a given region on interest

# index for location to make plot
iloc=[110,85] # SEA
# iloc=[135,200] # SWUS

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='2080-2100'

lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

lnt=[90]
lpc=[95]
realm='atmos'
freq='day'
varn1='tas' # y axis var
varn2='thd'# x axis var
varn='%s+%s'%(varn1,varn2)

for nt in lnt:
    for pc in lpc:
        for se in lse:
            # list of models
            lmd=mods(fo)

            odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn)

            if not os.path.exists(odir):
                os.makedirs(odir)

            for imd in tqdm(range(len(lmd))):
                md=lmd[imd]

                idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

                fn = '%s/ll%s%03d_%g-%g.%g.%s.pickle' % (odir,varn,nt,byr[0],byr[1],pc,se)
                [illtas,istdev] = pickle.load(open(fn,'rb'))
                illtas=illtas[:,iloc[0],iloc[1]]
                istdev=istdev[:,iloc[0],iloc[1]]

                if imd==0:
                    elltas=np.empty([len(lmd),illtas.shape[0]])
                    estdev=np.empty([len(lmd),istdev.shape[0]])

                elltas[imd,:]=illtas
                estdev[imd,:]=istdev

                lltas=np.nanmean(elltas,axis=0)
                stdev=np.sqrt(np.nansum(estdev**2,axis=0))

            pickle.dump([lltas,iarr2,stdev,elltas,estdev], open('%s/%s_%s.%g.%g.%s.pickle' % (odir,varn,yr,iloc[0],iloc[1],se), 'wb'), protocol=5)	
