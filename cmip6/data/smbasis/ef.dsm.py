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

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='2080-2100'

lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

realm='atmos'
freq='day'
varn1='ef' # y axis var
varn2='dmrsos'# x axis var
arr1=np.linspace(0,5,20)
arr2=np.linspace(-30,20,20)
msm,mef=np.mgrid[-30:20:20j,0:5:20j]
abm=np.vstack([msm.ravel(),mef.ravel()])
varn='%s+%s'%(varn1,varn2)

for se in lse:
    # list of models
    lmd=mods(fo)

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]

        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        fn = '%s/k%s_%s.%s.pickle' % (idir,varn,yr,se)
        kde = pickle.load(open(fn,'rb'))

        # create list to store BCs and stdev
        ef = np.empty([len(arr2),len(kde),len(kde[0])])
        stdev = np.empty([len(arr2),len(kde),len(kde[0])])

        for ilo in tqdm(range(len(kde[0]))):
            for ila in range(len(kde)):
                if kde[ila][ilo]==0:
                    pass
                else:
                    pdf=np.reshape(kde[ila][ilo](abm).T,msm.shape)
                    lef=np.trapz(pdf*mef,x=arr1,axis=1)/np.trapz(pdf,x=arr1,axis=1)
                    ef[:,ila,ilo]=lef
                    stdev[:,ila,ilo]=np.sqrt(np.trapz(pdf*(mef-lef[:,None])**2,x=arr1,axis=1)/np.trapz(pdf,x=arr1,axis=1))

        if not os.path.exists(odir):
            os.makedirs(odir)

        pickle.dump([ef,arr2,stdev], open('%s/%s_%s.%s.pickle' % (odir,varn,yr,se), 'wb'), protocol=5)	
