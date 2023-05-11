import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from cmip6util import mods,emem,simu,year
from glade_utils import grid

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

vbin='dmrsos' # variable to bin
vpct='dmrsos' # binning variable to bin according to percentile
varn='%s+%s'%(vbin,vpct) # variable name

realm='atmos' # data realm e.g., atmos, ocean, seaIce, etc
freq='day' # data frequency e.g., day, mon
lse = ['djf','son','mam'] # season (ann, djf, mam, jja, son)

# fo='historical'
# lcl = ['his']
# byr=[1980,2000] # output year bounds

fo='ssp370'
lcl = ['fut']
byr=[2080,2100] # output year bounds

for cl in lcl:
    for se in lse:
        # list of models
        lmd=mods(fo)

        # years and simulation names
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]

            odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

            if not os.path.exists(odir):
                os.makedirs(odir)

            c=0 # counter
            # load data to bin
            idir0='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,vbin)
            fn = '%s/cl%s_%g-%g.%s.nc' % (idir0,vbin,byr[0],byr[1],se)
            ds = xr.open_mfdataset(fn)
            vb = ds[vbin].load()

            # load binning data
            idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,vpct)
            fn = '%s/cl%s_%g-%g.%s.nc' % (idir,vpct,byr[0],byr[1],se)
            ds = xr.open_mfdataset(fn)
            vp = ds[vpct].load()
            # load percentiles
            p,gr=pickle.load(open('%s/p%s_%g-%g.%s.pickle' % (idir,vpct,byr[0],byr[1],se), 'rb'))	

            # initialize array to store binned data
            b = np.empty([p.shape[0]-1, p.shape[1], p.shape[2]])

            # loop through gridir points to compute binned values within percentiles
            for ln in tqdm(range(gr['lon'].size)):
                for la in range(gr['lat'].size):
                    lvb=vb[:,la,ln]
                    lvp=vp[:,la,ln]
                    lp=p[:,la,ln]
                    dg=np.digitize(lvp,lp)
                    b[:,la,ln]=[lvb[dg==i].mean() for i in range(1,len(lp))]
            pickle.dump([b, gr], open('%s/b%s_%g-%g.%s.pickle' % (odir,varn,byr[0],byr[1],se), 'wb'), protocol=5)	
