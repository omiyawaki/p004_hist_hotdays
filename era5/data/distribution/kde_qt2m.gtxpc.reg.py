import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint as td2q
from scipy.stats import gaussian_kde
from regions import rinfo

# this script aggregates the histogram of daily temperature for a given region on interest

varn='qt2m'
xpc='95'
lre=['swus','sea']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
by=[1950,1970] # year bounds for evaluating climatological xpc th percentile

y0=1950 # first year
y1=2020 # last year+1

lyr=[str(y) for y in np.arange(y0,y1+1)]

for re in lre:
    # file where selected region is provided
    rloc,rlat,rlon=rinfo(re)

    for se in lse:
        idir = '/project/mojave/observations/ERA5_daily'
        odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s' % (se)

        if not os.path.exists(odir):
            os.makedirs(odir)

        # Load climatology
        [ht2m0,lpc0]=pickle.load(open('%s/%s/clmean.%s.%s.%g.%g.%s.pickle' % (odir,'t2m','t2m',re,by[0],by[1],se), 'rb'))
        xpct2m=ht2m0[np.where(np.equal(lpc0,int(xpc)))[0][0]]

        for yr in tqdm(lyr):
            # load temp
            fn = '%s/T2m/t2m_%s.nc' % (idir,yr)
            ds = xr.open_dataset(fn)
            t2m = ds['t2m'].load()*units.kelvin
            # load dew pt temp
            fn = '%s/TD2m/td2m_%s.nc' % (idir,yr)
            ds = xr.open_dataset(fn)
            td2m = ds['td2m'].load()*units.kelvin
            # load surf pressure
            fn = '%s/PS/ps_%s.nc' % (idir,yr)
            ds = xr.open_dataset(fn)
            ps = ds['ps'].load()*units.pascal
            # calculate specific humidity
            q2m = td2q(ps,td2m)
            # gc
            td2m.close()
            ps.close()

            if se != 'ann':
                t2m=t2m.sel(time=t2m['time.season']==se.upper())
                q2m=q2m.sel(time=q2m['time.season']==se.upper())
            gr = {}
            gr['lon'] = ds['lon']
            gr['lat'] = ds['lat']

            if np.logical_or(np.not_equal(gr['lat'],rlat).any(),np.not_equal(gr['lon'],rlon).any()):
                error('Check that selected region is in same grid as data being applied to.')

            rt2m=np.empty([t2m.shape[0],len(rloc[0])])
            rq2m=np.empty([q2m.shape[0],len(rloc[0])])
            for it in range(t2m.shape[0]):
                lt=t2m[it,...].data
                rt2m[it,:]=lt[rloc] # regionally selected data
                lt=q2m[it,...].data
                rq2m[it,:]=lt[rloc] # regionally selected data
            # select data above xpc temp
            ixt2m=np.where(rt2m>xpct2m)
            xt2m=rt2m[ixt2m]
            xq2m=rq2m[ixt2m]
            kqt2m=gaussian_kde(np.vstack([xt2m,xq2m]))

            pickle.dump(kqt2m, open('%s/%s/k%s_%s.gt%s.%s.%s.pickle' % (odir,varn,varn,yr,xpc,re,se), 'wb'), protocol=5)	
