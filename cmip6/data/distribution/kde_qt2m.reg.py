import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/project2/tas1/miyawaki/common')
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

realm='atmos'
freq='day'
varn='qt2m'
lre=['swus','sea']
lfo=['historical']
lcl=['his']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

for re in lre:
    # file where selected region is provided
    rloc,rlat,rlon=rinfo(re)

    for se in lse:
        for fo in lfo:
            for cl in lcl:
                # list of models
                lmd=mods(fo)

                for imd in tqdm(range(len(lmd))):
                    md=lmd[imd]
                    ens=emem(md)
                    sim=simu(fo,cl)
                    grd=grid(fo,cl,md)
                    if sim=='ssp245':
                        lyr=['208001-210012']
                    elif sim=='historical':
                        lyr=['198001-200012']

                    idir='/project2/tas1/miyawaki/projects/000_hotdays/data/raw/%s/%s' % (sim,md)
                    odir='/project2/tas1/miyawaki/projects/000_hotdays/data/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

                    if not os.path.exists(odir):
                        os.makedirs(odir)

                    c=0 # counter
                    for yr in lyr:
                        if se=='ann':
                            fnt = '%s/%s_%s_%s_%s_%s_%s_%s.nc' % (idir,'tas',freq,md,sim,ens,grd,yr)
                            fnq = '%s/%s_%s_%s_%s_%s_%s_%s.nc' % (idir,'huss',freq,md,sim,ens,grd,yr)
                        else:
                            fnt = '%s/%s_%s_%s_%s_%s_%s_%s.%s.nc' % (idir,'tas',freq,md,sim,ens,grd,yr,se)
                            fnq = '%s/%s_%s_%s_%s_%s_%s_%s.%s.nc' % (idir,'huss',freq,md,sim,ens,grd,yr,se)

                        dst = xr.open_dataset(fnt)
                        dsq = xr.open_dataset(fnq)
                        if c==0:
                            t2m = dst['tas']
                            q2m = dsq['huss']
                        else:
                            t2m = xr.concat((t2m,dst['tas']),'time')
                            q2m = xr.concat((q2m,dst['huss']),'time')
                        c=c+1
                        
                    # save grid info
                    gr = {}
                    gr['lon'] = dst['lon']
                    gr['lat'] = dst['lat']
                    dst.close()
                    t2m.load()
                    dsq.close()
                    q2m.load()

                    rt2m=np.empty([t2m.shape[0],len(rloc[0])])
                    rq2m=np.empty([q2m.shape[0],len(rloc[0])])
                    for it in tqdm(range(t2m.shape[0])):
                        lt=t2m[it,...].data
                        lq=q2m[it,...].data
                        if len(gr['lon'])!=len(rlon):
                            fint=interp1d(gr['lon'],lt,axis=1,fill_value='extrapolate')
                            lt=fint(rlon)
                            fint=interp1d(gr['lon'],lq,axis=1,fill_value='extrapolate')
                            lq=fint(rlon)
                        if len(gr['lat'])!=len(rlat):
                            fint=interp1d(gr['lat'],lt,axis=0,fill_value='extrapolate')
                            lt=fint(rlat)
                            fint=interp1d(gr['lat'],lq,axis=0,fill_value='extrapolate')
                            lq=fint(rlat)
                        rt2m[it,:]=lt[rloc] # regionally selected data
                        rq2m[it,:]=lq[rloc] # regionally selected data

                    rt2m=rt2m.flatten()
                    rq2m=rq2m.flatten()
                    kqt2m=gaussian_kde(np.vstack([rt2m,rq2m]))

                    pickle.dump(kqt2m, open('%s/k%s_%s.%s.%s.pickle' % (odir,varn,yr,re,se), 'wb'), protocol=5)	
