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
xpc='95'
lre=['swus']
lfo=['ssp245']
lcl=['fut']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
byr_his=[1980,2000] # year bounds for evaluating climatological xpc th percentile
byr_fut=[2080,2100] # year bounds for evaluating climatological xpc th percentile

for re in lre:
    # file where selected region is provided
    rloc,rlat,rlon=rinfo(re)

    for se in lse:
        for fo in lfo:
            for cl in lcl:
                # list of models
                lmd=mods(fo)

                # years and sim names
                if cl == 'fut':
                    byr=byr_fut
                elif cl == 'his':
                    byr=byr_his

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

                    # Load climatology
                    if sim=='historical':
                        [ht2m0,lpc0]=pickle.load(open('%s/../%s/h%s_%g-%g.%s.%s.pickle' % (odir,'tas','t2m',byr_his[0],byr_his[1],re,se), 'rb'))
                        xpct2m=ht2m0[np.where(np.equal(lpc0,int(xpc)))[0][0]]
                    elif sim=='ssp245':
                        [ht2m1,lpc1]=pickle.load(open('%s/../%s/h%s_%g-%g.%s.%s.pickle' % (odir,'tas','t2m',byr_fut[0],byr_fut[1],re,se), 'rb'))
                        xpct2m=ht2m1[np.where(np.equal(lpc1,int(xpc)))[0][0]]

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

                    # select data above xpc temp
                    ixt2m=np.where(rt2m>xpct2m)
                    xt2m=rt2m[ixt2m]
                    xq2m=rq2m[ixt2m]
                    kqt2m=gaussian_kde(np.vstack([xt2m,xq2m]))

                    pickle.dump(kqt2m, open('%s/k%s_%s.gt%s.%s.%s.pickle' % (odir,varn,yr,xpc,re,se), 'wb'), protocol=5)	
