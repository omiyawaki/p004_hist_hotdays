import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm
from utils import corr

# spatial correlation between model and obs

se='ts'
fo='lens2'
yr='1950-2020'
varn='tas'
pc=[1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # percentiles to compute
p=95
ipc=pc.index(p)

tavg='summer'

# lat values in gpi
lat,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lmilatlon.pickle','rb'))

def selhemi(jja,djf,lat):
    hemi=np.copy(jja)
    if tavg=='summer':
        hemi[lat>0]=jja[lat>0]
        hemi[lat<=0]=djf[lat<=0]
    elif tavg=='winter':
        hemi[lat>=0]=djf[lat>0]
        hemi[lat<0]=jja[lat<0]
    return hemi

# load reanalysis
def load_era(seas):
    idir = '/project/amp02/miyawaki/data/p004/era5/%s/%s'%(se,varn)
    me=pickle.load(open('%s/trend.m.%s.%s.%s.pickle' %      (idir,varn,yr,seas), 'rb'))['slope']
    pe=[pickle.load(open('%s/trend.pc.%s.%s.%s.pickle' %      (idir,varn,yr,seas), 'rb'))[i]['slope'] for i in range(len(pc))]
    return me,pe

mej,pej=load_era('jja')
med,ped=load_era('djf')
me=selhemi(mej,med,lat)
pe=[selhemi(pej[i],ped[i],lat) for i in range(len(pc))]
de=[pei-me for pei in pe]

def load_cesm(md,seas):
    idir = '/project/amp02/miyawaki/data/p004/cesm2-le/%s/%s/%s/%s'%(se,fo,md,varn)
    me=pickle.load(open('%s/trend.m.%s.%s.%s.pickle' % (idir,varn,yr,seas), 'rb'))['slope']
    pe=[pickle.load(open('%s/trend.pc.%s.%s.%s.pickle' % (idir,varn,yr,seas), 'rb'))[i]['slope'] for i in range(len(pc))]
    return me,pe

    return me,pe

def rmr(md):
    odir='/project/amp02/miyawaki/data/p004/cesm2-le/%s/%s/%s/%s'%(se,fo,md,varn)

    mmj,pmj=load_cesm(md,'jja')
    mmd,pmd=load_cesm(md,'djf')
    mm=selhemi(mmj,mmd,lat)
    pm=[selhemi(pmj[i],pmd[i],lat) for i in range(len(pc))]
    dm=[pmi-mm for pmi in pm]

    r=[corr(de[i],dm[i],0) for i in range(len(pc))]
    print(md,r[-3])
    pickle.dump(r,open('%s/sp.corr.era5.%s.%s.%s.pickle' % (odir,varn,yr,tavg), 'wb'),protocol=5)

lens=['%03d'%i for i in np.arange(1,101,1)]
[rmr('CESM2.%s'%ens) for ens in tqdm(lens)]
