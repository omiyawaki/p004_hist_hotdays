import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import pwlf
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from tqdm import tqdm
from util import mods
from utils import corr,corr2d,monname
from regions import pointlocs
from CASutils import shapefile_utils as shp
from glade_utils import smile

def main():
    relb='fourcorners'
    re=['Utah','Colorado','Arizona','New Mexico']

    pct=np.linspace(1,99,101)
    varn='tas'
    ise='sc'
    ose='jja'
    lmo=[6,7,8]

    fo1='historical' # forcings 
    yr1='1980-2000'

    fo2='ssp370' # forcings 
    yr2='2080-2100'

    fo='%s+%s'%(fo1,fo2)
    fod='%s-%s'%(fo2,fo1)

    md1='CanESM5'
    md2='MPI-ESM1-2-LR'
    ne=20
    lme1=smile(md1,fo1,varn)[:ne] # first 20 ensembles
    lme2=smile(md2,fo1,varn)[:ne] # first 20 ensembles
    md='%s+%s'%(md1,md2)

    # load ocean indices
    _,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

    # grid
    rgdir='/project/amp/miyawaki/data/share/regrid'
    # open CESM data to get output grid
    cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
    cdat=xr.open_dataset(cfil)
    gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

    # mask
    shpfile = "/project/cas/islas/shapefiles/usa/gadm36_USA_1.shp"
    mask=shp.maskgen(shpfile,gr,re).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)
    print(mask.shape)

    odir1= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (ose,fo1,md,varn,relb)
    if not os.path.exists(odir1):
        os.makedirs(odir1)
    odir2= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (ose,fo2,md,varn,relb)
    if not os.path.exists(odir2):
        os.makedirs(odir2)
    odir3= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (ose,fo,md,varn,relb)
    if not os.path.exists(odir3):
        os.makedirs(odir3)
    odir4= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (ose,fod,md,varn,relb)
    if not os.path.exists(odir4):
        os.makedirs(odir4)

    def calc_pct(md,me,fo,yr):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo,md,varn)

        ds=xr.open_dataset('%s/lm.%s_%s.%s.%s.nc' % (idir,varn,yr,me,ise),engine='h5netcdf')
        vn=ds[varn]
        gpi=ds['gpi']

        # extract data for season
        vn=vn.sel(time=vn['time.season']==ose.upper())

        # delete data outside masked region
        vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
        vn=vn.flatten()

        # compute percentiles
        return np.percentile(vn,pct)

    # lpvn1=[calc_pct(md1,me) for me in tqdm(lme1)]
    with ProgressBar():
        tasks=[dask.delayed(calc_pct)(md1,me,fo1,yr1) for me in lme1]
        lpvn1=dask.compute(*tasks,scheduler='processes')
    with ProgressBar():
        tasks=[dask.delayed(calc_pct)(md1,me,fo2,yr2) for me in lme1]
        lpvn2=dask.compute(*tasks,scheduler='processes')
    with ProgressBar():
        tasks=[dask.delayed(calc_pct)(md2,me,fo1,yr1) for me in lme2]
        lpvn3=dask.compute(*tasks,scheduler='processes')
    with ProgressBar():
        tasks=[dask.delayed(calc_pct)(md2,me,fo2,yr2) for me in lme2]
        lpvn4=dask.compute(*tasks,scheduler='processes')

    lpvn1=np.stack(lpvn1,axis=0)
    lpvn2=np.stack(lpvn2,axis=0)
    lpvn3=np.stack(lpvn3,axis=0)
    lpvn4=np.stack(lpvn4,axis=0)
    ldvn21=lpvn2-lpvn1
    ldvn43=lpvn4-lpvn3
    mpvn1=np.nanmean(lpvn1,axis=0)
    mpvn2=np.nanmean(lpvn2,axis=0)
    mpvn3=np.nanmean(lpvn3,axis=0)
    mpvn4=np.nanmean(lpvn4,axis=0)
    mdvn43=np.nanmean(ldvn43,axis=0)
    mdvn21=np.nanmean(ldvn21,axis=0)
    cpvn1=1.96*np.nanstd(lpvn1,axis=0)/len(lme1)
    cpvn2=1.96*np.nanstd(lpvn2,axis=0)/len(lme1)
    cpvn3=1.96*np.nanstd(lpvn3,axis=0)/len(lme2)
    cpvn4=1.96*np.nanstd(lpvn4,axis=0)/len(lme2)
    cdvn43=1.96*np.nanstd(ldvn43,axis=0)/len(lme1)
    cdvn21=1.96*np.nanstd(ldvn21,axis=0)/len(lme1)

    tname=r'%s' % (relb)
    oname1='%s/p.%s_%s_%s.em.%s' % (odir4,varn,yr1,yr2,ose)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    # ax.fill_between(pct,mdvn21-cdvn21,mdvn21+cdvn21,color='k',alpha=0.2)
    # ax.fill_between(pct,mdvn43-cdvn43,mdvn43+cdvn43,color='k',alpha=0.2)
    for i in range(ne):
        ax.plot(pct,ldvn21[i],linewidth=0.5,color='k')
        ax.plot(pct,ldvn43[i],':',linewidth=0.5,color='k')
    clf=ax.plot(pct,mdvn21,linewidth=1,color='tab:red',label=md1)
    clf=ax.plot(pct,mdvn43,':',linewidth=1,color='tab:red',label=md2)
    ax.set_title(r'%s' % (tname),fontsize=16)
    ax.set_xlabel('$T$ Percentile')
    ax.set_ylabel('$\Delta T$ (K)')
    ax.legend(frameon=False)
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

    tname=r'%s' % (relb)
    oname1='%s/p.%s_%s_%s.em.%s' % (odir3,varn,yr1,yr2,ose)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    clf=ax.plot(pct,mpvn1,color='tab:blue',label=md1)
    clf=ax.plot(pct,mpvn3,':',color='tab:blue',label=md2)
    clf=ax.plot(pct,mpvn2,color='tab:orange')
    clf=ax.plot(pct,mpvn4,':',color='tab:orange')
    ax.set_title(r'%s' % (tname),fontsize=16)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('$T$ (K)')
    ax.legend(frameon=False)
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

    tname=r'%s' % (relb)
    oname1='%s/p.%s_%s.em.%s' % (odir1,varn,yr1,ose)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    clf=ax.plot(pct,mpvn1,color='tab:blue',label=md1)
    clf=ax.plot(pct,mpvn3,':',color='tab:blue',label=md2)
    ax.set_title(r'%s' % (tname),fontsize=16)
    ax.set_xlabel('$T$ (K)')
    ax.set_ylabel('Kernel density (unitless)')
    ax.legend(frameon=False)
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

    tname=r'%s' % (relb)
    oname1='%s/p.%s_%s.em.%s' % (odir2,varn,yr2,ose)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    clf=ax.plot(pct,mpvn2,color='tab:orange',label=md1)
    clf=ax.plot(pct,mpvn4,':',color='tab:orange',label=md2)
    ax.set_title(r'%s' % (tname),fontsize=16)
    ax.set_xlabel('$T$ (K)')
    ax.set_ylabel('Kernel density (unitless)')
    ax.legend(frameon=False)
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

if __name__ == '__main__':
     main()

