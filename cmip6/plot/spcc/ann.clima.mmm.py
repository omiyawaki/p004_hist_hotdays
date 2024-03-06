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
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LatitudeFormatter
from scipy.stats import linregress
from scipy.stats import gaussian_kde
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb

nt=7 # window size in days
p=95
npdf=int(1e3) # number of points to evaluate pdf
tlat=30 # latitude separating tropics and ML
plat=50 # latitude separating ML and poles
lvn=['annai']
se = 'sc' # season (ann, djf, mam, jja, son)
fo='historical' # forcings 
byr='1980-2000'
# fo='ssp370' # forcings 
# byr='gwl2.0'
dpi=600
skip507599=True

md='mmm'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def vmax(vn):
    lvm={   'tas':  [-4,4,0.25,'RdBu_r'],
            'hfls': [-30,30,2,'RdBu_r'],
            'ooai': [0,5,0.2,'BrBG_r'],
            'annai': [0,5,0.2,'BrBG_r'],
            }
    return lvm[vn]

def plot(vn):
    vmn,vmx,dvm,cmap=vmax(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # clima
    cvn=xr.open_dataarray('%s/m.%s_%s.%s.nc' % (idir,vn,byr,se))

    # annual mean
    cvn=cvn.mean(dim='month')

    # remap to lat x lon
    llcvn=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llcvn[lmi]=cvn.data
    llcvn=np.reshape(llcvn,(gr['lat'].size,gr['lon'].size))

    # distributions by latitude
    cvn_tr=llcvn[np.abs(gr['lat'])<tlat,:].flatten()
    cvn_hl=llcvn[np.abs(gr['lat'])>plat,:].flatten()
    cvn_ml=llcvn[np.logical_and(np.abs(gr['lat'])<plat,np.abs(gr['lat']>tlat)),:].flatten()
    cvn_tr=cvn_tr[~np.isnan(cvn_tr)]
    cvn_ml=cvn_ml[~np.isnan(cvn_ml)]
    cvn_hl=cvn_hl[~np.isnan(cvn_hl)]
    lcvn=np.linspace(np.nanmin(cvn),np.nanmax(cvn),npdf)
    kde_tr=gaussian_kde(cvn_tr)
    kde_ml=gaussian_kde(cvn_ml)
    kde_hl=gaussian_kde(cvn_hl)
    pdf_tr=kde_tr(lcvn)
    pdf_ml=kde_ml(lcvn)
    pdf_hl=kde_hl(lcvn)

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    llcvn = np.append(llcvn, llcvn[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot clima
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llcvn, np.arange(vmn,vmx+dvm,dvm),extend='both', vmax=vmx, vmin=-vmn, transform=ccrs.PlateCarree(), cmap=cmap)
    ax.coastlines()
    # ax.set_title(r'%s %s ANN' % (md.upper(),fo.upper()),fontsize=16)
    ax.set_title(r'Annual mean',fontsize=16)
    gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
    gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
    gl.yformatter=LatitudeFormatter()
    gl.xlines=False
    gl.left_labels=False
    gl.bottom_labels=False
    gl.right_labels=True
    gl.top_labels=False
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$%s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/ann.%s.%s.png' % (odir,vn,fo), format='png', dpi=dpi)
    fig.savefig('%s/ann.%s.%s.pdf' % (odir,vn,fo), format='pdf', dpi=dpi)

    # plot distributions of AI
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    clf=ax.plot(lcvn,pdf_tr,color='tab:orange')
    clf=ax.plot(lcvn,pdf_ml,color='tab:gray')
    clf=ax.plot(lcvn,pdf_hl,color='tab:blue')
    ax.set_xlim([-10,20])
    ax.set_title(r'PDF of Annual mean AI',fontsize=16)
    ax.set_xlabel('%s (%s)'%(varnlb(vn),unitlb(vn)))
    ax.set_ylabel('Density')
    fig.savefig('%s/pdf.ann.%s.%s.png' % (odir,vn,fo), format='png', dpi=dpi)
    fig.savefig('%s/pdf.ann.%s.%s.pdf' % (odir,vn,fo), format='pdf', dpi=dpi)


# run
[plot(vn) for vn in lvn]
