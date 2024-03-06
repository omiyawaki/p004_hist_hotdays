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
from util import mods
from utils import monname,varnlb,unitlb

nt=7 # window size in days
p=95
vn='ooplh'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='gwl2.0'
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

def vstr(vn):
    d={ 'ooplh':        r'$BC_{all}$',
        'ooplh_fixbc':  r'$BC_{hist}$',
        'ooplh_dbc':    r'$SM_{hist}$',
        'ooplh_rbcsm':  r'Residual',
        'ooplh_rddsm':  r'(b)$-$(c)',
        'ooplh_fixmsm': r'$BC_{hist}$, $\Delta\delta SM=0$',
        'ooplh_fixasm': r'$BC_{hist}$, $\Delta\delta SM=0$',
        'ooplh_mtr':    r'$\frac{\mathrm{d}LH}{\mathrm{d}SM}_{hist}\Delta SM$',
        'oopef':        r'$BC_{all}$',
        'oopef_fixbc':  r'$BC_{hist}$',
        'mrsos':        r'$SM$',
        'pr':           r'$P$',
            }
    return d[vn]

def vmax(vn):
    lvm={   'tas':  [1.5,0.1],
            'pr':   [1.5,0.1],
            'hfls': [15,1],
            'plh': [30,2],
            'plh_fixbc': [30,2],
            'ooplh': [10,1],
            'ooplh_orig': [30,2],
            'ooplh_fixbc': [30,2],
            }
    return lvm[vn]

def plot():
    vm,dvm=vmax(vn)

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    def load_vn(varn):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        return xr.open_dataarray('%s/ddpc.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
    ooplh=load_vn('ooplh')
    ooplh_fixbc=load_vn('ooplh_fixbc')
    ooplh_fixmsm=load_vn('ooplh_fixmsm')
    ooplh_rddsm=load_vn('ooplh_rddsm')

    # remap to lat x lon
    def remap(vn):
        vn=vn.squeeze()
        ll=np.nan*np.ones([vn.shape[0],gr['lat'].size*gr['lon'].size])
        ll[:,lmi]=vn.data
        return np.reshape(ll,(vn.shape[0],gr['lat'].size,gr['lon'].size))

    ooplh=remap(ooplh)
    ooplh_fixbc=remap(ooplh_fixbc)
    ooplh_fixmsm=remap(ooplh_fixmsm)
    ooplh_rddsm=remap(ooplh_rddsm)

    # phase shift SH by 6 months
    def pshift(v):
        sv=np.roll(v,6,axis=0)
        v[:,gr['lat']<0,:]=sv[:,gr['lat']<0,:]
        return v
    ooplh=pshift(ooplh)
    ooplh_fixbc=pshift(ooplh_fixbc)
    ooplh_fixmsm=pshift(ooplh_fixmsm)
    ooplh_rddsm=pshift(ooplh_rddsm)

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    def extend(vn):
        return np.append(vn, vn[...,0][...,None],axis=2)
    ooplh=extend(ooplh)
    ooplh_fixbc=extend(ooplh_fixbc)
    ooplh_fixmsm=extend(ooplh_fixmsm)
    ooplh_rddsm=extend(ooplh_rddsm)
    ld=[ooplh,ooplh_fixbc,ooplh_fixmsm,ooplh_rddsm] # store in list
    ls=[vstr('ooplh'),vstr('ooplh_fixbc'),vstr('ooplh_fixmsm'),vstr('ooplh_rddsm')]

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    for m in tqdm(range(12)):
        # plot pct warming - mean warming
        fig,ax=plt.subplots(nrows=1,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,3),constrained_layout=True)
        ax=ax.flatten()
        if m==0:
            fig.suptitle(r'%s month after winter solstice' % (m+1),fontsize=16)
        else:
            fig.suptitle(r'%s months after winter solstice' % (m+1),fontsize=16)

        for v in range(4):
            if m==6 and v=='ooplh_rddsm':
                clf=ax[v].contourf(mlon, mlat, ld[v][m,...], np.arange(-vm,vm+dvm,dvm+dvm*1e-2),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            else:
                clf=ax[v].contourf(mlon, mlat, ld[v][m,...], np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            ax[v].coastlines()
            ax[v].set_title(r'%s' % (ls[v]),fontsize=16)
            fig.savefig('%s/decomp.ddp%02d%s.%s.%02d.png' % (odir,p,vn,fo,m+1), format='png', dpi=300)
        cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=16)
        cb.set_label(label=r'$\Delta \delta %s^{%02d}$ (%s)'%(varnlb('hfls'),p,unitlb('hfls')),size=16)
        fig.savefig('%s/decomp.ddp%02d%s.%s.%02d.png' % (odir,p,vn,fo,m+1), format='png', dpi=300)

# run
plot()
