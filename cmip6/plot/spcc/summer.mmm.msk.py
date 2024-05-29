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
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb
from regions import masklev0,regionsets
from skimage import measure, morphology

nt=7 # window size in days
p=97.5
lvn=['tas']
vnp= 'tas'
thd=0.25 # threshold to define hotspot
tlat=40
reverse=False
nhhl=False
af=False
tr=True
# lvn=['ooplh','ooplh_fixbc','ooplh_fixmsm','ooplh_rddsm']
# vnp='ooplh'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
dpi=600
skip507599=True

md='mmm'

# load land indices
lmi,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def cmap(vn):
    vbr=['td_mrsos','ti_pr','ti_ev']
    if vn in vbr:
        return 'BrBG'
    else:
        return 'RdBu_r'

def vmaxdd(vn):
    lvm={   
            'tas':  [1,0.1],
            }
    return lvm[vn]

def plot(vn):
    vmdd,dvmdd=vmaxdd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    ddpvn=xr.open_dataarray('%s/ddpc.md.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))
    pct=ddpvn['percentile']
    gpi=ddpvn['gpi']
    ddpvn=ddpvn.sel(percentile=pct==p).squeeze()
    if reverse and (vn in ['gflx','hfss','hfls','fa850','fat850','advt850','advtx850','advty850','advm850','advmx850','advmy850','rfa'] or 'ooplh' in vn):
        ddpvn=-ddpvn

    # jja and djf means
    ddpvnj=np.nanmean(ddpvn.data[5:8,:],axis=0)
    ddpvnd=np.nanmean(np.roll(ddpvn.data,1,axis=0)[:3,:],axis=0)

    # remap to lat x lon
    llddpvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llddpvnj[lmi]=ddpvnj.data
    llddpvnj=np.reshape(llddpvnj,(gr['lat'].size,gr['lon'].size))

    llddpvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llddpvnd[lmi]=ddpvnd.data
    llddpvnd=np.reshape(llddpvnd,(gr['lat'].size,gr['lon'].size))

    da=xr.DataArray(llddpvnd,coords={'lat':gr['lat'],'lon':gr['lon']},dims=('lat','lon'))
    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    llddpvnj = np.append(llddpvnj, llddpvnj[...,0][...,None],axis=1)
    llddpvnd = np.append(llddpvnd, llddpvnd[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use jja for nh, djf for sh
    llddpvn=np.copy(llddpvnd)
    llddpvn[gr['lat']>0]=llddpvnj[gr['lat']>0]

    # hotspot mask
    msk=(llddpvn>=thd).astype(int)
    print(msk.flatten().sum())

    # tropics, nh, and sh masks
    mtr=(np.abs(mlat)<=tlat).astype(int)
    mnh=(mlat>=0).astype(int)
    msh=(mlat<=0).astype(int)

    def cluster(mask,nshift):
        mask=np.roll(mask,nshift,axis=1) # shift to keep shape within domain center
        mask=mask.astype(bool)
        lbl=measure.label(mask)
        rp=measure.regionprops(lbl)
        size=max([i.area for i in rp]) # largest size of cluster
        mask=morphology.remove_small_objects(mask, min_size=size-50) # keep only largest cluster
        mask=morphology.remove_small_holes(mask, area_threshold=9) # remove small holes
        return np.roll(mask,-nshift,axis=1)

    def makemask(reg,mplus,nshift):
        mask=masklev0(regionsets(reg),da,'country').data
        mask=np.append(mask,mask[...,0][...,None],axis=1)
        mask=mask*mplus
        mask[np.isnan(mask)]=0
        mask=cluster(mask,nshift)
        print(reg,' has ',mask.flatten().sum(),' gridpoints')
        omask=mask[:,:-1].flatten()
        omask=np.delete(omask,omi)
        pickle.dump(omask,open('/project/amp02/miyawaki/data/p004/cmip6/hotspots/%s.pickle'%reg,'wb'))
        return mask

    ind=makemask('ind',msk*mtr,0)       # india
    sea=makemask('sea',msk*mtr,0)       # southeast asia
    ca= makemask('ca' ,msk*mtr,0)       # central america
    sa= makemask('sa' ,msk*mtr,0)       # south america
    shl=makemask('shl',msk*mtr*mnh,144) # sahel
    saf=makemask('saf',msk*mtr*msh,144) # southern africa

    # assign unique values for each region
    msk[ind==1]=4
    msk[sea==1]=5
    msk[ca==1]=6
    msk[sa==1]=7
    msk[shl==1]=8
    msk[saf==1]=9

    if tr:
        # plot TROPICS ONLY
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(),cmap='RdBu_r')
        ax.contour(mlon, mlat, msk, transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_extent((-180,180,-tlat,tlat),crs=ccrs.PlateCarree())
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
        gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
        gl.yformatter=LatitudeFormatter()
        gl.xlines=False
        gl.left_labels=False
        gl.bottom_labels=False
        gl.right_labels=True
        gl.top_labels=False
        cb=fig.colorbar(clf,location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label=r'$\Delta \delta %s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
        fig.savefig('%s/summer.ddp%02d%s.%s.%s.tr.msk.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.contour(mlon, mlat, msk, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \delta %s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/summer.ddp%02d%s.%s.%s.msk.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
