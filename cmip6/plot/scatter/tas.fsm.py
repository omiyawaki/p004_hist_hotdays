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

dt=0.25
tbin=np.arange(-10,10+dt,dt)
tvn='tas'
lvn=['fsm']
vnp= 'fsm'
tlat=30
plat=30
nhhl=True
tropics=False
reverse=True
# lvn=['ooplh','ooplh_fixbc','ooplh_fixmsm','ooplh_rsm']
# vnp='ooplh'
se = 'sc' # season (ann, djf, mam, jja, son)

fo='historical' # forcings 
yr='1980-2000'

# fo='ssp370' # forcings 
# yr='gwl2.0'

dpi=600
skip507599=True

md='CESM2'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

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
    elif vn=='snc':
        return 'RdBu'
    else:
        return 'RdBu_r'

def vmax(vn):
    lvm={   
            'tas':  [30,5],
            'snc':  [1,0.1],
            'fsm':  [10,1],
            'hfss':  [20,2],
            'ta850':  [1,0.1],
            'wap850':  [50,5],
            'va850':  [20,2],
            }
    return lvm[vn]

def vmaxd(vn):
    lvm={   
            'tas':  [10,1],
            'snc':  [1,0.1],
            'fsm':  [10,1],
            'hfss':  [20,2],
            'ta850':  [4,0.25],
            'wap850':  [50,5],
            'va850':  [10,0.1],
            }
    return lvm[vn]

def plot(vn):
    vm,dvm=vmax(vn)
    vmd,dvmd=vmaxd(vn)
    vnlb,unlb=varnlb(vn),unitlb(vn)
    if 'tas' in vn:
        unlb='$^\circ$C'

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s' % (se,fo,md)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # climo 
    def loadvn(px,varn):
        xvn=xr.open_dataarray('%s/%s/%s.%s_%s.%s.nc' % (idir,varn,px,varn,yr,se))
        if reverse and (varn in ['fsm','gflx','hfss','hfls','fat850','fa850','advt850','advtx850','advty850','advm850','advmx850','advmy850','rfa'] or 'ooplh' in varn):
            xvn=-xvn
        if 'wap' in varn:
            xvn=xvn*86400/100
        if 'tas' in varn:
            xvn=xvn-273.15
        return xvn

    avn=loadvn('lm',vn).data.flatten()
    atas=loadvn('lm',tvn).data.flatten()

    # T bins
    lb=np.digitize(atas,tbin)
    bvn=[np.nanmean(avn[lb==ib]) for ib in tqdm(range(len(tbin)))]

    # scatter
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
    # ax.plot(atas,avn,'.k',markersize=0.5,alpha=0.2)
    ax.plot(tbin,bvn,'-k')
    ax.set_xlabel('$%s$ (%s)'%(varnlb(tvn),unitlb(tvn)))
    ax.set_ylabel('$%s$ (%s)'%(vnlb,unlb))
    fig.savefig('%s/scat.a.%s.%s.png'%(odir,tvn,vn),format='png',dpi=dpi)

# run
[plot(vn) for vn in lvn]
