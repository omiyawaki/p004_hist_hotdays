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
from cmip6util import mods
from utils import corr

nt=7 # window size in days
pref1='ddp'
varn1='hfls'
pref2='ddp'
varn2='mrsos'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc'
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

md='mi'
idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
if not os.path.exists(odir):
    os.makedirs(odir)

i1=pickle.load(open('%s/%s%s_%s_%s.%s.pickle' % (idir1,pref1,varn1,his,fut,se), 'rb'))
i2=pickle.load(open('%s/%s%s_%s_%s.%s.pickle' % (idir2,pref2,varn2,his,fut,se), 'rb'))

if i2.shape != i1.shape:
    i2=np.transpose(i2[...,None],[0,1,4,2,3])

r=corr(i1,i2,0)
pickle.dump(r,open('%s/rsq.%s_%s_%s.%s.pickle' % (odir,varn,his,fut,se), 'wb'),protocol=5)
