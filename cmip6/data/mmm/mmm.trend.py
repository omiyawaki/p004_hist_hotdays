import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
from scipy.stats import linregress
from tqdm import tqdm
from util import mods

lvn=['tas'] # input1
se='ts'
tavg='jja'
fo1='historical'
fo2='ssp370'
fo='%s+%s'%(fo1,fo2)
yr='1950-2020'

lmd=mods(fo1)

def calc_mmm(varn):

    for i,md in enumerate(tqdm(lmd)):
        idir0 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        odir0 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir0):
            os.makedirs(odir0)

        # percentile trend
        pvn=pickle.load(open('%s/trend.pc.%s.%s.%s.pickle' % (idir0,varn,yr,tavg),'rb'))
        pvn=np.stack([pvn[i]['slope'] for i in range(len(pvn))])

        # mean trend
        mvn=pickle.load(open('%s/trend.m.%s.%s.%s.pickle' % (idir0,varn,yr,tavg),'rb'))['slope']

        # relative trend
        ddpvn=pvn-np.transpose(mvn[...,None],[1,0])

        if i==0:
            imvn=np.empty(np.insert(np.asarray(mvn.shape),0,len(lmd)))
            ipvn=np.empty(np.insert(np.asarray(pvn.shape),0,len(lmd)))
            iddpvn=np.empty(np.insert(np.asarray(ddpvn.shape),0,len(lmd)))

        imvn[i,...]=mvn
        ipvn[i,...]=pvn
        iddpvn[i,...]=ddpvn

        # save individual model data
        ddpvn=[ddpvn[i,:] for i in range(ddpvn.shape[0])]
        pickle.dump(ddpvn,open('%s/trend.ddpc.%s.%s.%s.pickle' % (odir0,varn,yr,tavg),'wb'))

    # compute mmm and std
    mmvn=np.nanmean(imvn,axis=0)
    mpvn=np.nanmean(ipvn,axis=0)
    mddpvn=np.nanmean(iddpvn,axis=0)

    smvn=np.nanstd(imvn,axis=0)
    spvn=np.nanstd(ipvn,axis=0)
    sddpvn=np.nanstd(iddpvn,axis=0)

    # save mmm and std
    odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mmm',varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    lmpvn=[mpvn[i,:] for i in range(mpvn.shape[0])]
    lspvn=[mpvn[i,:] for i in range(mpvn.shape[0])]

    lmddpvn=[mddpvn[i,:] for i in range(mddpvn.shape[0])]
    lsddpvn=[mddpvn[i,:] for i in range(mddpvn.shape[0])]

    pickle.dump(mmvn,open('%s/trend.m.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))
    pickle.dump(smvn,open('%s/std.trend.m.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))

    pickle.dump(lmpvn,open('%s/trend.pc.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))
    pickle.dump(lspvn,open('%s/std.trend.pc.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))

    pickle.dump(lmddpvn,open('%s/trend.ddpc.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))
    pickle.dump(lsddpvn,open('%s/std.trend.ddpc.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))

[calc_mmm(vn) for vn in lvn]
