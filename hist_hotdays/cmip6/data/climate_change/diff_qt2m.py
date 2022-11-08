import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/project2/tas1/miyawaki/common')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
from regions import rbin
from cmip6util import mods,simu

varn='qt2m'
lpc=['95'] # evaluate kde for values exceeding the gt percentile
fo1='ssp245'
cl1='fut'
yr1='208001-210012'
fo0='historical'
cl0='his'
yr0='198001-200012'
cl='fut-his'
fo='%s-%s'%(fo1,fo0)
lre=['swus'] # can be empty
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)

def mdir(se,cl,fo,md,varn):
    rdir = '/project2/tas1/miyawaki/projects/000_hotdays/data/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    return rdir

lmd=mods(fo1)
for pc in lpc:
    for re in lre:
        for se in lse:
            c0=0 # counter
            for imd in tqdm(range(len(lmd))):
                md=lmd[imd]
                idir1=mdir(se,cl1,fo1,md,varn)
                idir0=mdir(se,cl0,fo0,md,varn)
                odir=mdir(se,cl,fo,md,varn)
                if not os.path.exists(odir):
                    os.makedirs(odir)

                if pc=='':
                    pdf1,mt,mq = pickle.load(open('%s/clmean.kqt2m.%s.%s.%s.pickle' % (idir1,yr1,re,se), 'rb'))
                    pdf0,_,_ = pickle.load(open('%s/clmean.kqt2m.%s.%s.%s.pickle' % (idir0,yr0,re,se), 'rb'))
                else:
                    pdf1,mt,mq = pickle.load(open('%s/clmean.kqt2m.gt%s.%s.%s.%s.pickle' % (idir1,pc,yr1,re,se), 'rb'))
                    pdf0,_,_ = pickle.load(open('%s/clmean.kqt2m.gt%s.%s.%s.%s.pickle' % (idir0,pc,yr0,re,se), 'rb'))
                dpdf=pdf1-pdf0

                if c0==0:
                    idpdf=np.empty([len(lmd),dpdf.shape[0],dpdf.shape[1]])
                    c0=1

                idpdf[imd,...]=dpdf

                if pc=='':
                    pickle.dump([dpdf,mt,mq], open('%s/diff.kqt2m.%s.%s.%s.%s.pickle' % (odir,yr0,yr1,re,se), 'wb'), protocol=5)	
                else:
                    pickle.dump([dpdf,mt,mq], open('%s/diff.kqt2m.gt%s.%s.%s.%s.%s.pickle' % (odir,pc,yr0,yr1,re,se), 'wb'), protocol=5)	

            # ENSEMBLE
            edir=mdir(se,cl,fo,'mmm',varn)
            if not os.path.exists(edir):
                os.makedirs(edir)

            # compute ensemble statistics
            avgdpdf = np.empty_like(dpdf) # ensemble average
            stddpdf = np.empty_like(avgdpdf) # ensemble standard deviation
            p50dpdf = np.empty_like(avgdpdf) # ensemble median
            p25dpdf = np.empty_like(avgdpdf) # ensemble 25th prc
            p75dpdf = np.empty_like(avgdpdf) # ensemble 75th prc
            iqrdpdf = np.empty_like(avgdpdf) # ensemble IQR
            mindpdf = np.empty_like(avgdpdf) # ensemble minimum
            maxdpdf = np.empty_like(avgdpdf) # ensemble maximum
            ptpdpdf = np.empty_like(avgdpdf) # ensemble range
            for it in tqdm(range(dpdf.shape[0])):
                for iq in range(dpdf.shape[1]):
                    avgdpdf[it,iq] = np.mean(      idpdf[:,it,iq],axis=0)
                    stddpdf[it,iq] = np.std(       idpdf[:,it,iq],axis=0)
                    p50dpdf[it,iq] = np.percentile(idpdf[:,it,iq],50,axis=0)
                    p25dpdf[it,iq] = np.percentile(idpdf[:,it,iq],25,axis=0)
                    p75dpdf[it,iq] = np.percentile(idpdf[:,it,iq],75,axis=0)
                    mindpdf[it,iq] = np.amin(      idpdf[:,it,iq],axis=0)
                    maxdpdf[it,iq] = np.amax(      idpdf[:,it,iq],axis=0)
                    ptpdpdf[it,iq] = np.ptp(       idpdf[:,it,iq],axis=0)

                stats={}
                stats['mean'] = avgdpdf
                stats['stdev'] = stddpdf
                stats['median'] = p50dpdf
                stats['prc25'] = p25dpdf
                stats['prc75'] = p75dpdf
                stats['min'] = mindpdf
                stats['max'] = maxdpdf
                stats['range'] = ptpdpdf

                if pc=='':
                    pickle.dump([stats,mt,mq], open('%s/diff.kqt2m.%s.%s.%s.%s.pickle' % (edir,yr0,yr1,re,se), 'wb'), protocol=5)	
                else:
                    pickle.dump([stats,mt,mq], open('%s/diff.kqt2m.gt%s.%s.%s.%s.%s.pickle' % (edir,pc,yr0,yr1,re,se), 'wb'), protocol=5)	

