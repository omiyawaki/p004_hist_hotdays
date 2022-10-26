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
lfo=['historical']
lcl=['his']
lre=['sea'] # can be empty
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)

def mdir(se,cl,fo,md,varn):
    rdir = '/project2/tas1/miyawaki/projects/000_hotdays/data/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    return rdir

for fo in lfo:
    lmd=mods(fo)
    for pc in lpc:
        for re in lre:
            mt,mq=rbin(re)
            abm=np.vstack([mt.ravel(),mq.ravel()])
            for se in lse:
                for cl in lcl:
                    c0=0 # counter
                    for imd in tqdm(range(len(lmd))):
                        md=lmd[imd]
                        sim=simu(fo,cl)
                        if sim=='ssp245':
                            yr='208001-210012'
                        elif sim=='historical':
                            yr='198001-200012'
                        odir=mdir(se,cl,fo,md,varn)
                        if not os.path.exists(odir):
                            os.makedirs(odir)

                        if pc=='':
                            kqpdf = pickle.load(open('%s/k%s_%s.%s.%s.pickle' % (odir,varn,yr,re,se), 'rb'))
                        else:
                            kqpdf = pickle.load(open('%s/k%s_%s.gt%s.%s.%s.pickle' % (odir,varn,yr,pc,re,se), 'rb'))
                        pdf=np.reshape(kqpdf(abm).T,mt.shape)

                        if c0==0:
                            ipdf=np.empty([len(lmd),pdf.shape[0],pdf.shape[1]])
                            c0=1

                        ipdf[imd,...]=pdf

                        if pc=='':
                            pickle.dump([pdf,mt,mq], open('%s/clmean.kqt2m.%s.%s.%s.pickle' % (odir,yr,re,se), 'wb'), protocol=5)	
                        else:
                            pickle.dump([pdf,mt,mq], open('%s/clmean.kqt2m.gt%s.%s.%s.%s.pickle' % (odir,pc,yr,re,se), 'wb'), protocol=5)	

                    # ENSEMBLE
                    edir=mdir(se,cl,fo,'mmm',varn)
                    if not os.path.exists(edir):
                        os.makedirs(edir)

                    # compute ensemble statistics
                    avgpdf = np.empty_like(pdf) # ensemble average
                    stdpdf = np.empty_like(avgpdf) # ensemble standard deviation
                    p50pdf = np.empty_like(avgpdf) # ensemble median
                    p25pdf = np.empty_like(avgpdf) # ensemble 25th prc
                    p75pdf = np.empty_like(avgpdf) # ensemble 75th prc
                    iqrpdf = np.empty_like(avgpdf) # ensemble IQR
                    minpdf = np.empty_like(avgpdf) # ensemble minimum
                    maxpdf = np.empty_like(avgpdf) # ensemble maximum
                    ptppdf = np.empty_like(avgpdf) # ensemble range
                    for it in tqdm(range(pdf.shape[0])):
                        for iq in range(pdf.shape[1]):
                            avgpdf[it,iq] = np.mean(      ipdf[:,it,iq],axis=0)
                            stdpdf[it,iq] = np.std(       ipdf[:,it,iq],axis=0)
                            p50pdf[it,iq] = np.percentile(ipdf[:,it,iq],50,axis=0)
                            p25pdf[it,iq] = np.percentile(ipdf[:,it,iq],25,axis=0)
                            p75pdf[it,iq] = np.percentile(ipdf[:,it,iq],75,axis=0)
                            minpdf[it,iq] = np.amin(      ipdf[:,it,iq],axis=0)
                            maxpdf[it,iq] = np.amax(      ipdf[:,it,iq],axis=0)
                            ptppdf[it,iq] = np.ptp(       ipdf[:,it,iq],axis=0)

                        stats={}
                        stats['mean'] = avgpdf
                        stats['stdev'] = stdpdf
                        stats['median'] = p50pdf
                        stats['prc25'] = p25pdf
                        stats['prc75'] = p75pdf
                        stats['min'] = minpdf
                        stats['max'] = maxpdf
                        stats['range'] = ptppdf

                        if pc=='':
                            pickle.dump([stats,mt,mq], open('%s/clmean.kqt2m.%s.%s.%s.pickle' % (edir,yr,re,se), 'wb'), protocol=5)	
                        else:
                            pickle.dump([stats,mt,mq], open('%s/clmean.kqt2m.gt%s.%s.%s.%s.pickle' % (edir,pc,yr,re,se), 'wb'), protocol=5)	

