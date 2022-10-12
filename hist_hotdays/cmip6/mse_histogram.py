import sys
sys.path.append('/project2/tas1/miyawaki/projects/000_hotdays/scripts')
# from misc.dirnames import get_datadir, get_plotdir
# from misc.filenames import filenames_raw
# from proc.r1 import save_r1
# from plot.titles import make_title_sim_time
import os
import pickle
import glob
import numpy as np
from misc import par
from tqdm import tqdm
from titlestr import make_titlestr
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from netCDF4 import Dataset
# import tikzplotlib

mmm = 1
try_load = 1
proj = '000_hotdays'
# varname = 'mses'
varname = 'mse50000'
# varname = 'tas'
# modellist = ['UKESM1-0-LL']
# modellist = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'GFDL-CM4', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 'INM-CM5-0', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'UKESM1-0-LL']
modellist = ['ACCESS-CM2', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'UKESM1-0-LL', 'INM-CM4-8', 'INM-CM5-0', 'KACE-1-0-G', 'GFDL-CM4', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'UKESM1-0-LL']
yr_span_hist = '198001-200012'
yr_span_rcp = '208001-210012'
imethod = 'linear'
# regionlist = ['trop', 'nhmid']
regionlist = ['trop']
# seaslist = [None, 'jja', 'son', 'djf', 'mam'] # None for all data points, jja, djf, son, mam
seaslist = [None] # None for all data points, jja, djf, son, mam
prct = np.concatenate((np.array([0.001, 1.]), np.arange(5.,81.,5.), np.arange(82.5,98,2.5), [99]))

# specify latitude bounds for select regions
lat_bounds = {}
lat_bounds['trop'] = {}
lat_bounds['trop']['lo'] = -20
lat_bounds['trop']['up'] = 20
# lat_bounds['nhmid']['lo'] = 20
# lat_bounds['nhmid']['up'] = 40
lat_bounds['nhmid'] = {}
lat_bounds['nhmid']['lo'] = 40
lat_bounds['nhmid']['up'] = 60

# loop over models
for model in modellist:
    if mmm:
        model = 'mmm'

    print('\nWorking on model: %s' % (model))

    # loop over regions
    for region in regionlist:
        print('%s over region: %s' % (' '*10, region))

        # assign lower and upper latitude bounds for the latitudinal mean
        lat_lo = lat_bounds[region]['lo']
        lat_up = lat_bounds[region]['up']

        # loop over seasons
        for seas in seaslist:
            print('%s on season: %s' % (' '*20, seas))

            # create title string
            titlestr = make_titlestr(varname=varname, seas=seas, region=region)
            print(titlestr)

            datadir = '/project2/tas1/miyawaki/projects/000_hotdays/data/proc/%s/%s/%s_%g_%g' % (model, varname, region, lat_lo, lat_up)
            plotdir = '/project2/tas1/miyawaki/projects/000_hotdays/plots/%s/%s/%s_%g_%g' % (model, varname, region, lat_lo, lat_up)

            # create directories is they don't exist
            if not os.path.isdir(datadir):
                os.makedirs(datadir)
            if not os.path.isdir(plotdir):
                os.makedirs(plotdir)

            # location of pickled percentile dmse data
            dcmse_file = '%s/dc%s_%s_%s.pickle' % (datadir, varname, region, seas)

            # load data
            if not (os.path.isfile(dcmse_file) and try_load) and not mmm:
                ###########################
                # load land/ocean mask
                ###########################
                # sftlf_file = Dataset('/project2/tas1/ockham/data9/tas/CMIP5_RAW/%s/historical/atmos/fx/sftlf/r0i0p0/sftlf_fx_%s_historical_r0i0p0.nc' % (model, model), 'r')
                sftlf_file = Dataset(glob.glob('/project2/tas1/ockham/data9/tas/CMIP6_RAW/%s/historical/atmos/fx/sftlf/%s/sftlf_fx_%s_historical_%s_*.nc' % (model, par.ens['historical'][model], model, par.ens['historical'][model]))[0], 'r')
                sftlf_hist = sftlf_file.variables['sftlf'][:]
                # same for RCP85
                # sftlf_file = Dataset('/project2/tas1/ockham/data9/tas/CMIP5_RAW/%s/rcp85/atmos/fx/sftlf/r0i0p0/sftlf_fx_%s_rcp85_r0i0p0.nc' % (model, model), 'r')
                sftlf_file = Dataset(glob.glob('/project2/tas1/ockham/data9/tas/CMIP6_RAW/%s/ssp245/atmos/fx/sftlf/%s/sftlf_fx_%s_ssp245_%s_*.nc' % (model, par.ens['ssp245'][model], model, par.ens['ssp245'][model]))[0], 'r')
                sftlf_rcp = sftlf_file.variables['sftlf'][:]

                ###########################
                # load grid data
                ###########################
                grid = {}
                grid['lat'] = sftlf_file.variables['lat'][:]
                grid['lon'] = sftlf_file.variables['lon'][:]

                ###########################
                # load temperature data
                ###########################
                if seas is None:
                    mse_hist_fname = glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/historical/%s/%s_day_%s_historical_%s_*_%s.nc' % (proj, model, varname, model, par.ens['historical'][model], yr_span_hist))[0]
                    tas_hist_fname = mse_hist_fname.replace('mse', 'ta')
                    hus_hist_fname = mse_hist_fname.replace('mse', 'hus')

                    mse_hist_file = Dataset(mse_hist_fname, 'r')
                    tas_hist_file = Dataset(tas_hist_fname, 'r')
                    hus_hist_file = Dataset(hus_hist_fname, 'r')

                    mse_rcp_fname = glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/ssp245/%s/%s_day_%s_ssp245_%s_*_%s.nc' % (proj, model, varname, model, par.ens['ssp245'][model], yr_span_rcp))[0]
                    tas_rcp_fname = mse_rcp_fname.replace('mse', 'ta')
                    hus_rcp_fname = mse_rcp_fname.replace('mse', 'hus')

                    mse_rcp_file = Dataset(mse_rcp_fname, 'r')
                    tas_rcp_file = Dataset(tas_rcp_fname, 'r')
                    hus_rcp_file = Dataset(hus_rcp_fname, 'r')
                else:
                    mse_hist_fname = glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/historical/%s/%s_day_%s_historical_%s_*_%s.%s.nc' % (proj, model, varname, model, par.ens['historical'][model], yr_span_hist, seas))[0]
                    tas_hist_fname = mse_hist_fname.replace('mse', 'ta')
                    hus_hist_fname = mse_hist_fname.replace('mse', 'hus')

                    mse_hist_file = Dataset(mse_hist_fname, 'r')
                    tas_hist_file = Dataset(tas_hist_fname, 'r')
                    hus_hist_file = Dataset(hus_hist_fname, 'r')

                    mse_rcp_fname = glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/ssp245/%s/%s_day_%s_ssp245_%s_*_%s.%s.nc' % (proj, model, varname, model, par.ens['ssp245'][model], yr_span_rcp, seas))[0]
                    tas_rcp_fname = mse_rcp_fname.replace('mse', 'ta')
                    hus_rcp_fname = mse_rcp_fname.replace('mse', 'hus')

                    mse_rcp_file = Dataset(mse_hist_fname, 'r')
                    tas_rcp_file = Dataset(tas_rcp_fname, 'r')
                    hus_rcp_file = Dataset(hus_rcp_fname, 'r')

                if varname == 'mses':
                    mse_hist = np.squeeze(mse_hist_file.variables['mses'][:])
                    mse_rcp = np.squeeze(mse_rcp_file.variables['mses'][:])
                    tas_hist = np.squeeze(tas_hist_file.variables['tas'][:])
                    tas_rcp = np.squeeze(tas_rcp_file.variables['tas'][:])
                    hus_hist = np.squeeze(hus_hist_file.variables['huss'][:])
                    hus_rcp = np.squeeze(hus_rcp_file.variables['huss'][:])
                else:
                    mse_hist = np.squeeze(mse_hist_file.variables['mse'][:])
                    mse_rcp = np.squeeze(mse_rcp_file.variables['mse'][:])
                    tas_hist = np.squeeze(tas_hist_file.variables['ta'][:])
                    tas_rcp = np.squeeze(tas_rcp_file.variables['ta'][:])
                    hus_hist = np.squeeze(hus_hist_file.variables['hus'][:])
                    hus_rcp = np.squeeze(hus_rcp_file.variables['hus'][:])

                ###########################
                # fill missing data with nan
                ###########################
                mse_hist = mse_hist.filled(fill_value=np.nan)
                mse_rcp = mse_rcp.filled(fill_value=np.nan)
                tas_hist = tas_hist.filled(fill_value=np.nan)
                tas_rcp = tas_rcp.filled(fill_value=np.nan)
                hus_hist = hus_hist.filled(fill_value=np.nan)
                hus_rcp = hus_rcp.filled(fill_value=np.nan)

                ###########################
                # separate over land/ocean
                ###########################
                lmask_hist = np.empty_like(sftlf_hist)
                omask_hist = np.empty_like(sftlf_hist)
                # make sftlf into a binary land/ocean mask
                lmask_hist[sftlf_hist <= 50] = np.nan
                lmask_hist[sftlf_hist > 50] = 1
                omask_hist[sftlf_hist <= 50] = 1
                omask_hist[sftlf_hist > 50] = np.nan
                # repeat for rcp
                lmask_rcp = np.empty_like(sftlf_rcp)
                omask_rcp = np.empty_like(sftlf_rcp)
                lmask_rcp[sftlf_rcp <= 50] = np.nan
                lmask_rcp[sftlf_rcp > 50] = 1
                omask_rcp[sftlf_rcp <= 50] = 1
                omask_rcp[sftlf_rcp > 50] = np.nan

                # apply mask to tas
                mse_hist_l = mse_hist*lmask_hist
                mse_hist_o = mse_hist*omask_hist
                tas_hist_l = tas_hist*lmask_hist
                tas_hist_o = tas_hist*omask_hist
                hus_hist_l = hus_hist*lmask_hist
                hus_hist_o = hus_hist*omask_hist
                # repeat for rcp
                mse_rcp_l = mse_rcp*lmask_rcp
                mse_rcp_o = mse_rcp*omask_rcp
                tas_rcp_l = tas_rcp*lmask_rcp
                tas_rcp_o = tas_rcp*omask_rcp
                hus_rcp_l = hus_rcp*lmask_rcp
                hus_rcp_o = hus_rcp*omask_rcp

                # tas_hist=None; tas_rcp=None;

                ###########################
                # limit to specified region
                ###########################
                idx_trop = np.transpose(np.argwhere((grid['lat']>lat_lo) & (grid['lat']<lat_up)))
                lat_trop = np.squeeze(grid['lat'][idx_trop])

                mse_hist_trop = np.squeeze(mse_hist[:,idx_trop,:])
                mse_hist_l_trop = np.squeeze(mse_hist_l[:,idx_trop,:])
                mse_hist_o_trop = np.squeeze(mse_hist_o[:,idx_trop,:])

                mse_rcp_trop = np.squeeze(mse_rcp[:,idx_trop,:])
                mse_rcp_l_trop = np.squeeze(mse_rcp_l[:,idx_trop,:])
                mse_rcp_o_trop = np.squeeze(mse_rcp_o[:,idx_trop,:])

                tas_hist_trop = np.squeeze(tas_hist[:,idx_trop,:])
                tas_hist_l_trop = np.squeeze(tas_hist_l[:,idx_trop,:])
                tas_hist_o_trop = np.squeeze(tas_hist_o[:,idx_trop,:])

                tas_rcp_trop = np.squeeze(tas_rcp[:,idx_trop,:])
                tas_rcp_l_trop = np.squeeze(tas_rcp_l[:,idx_trop,:])
                tas_rcp_o_trop = np.squeeze(tas_rcp_o[:,idx_trop,:])

                hus_hist_trop = np.squeeze(hus_hist[:,idx_trop,:])
                hus_hist_l_trop = np.squeeze(hus_hist_l[:,idx_trop,:])
                hus_hist_o_trop = np.squeeze(hus_hist_o[:,idx_trop,:])

                hus_rcp_trop = np.squeeze(hus_rcp[:,idx_trop,:])
                hus_rcp_l_trop = np.squeeze(hus_rcp_l[:,idx_trop,:])
                hus_rcp_o_trop = np.squeeze(hus_rcp_o[:,idx_trop,:])

                ###########################
                # sample percentiles for each latitude
                ###########################
                cmse_hist = np.empty([mse_hist_trop.shape[1],len(prct)])
                cmse_hist_l = np.empty([mse_hist_l_trop.shape[1],len(prct)])
                cmse_hist_o = np.empty([mse_hist_o_trop.shape[1],len(prct)])
                cmse_rcp = np.empty([mse_rcp_trop.shape[1],len(prct)])
                cmse_rcp_l = np.empty([mse_rcp_l_trop.shape[1],len(prct)])
                cmse_rcp_o = np.empty([mse_rcp_o_trop.shape[1],len(prct)])

                ctas_hist = np.empty([tas_hist_trop.shape[1],len(prct)])
                ctas_hist_l = np.empty([tas_hist_l_trop.shape[1],len(prct)])
                ctas_hist_o = np.empty([tas_hist_o_trop.shape[1],len(prct)])
                ctas_rcp = np.empty([tas_rcp_trop.shape[1],len(prct)])
                ctas_rcp_l = np.empty([tas_rcp_l_trop.shape[1],len(prct)])
                ctas_rcp_o = np.empty([tas_rcp_o_trop.shape[1],len(prct)])

                chus_hist = np.empty([hus_hist_trop.shape[1],len(prct)])
                chus_hist_l = np.empty([hus_hist_l_trop.shape[1],len(prct)])
                chus_hist_o = np.empty([hus_hist_o_trop.shape[1],len(prct)])
                chus_rcp = np.empty([hus_rcp_trop.shape[1],len(prct)])
                chus_rcp_l = np.empty([hus_rcp_l_trop.shape[1],len(prct)])
                chus_rcp_o = np.empty([hus_rcp_o_trop.shape[1],len(prct)])

                for ilat in range(tas_hist_l_trop.shape[1]):
                    # flatten arrays
                    mse_hist_flat   = np.ndarray.flatten(mse_hist_trop[:,ilat,:])
                    mse_hist_l_flat = np.ndarray.flatten(mse_hist_l_trop[:,ilat,:])
                    mse_hist_o_flat = np.ndarray.flatten(mse_hist_o_trop[:,ilat,:])
                    mse_rcp_flat    = np.ndarray.flatten(mse_rcp_trop[:,ilat,:])
                    mse_rcp_l_flat  = np.ndarray.flatten(mse_rcp_l_trop[:,ilat,:])
                    mse_rcp_o_flat  = np.ndarray.flatten(mse_rcp_o_trop[:,ilat,:])

                    tas_hist_flat   = np.ndarray.flatten(tas_hist_trop[:,ilat,:])
                    tas_hist_l_flat = np.ndarray.flatten(tas_hist_l_trop[:,ilat,:])
                    tas_hist_o_flat = np.ndarray.flatten(tas_hist_o_trop[:,ilat,:])
                    tas_rcp_flat    = np.ndarray.flatten(tas_rcp_trop[:,ilat,:])
                    tas_rcp_l_flat  = np.ndarray.flatten(tas_rcp_l_trop[:,ilat,:])
                    tas_rcp_o_flat  = np.ndarray.flatten(tas_rcp_o_trop[:,ilat,:])

                    hus_hist_flat   = np.ndarray.flatten(hus_hist_trop[:,ilat,:])
                    hus_hist_l_flat = np.ndarray.flatten(hus_hist_l_trop[:,ilat,:])
                    hus_hist_o_flat = np.ndarray.flatten(hus_hist_o_trop[:,ilat,:])
                    hus_rcp_flat    = np.ndarray.flatten(hus_rcp_trop[:,ilat,:])
                    hus_rcp_l_flat  = np.ndarray.flatten(hus_rcp_l_trop[:,ilat,:])
                    hus_rcp_o_flat  = np.ndarray.flatten(hus_rcp_o_trop[:,ilat,:])

                    # remove nans
                    mse_hist_flat = mse_hist_flat[~np.isnan(mse_hist_flat)]
                    mse_hist_l_flat = mse_hist_l_flat[~np.isnan(mse_hist_l_flat)]
                    mse_hist_o_flat = mse_hist_o_flat[~np.isnan(mse_hist_o_flat)]
                    mse_rcp_flat = mse_rcp_flat[~np.isnan(mse_rcp_flat)]
                    mse_rcp_l_flat = mse_rcp_l_flat[~np.isnan(mse_rcp_l_flat)]
                    mse_rcp_o_flat = mse_rcp_o_flat[~np.isnan(mse_rcp_o_flat)]

                    tas_hist_flat = tas_hist_flat[~np.isnan(tas_hist_flat)]
                    tas_hist_l_flat = tas_hist_l_flat[~np.isnan(tas_hist_l_flat)]
                    tas_hist_o_flat = tas_hist_o_flat[~np.isnan(tas_hist_o_flat)]
                    tas_rcp_flat = tas_rcp_flat[~np.isnan(tas_rcp_flat)]
                    tas_rcp_l_flat = tas_rcp_l_flat[~np.isnan(tas_rcp_l_flat)]
                    tas_rcp_o_flat = tas_rcp_o_flat[~np.isnan(tas_rcp_o_flat)]

                    hus_hist_flat = hus_hist_flat[~np.isnan(hus_hist_flat)]
                    hus_hist_l_flat = hus_hist_l_flat[~np.isnan(hus_hist_l_flat)]
                    hus_hist_o_flat = hus_hist_o_flat[~np.isnan(hus_hist_o_flat)]
                    hus_rcp_flat = hus_rcp_flat[~np.isnan(hus_rcp_flat)]
                    hus_rcp_l_flat = hus_rcp_l_flat[~np.isnan(hus_rcp_l_flat)]
                    hus_rcp_o_flat = hus_rcp_o_flat[~np.isnan(hus_rcp_o_flat)]

                    # compute the percentile values
                    pmse_hist = np.percentile(mse_hist_flat, prct, interpolation=imethod)
                    pmse_hist_l = np.percentile(mse_hist_l_flat, prct, interpolation=imethod)
                    pmse_hist_o = np.percentile(mse_hist_o_flat, prct, interpolation=imethod)
                    pmse_rcp = np.percentile(mse_rcp_flat, prct, interpolation=imethod)
                    pmse_rcp_l = np.percentile(mse_rcp_l_flat, prct, interpolation=imethod)
                    pmse_rcp_o = np.percentile(mse_rcp_o_flat, prct, interpolation=imethod)

                    ptas_hist = np.percentile(tas_hist_flat, prct, interpolation=imethod)
                    ptas_hist_l = np.percentile(tas_hist_l_flat, prct, interpolation=imethod)
                    ptas_hist_o = np.percentile(tas_hist_o_flat, prct, interpolation=imethod)
                    ptas_rcp = np.percentile(tas_rcp_flat, prct, interpolation=imethod)
                    ptas_rcp_l = np.percentile(tas_rcp_l_flat, prct, interpolation=imethod)
                    ptas_rcp_o = np.percentile(tas_rcp_o_flat, prct, interpolation=imethod)

                    phus_hist = np.percentile(hus_hist_flat, prct, interpolation=imethod)
                    phus_hist_l = np.percentile(hus_hist_l_flat, prct, interpolation=imethod)
                    phus_hist_o = np.percentile(hus_hist_o_flat, prct, interpolation=imethod)
                    phus_rcp = np.percentile(hus_rcp_flat, prct, interpolation=imethod)
                    phus_rcp_l = np.percentile(hus_rcp_l_flat, prct, interpolation=imethod)
                    phus_rcp_o = np.percentile(hus_rcp_o_flat, prct, interpolation=imethod)

                    # take the cumulative mean (i.e. 0th cumulative percentile is the average of all days because any day exceeds the 0th percentile
                    for iprct in range(len(prct)):
                        cmse_hist[ilat,iprct] = np.sum(mse_hist_flat[mse_hist_flat>pmse_hist[iprct]])/len(mse_hist_flat[mse_hist_flat>pmse_hist[iprct]])
                        cmse_hist_l[ilat,iprct] = np.sum(mse_hist_l_flat[mse_hist_l_flat>pmse_hist_l[iprct]])/len(mse_hist_l_flat[mse_hist_l_flat>pmse_hist_l[iprct]])
                        cmse_hist_o[ilat,iprct] = np.sum(mse_hist_o_flat[mse_hist_o_flat>pmse_hist_o[iprct]])/len(mse_hist_o_flat[mse_hist_o_flat>pmse_hist_o[iprct]])
                        cmse_rcp[ilat,iprct] = np.sum(mse_rcp_flat[mse_rcp_flat>pmse_rcp[iprct]])/len(mse_rcp_flat[mse_rcp_flat>pmse_rcp[iprct]])
                        cmse_rcp_l[ilat,iprct] = np.sum(mse_rcp_l_flat[mse_rcp_l_flat>pmse_rcp_l[iprct]])/len(mse_rcp_l_flat[mse_rcp_l_flat>pmse_rcp_l[iprct]])
                        cmse_rcp_o[ilat,iprct] = np.sum(mse_rcp_o_flat[mse_rcp_o_flat>pmse_rcp_o[iprct]])/len(mse_rcp_o_flat[mse_rcp_o_flat>pmse_rcp_o[iprct]])

                        ctas_hist[ilat,iprct] = np.sum(tas_hist_flat[tas_hist_flat>ptas_hist[iprct]])/len(tas_hist_flat[tas_hist_flat>ptas_hist[iprct]])
                        ctas_hist_l[ilat,iprct] = np.sum(tas_hist_l_flat[tas_hist_l_flat>ptas_hist_l[iprct]])/len(tas_hist_l_flat[tas_hist_l_flat>ptas_hist_l[iprct]])
                        ctas_hist_o[ilat,iprct] = np.sum(tas_hist_o_flat[tas_hist_o_flat>ptas_hist_o[iprct]])/len(tas_hist_o_flat[tas_hist_o_flat>ptas_hist_o[iprct]])
                        ctas_rcp[ilat,iprct] = np.sum(tas_rcp_flat[tas_rcp_flat>ptas_rcp[iprct]])/len(tas_rcp_flat[tas_rcp_flat>ptas_rcp[iprct]])
                        ctas_rcp_l[ilat,iprct] = np.sum(tas_rcp_l_flat[tas_rcp_l_flat>ptas_rcp_l[iprct]])/len(tas_rcp_l_flat[tas_rcp_l_flat>ptas_rcp_l[iprct]])
                        ctas_rcp_o[ilat,iprct] = np.sum(tas_rcp_o_flat[tas_rcp_o_flat>ptas_rcp_o[iprct]])/len(tas_rcp_o_flat[tas_rcp_o_flat>ptas_rcp_o[iprct]])

                        chus_hist[ilat,iprct] = np.sum(hus_hist_flat[hus_hist_flat>phus_hist[iprct]])/len(hus_hist_flat[hus_hist_flat>phus_hist[iprct]])
                        chus_hist_l[ilat,iprct] = np.sum(hus_hist_l_flat[hus_hist_l_flat>phus_hist_l[iprct]])/len(hus_hist_l_flat[hus_hist_l_flat>phus_hist_l[iprct]])
                        chus_hist_o[ilat,iprct] = np.sum(hus_hist_o_flat[hus_hist_o_flat>phus_hist_o[iprct]])/len(hus_hist_o_flat[hus_hist_o_flat>phus_hist_o[iprct]])
                        chus_rcp[ilat,iprct] = np.sum(hus_rcp_flat[hus_rcp_flat>phus_rcp[iprct]])/len(hus_rcp_flat[hus_rcp_flat>phus_rcp[iprct]])
                        chus_rcp_l[ilat,iprct] = np.sum(hus_rcp_l_flat[hus_rcp_l_flat>phus_rcp_l[iprct]])/len(hus_rcp_l_flat[hus_rcp_l_flat>phus_rcp_l[iprct]])
                        chus_rcp_o[ilat,iprct] = np.sum(hus_rcp_o_flat[hus_rcp_o_flat>phus_rcp_o[iprct]])/len(hus_rcp_o_flat[hus_rcp_o_flat>phus_rcp_o[iprct]])

                ############################
                ## take area weighted mean of the percentiles
                ############################
                #clat = np.transpose(np.cos(np.deg2rad(grid['lat'][idx_trop])))
                #ctas_hist_l_areaavg = np.nansum(clat * ctas_hist_l, axis=0) / np.nansum(clat)
                #ctas_hist_o_areaavg = np.nansum(clat * ctas_hist_o, axis=0) / np.nansum(clat)
                #ctas_rcp_l_areaavg = np.nansum(clat * ctas_rcp_l, axis=0) / np.nansum(clat)
                #ctas_rcp_o_areaavg = np.nansum(clat * ctas_rcp_o, axis=0) / np.nansum(clat)

                ############################
                ## warming as a function of percentile
                ############################
                #dcmse_l = ctas_rcp_l_areaavg - ctas_hist_l_areaavg
                #dcmse_o = ctas_rcp_o_areaavg - ctas_hist_o_areaavg

                ###########################
                # warming as a function of percentile
                ###########################
                dcmse_l = cmse_rcp_l - cmse_hist_l
                dcmse_o = cmse_rcp_o - cmse_hist_o

                dctas_l = ctas_rcp_l - ctas_hist_l
                dctas_o = ctas_rcp_o - ctas_hist_o

                dchus_l = chus_rcp_l - chus_hist_l
                dchus_o = chus_rcp_o - chus_hist_o

                ###########################
                # mean warming as a function of percentile
                ###########################
                dcmse_l_mean = np.nanmean(mse_rcp_l_trop, axis=(0,2)) - np.nanmean(mse_hist_l_trop, axis=(0,2))
                dcmse_o_mean = np.nanmean(mse_rcp_o_trop, axis=(0,2)) - np.nanmean(mse_hist_o_trop, axis=(0,2))

                dctas_l_mean = np.nanmean(tas_rcp_l_trop, axis=(0,2)) - np.nanmean(tas_hist_l_trop, axis=(0,2))
                dctas_o_mean = np.nanmean(tas_rcp_o_trop, axis=(0,2)) - np.nanmean(tas_hist_o_trop, axis=(0,2))

                dchus_l_mean = np.nanmean(hus_rcp_l_trop, axis=(0,2)) - np.nanmean(hus_hist_l_trop, axis=(0,2))
                dchus_o_mean = np.nanmean(hus_rcp_o_trop, axis=(0,2)) - np.nanmean(hus_hist_o_trop, axis=(0,2))

                ###########################
                # take area weighted mean of the percentiles
                ###########################
                clat = np.transpose(np.cos(np.deg2rad(grid['lat'][idx_trop])))
                dcmse_l = np.nansum(clat * dcmse_l, axis=0) / np.nansum(clat)
                dcmse_o = np.nansum(clat * dcmse_o, axis=0) / np.nansum(clat)

                dctas_l = np.nansum(clat * dctas_l, axis=0) / np.nansum(clat)
                dctas_o = np.nansum(clat * dctas_o, axis=0) / np.nansum(clat)

                dchus_l = np.nansum(clat * dchus_l, axis=0) / np.nansum(clat)
                dchus_o = np.nansum(clat * dchus_o, axis=0) / np.nansum(clat)

                ###########################
                # save data
                ###########################
                pickle.dump([dcmse_l, dcmse_o, dctas_l, dctas_o, dchus_l, dchus_o], open(dcmse_file, 'wb'))

            # load pickled data
            if mmm:
                dcmse_l_agg = np.empty([len(prct), len(modellist)])
                dcmse_o_agg = np.empty([len(prct), len(modellist)])
                dctas_l_agg = np.empty([len(prct), len(modellist)])
                dctas_o_agg = np.empty([len(prct), len(modellist)])
                dchus_l_agg = np.empty([len(prct), len(modellist)])
                dchus_o_agg = np.empty([len(prct), len(modellist)])
                for imodel in tqdm(range(len(modellist))):
                    currentmodel = modellist[imodel]

                    datadir0 = '/project2/tas1/miyawaki/projects/000_hotdays/data/proc/%s/%s/%s_%g_%g' % (currentmodel, varname, region, lat_lo, lat_up)
                    dcmse_file0 = '%s/dc%s_%s_%s.pickle' % (datadir0, varname, region, seas)

                    [dcmse_l_agg[:,imodel], dcmse_o_agg[:,imodel], dctas_l_agg[:,imodel], dctas_o_agg[:,imodel], dchus_l_agg[:,imodel], dchus_o_agg[:,imodel]] = pickle.load(open(dcmse_file0, 'rb'))

                # take multimodel mean
                dcmse_l = np.mean(dcmse_l_agg,1)
                dcmse_o = np.mean(dcmse_o_agg,1)
                dctas_l = np.mean(dctas_l_agg,1)
                dctas_o = np.mean(dctas_o_agg,1)
                dchus_l = np.mean(dchus_l_agg,1)
                dchus_o = np.mean(dchus_o_agg,1)

            else:
                [dcmse_l, dcmse_o, dctas_l, dctas_o, dchus_l, dchus_o] = pickle.load(open(dcmse_file, 'rb'))

            ###########################
            # plot dcmse
            ###########################
            if seas is None:
                plotname = '%s/dcmse' % (plotdir)
            else:
                plotname = '%s/dcmse.%s' % (plotdir, seas)
            fig, ax = plt.subplots()
            vmin = 0
            vmax = 100
            ax.axhline(1,0, 100, linewidth=0.5, color='k')
            ax.plot(prct, dcmse_l/dcmse_o, ':k', label='MSE')
            ax.plot(prct, dctas_l/dctas_o, '-k', label='Temperature')
            ax.plot(prct, dchus_l/dchus_o, '--k', label='Specific humidity')
            ax.set_title(titlestr)
            # make_title_sim_time(ax, sim, model=model, timemean=timemean)
            ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
            # if 'ymonmean' in timemean:
            #     ax.set_xticks(np.arange(0,12,1))
            #     ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            # else:
            ax.set_xlim(0,100)
            ax.set_xlabel('Percentile')
            ax.set_ylabel('$\delta$ Land / $\delta$ Ocean (unitless)')
            # ax.set_yticks(np.arange(-90,91,30))
            # ax.xaxis.set_minor_locator(MultipleLocator(10))
            # ax.yaxis.set_minor_locator(MultipleLocator(10))
            plt.legend()
            fig.set_size_inches(5, 4)
            plt.savefig('%s.pdf' % (plotname), format='pdf', dpi=300)
            plt.close()

        # end seas loop

    # end region loop

    # stop after one loop if multimodel mean
    if mmm:
        break

# end model loop
