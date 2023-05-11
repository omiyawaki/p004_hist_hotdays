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
from misc.e2q import e2q
from misc.comp_esat import comp_esat
from tqdm import tqdm
from titlestr import make_titlestr
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from netCDF4 import Dataset
# import tikzplotlib

# mmm = 1 
# try_load = 1
mmm = 0 
try_load = 0
proj = '000_hotdays'
pref = 1e5
# lev = 50000
# varname = 'ta%g' % (lev)
varname = 'ts'
if varname == 'ts':
    varname_q = 'huss'
else:
    varname_q = 'hus%g' % (lev)
# modellist = ['MPI-ESM1-2-LR']
# modellist = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'GFDL-CM4', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 'INM-CM5-0', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'UKESM1-0-LL']
modellist = ['ACCESS-CM2', 'CanESM5', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 'INM-CM5-0', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'UKESM1-0-LL']
# modellist = ['GFDL-ESM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 'INM-CM5-0', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'UKESM1-0-LL']
yr_span_hist = '198001-200012'
yr_span_rcp = '208001-210012'
imethod = 'linear'
sample_method = 'latband' # latband: sample over latitude bands; gridpt: sample at each lat-lon grid point
# regionlist = ['trop', 'nhmid']
regionlist = ['nhmid']
# seaslist = [None, 'jja', 'son', 'djf', 'mam'] # None for all data points, jja, djf, son, mam
seaslist = [None] # None for all data points, jja, djf, son, mam
prct = np.concatenate((np.array([0.001, 1.]), np.arange(5.,81.,5.), np.arange(82.5,98,2.5), [99]))

# specify latitude bounds for select regions
lat_bounds = {}
lat_bounds['trop'] = {}
lat_bounds['trop']['lo'] = -20
lat_bounds['trop']['up'] = 20

# lat_bounds['nhmid'] = {}
# lat_bounds['nhmid']['lo'] = 20
# lat_bounds['nhmid']['up'] = 40
lat_bounds['nhmid'] = {}
lat_bounds['nhmid']['lo'] = 30
lat_bounds['nhmid']['up'] = 50
# lat_bounds['nhmid'] = {}
# lat_bounds['nhmid']['lo'] = 40
# lat_bounds['nhmid']['up'] = 60

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

            datadir = '/project2/tas1/miyawaki/projects/000_hotdays/data/proc/%s/%s/%s/%s_%g_%g' % (model, sample_method, varname, region, lat_lo, lat_up)
            plotdir = '/project2/tas1/miyawaki/projects/000_hotdays/plots/%s/%s/%s/%s_%g_%g' % (model, sample_method, varname, region, lat_lo, lat_up)

            # create directories is they don't exist
            if not os.path.isdir(datadir):
                os.makedirs(datadir)
            if not os.path.isdir(plotdir):
                os.makedirs(plotdir)

            # location of pickled percentile dtas data
            dctas_file = '%s/dc%s_%s_%s.pickle' % (datadir, varname, region, seas)

            # load data
            if not (os.path.isfile(dctas_file) and try_load) and not mmm:
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
                    tas_hist_file = Dataset(glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/historical/%s/%s_day_%s_historical_%s_*_%s.nc' % (proj, model, varname, model, par.ens['historical'][model], yr_span_hist))[0], 'r')
                    tas_rcp_file = Dataset(glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/ssp245/%s/%s_day_%s_ssp245_%s_*_%s.nc' % (proj, model, varname, model, par.ens['ssp245'][model], yr_span_rcp))[0], 'r')
                    huss_hist_file = Dataset(glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/historical/%s/%s_day_%s_historical_%s_*_%s.nc' % (proj, model, varname_q, model, par.ens['historical'][model], yr_span_hist))[0], 'r')
                    huss_rcp_file = Dataset(glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/ssp245/%s/%s_day_%s_ssp245_%s_*_%s.nc' % (proj, model, varname_q, model, par.ens['ssp245'][model], yr_span_rcp))[0], 'r')
                else:
                    tas_hist_file = Dataset(glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/historical/%s/%s_day_%s_historical_%s_*_%s.%s.nc' % (proj, model, varname, model, par.ens['historical'][model], yr_span_hist, seas))[0], 'r')
                    tas_rcp_file = Dataset(glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/ssp245/%s/%s_day_%s_ssp245_%s_*_%s.%s.nc' % (proj, model, varname, model, par.ens['ssp245'][model], yr_span_rcp, seas))[0], 'r')
                    huss_hist_file = Dataset(glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/historical/%s/%s_day_%s_historical_%s_*_%s.%s.nc' % (proj, model, varname_q, model, par.ens['historical'][model], yr_span_hist, seas))[0], 'r')
                    huss_rcp_file = Dataset(glob.glob('/project2/tas1/miyawaki/projects/%s/data/raw/ssp245/%s/%s_day_%s_ssp245_%s_*_%s.%s.nc' % (proj, model, varname_q, model, par.ens['ssp245'][model], yr_span_rcp, seas))[0], 'r')

                if varname == 'ts':
                    ts_hist = np.squeeze(ts_hist_file.variables['ts'][:])
                    ts_rcp = np.squeeze(ts_rcp_file.variables['ts'][:])
                    huss_hist = np.squeeze(huss_hist_file.variables['huss'][:])
                    huss_rcp = np.squeeze(huss_rcp_file.variables['huss'][:])
                else:
                    ts_hist = np.squeeze(ts_hist_file.variables['ta'][:])
                    ts_rcp = np.squeeze(ts_rcp_file.variables['ta'][:])
                    huss_hist = np.squeeze(huss_hist_file.variables['hus'][:])
                    huss_rcp = np.squeeze(huss_rcp_file.variables['hus'][:])

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

                # apply mask to ts
                ts_hist_l = ts_hist*lmask_hist
                ts_hist_o = ts_hist*omask_hist
                huss_hist_l = huss_hist*lmask_hist
                huss_hist_o = huss_hist*omask_hist
                # repeat for rcp
                ts_rcp_l = ts_rcp*lmask_rcp
                ts_rcp_o = ts_rcp*omask_rcp
                huss_rcp_l = huss_rcp*lmask_rcp
                huss_rcp_o = huss_rcp*omask_rcp

                # ts_hist=None; ts_rcp=None;

                ###########################
                # limit to specified region
                ###########################
                idx_trop = np.transpose(np.argwhere((grid['lat']>lat_lo) & (grid['lat']<lat_up)))
                lat_trop = np.squeeze(grid['lat'][idx_trop])

                lmask_hist_trop = np.squeeze(lmask_hist[idx_trop,:])
                ts_hist_trop = np.squeeze(ts_hist[:,idx_trop,:])
                ts_hist_l_trop = np.squeeze(ts_hist_l[:,idx_trop,:])
                ts_hist_o_trop = np.squeeze(ts_hist_o[:,idx_trop,:])
                huss_hist_trop = np.squeeze(huss_hist[:,idx_trop,:])
                huss_hist_l_trop = np.squeeze(huss_hist_l[:,idx_trop,:])
                huss_hist_o_trop = np.squeeze(huss_hist_o[:,idx_trop,:])

                lmask_rcp_trop = np.squeeze(lmask_rcp[idx_trop,:])
                ts_rcp_trop = np.squeeze(ts_rcp[:,idx_trop,:])
                ts_rcp_l_trop = np.squeeze(ts_rcp_l[:,idx_trop,:])
                ts_rcp_o_trop = np.squeeze(ts_rcp_o[:,idx_trop,:])
                huss_rcp_trop = np.squeeze(huss_rcp[:,idx_trop,:])
                huss_rcp_l_trop = np.squeeze(huss_rcp_l[:,idx_trop,:])
                huss_rcp_o_trop = np.squeeze(huss_rcp_o[:,idx_trop,:])

                if sample_method == 'latband':
                    ###########################
                    # sample percentiles for each latitude
                    ###########################
                    cts_hist = np.empty([ts_hist_trop.shape[1],len(prct)])
                    cts_hist_l = np.empty([ts_hist_l_trop.shape[1],len(prct)])
                    cts_hist_o = np.empty([ts_hist_o_trop.shape[1],len(prct)])
                    cts_rcp = np.empty([ts_rcp_trop.shape[1],len(prct)])
                    cts_rcp_l = np.empty([ts_rcp_l_trop.shape[1],len(prct)])
                    cts_rcp_o = np.empty([ts_rcp_o_trop.shape[1],len(prct)])
                    chuss_hist = np.empty([huss_hist_trop.shape[1],len(prct)])
                    chuss_hist_l = np.empty([huss_hist_l_trop.shape[1],len(prct)])
                    chuss_hist_o = np.empty([huss_hist_o_trop.shape[1],len(prct)])
                    chuss_rcp = np.empty([huss_rcp_trop.shape[1],len(prct)])
                    chuss_rcp_l = np.empty([huss_rcp_l_trop.shape[1],len(prct)])
                    chuss_rcp_o = np.empty([huss_rcp_o_trop.shape[1],len(prct)])
                    for ilat in range(ts_hist_l_trop.shape[1]):
                        # flatten arrays
                        ts_hist_flat   = np.ndarray.flatten(ts_hist_trop[:,ilat,:])
                        ts_hist_l_flat = np.ndarray.flatten(ts_hist_l_trop[:,ilat,:])
                        ts_hist_o_flat = np.ndarray.flatten(ts_hist_o_trop[:,ilat,:])
                        ts_rcp_flat    = np.ndarray.flatten(ts_rcp_trop[:,ilat,:])
                        ts_rcp_l_flat  = np.ndarray.flatten(ts_rcp_l_trop[:,ilat,:])
                        ts_rcp_o_flat  = np.ndarray.flatten(ts_rcp_o_trop[:,ilat,:])
                        huss_hist_flat   = np.ndarray.flatten(huss_hist_trop[:,ilat,:])
                        huss_hist_l_flat = np.ndarray.flatten(huss_hist_l_trop[:,ilat,:])
                        huss_hist_o_flat = np.ndarray.flatten(huss_hist_o_trop[:,ilat,:])
                        huss_rcp_flat    = np.ndarray.flatten(huss_rcp_trop[:,ilat,:])
                        huss_rcp_l_flat  = np.ndarray.flatten(huss_rcp_l_trop[:,ilat,:])
                        huss_rcp_o_flat  = np.ndarray.flatten(huss_rcp_o_trop[:,ilat,:])

                        # remove nans
                        ts_hist_flat = ts_hist_flat[~np.isnan(ts_hist_flat)]
                        ts_hist_l_flat = ts_hist_l_flat[~np.isnan(ts_hist_l_flat)]
                        ts_hist_o_flat = ts_hist_o_flat[~np.isnan(ts_hist_o_flat)]
                        ts_rcp_flat = ts_rcp_flat[~np.isnan(ts_rcp_flat)]
                        ts_rcp_l_flat = ts_rcp_l_flat[~np.isnan(ts_rcp_l_flat)]
                        ts_rcp_o_flat = ts_rcp_o_flat[~np.isnan(ts_rcp_o_flat)]
                        huss_hist_flat = huss_hist_flat[~np.isnan(huss_hist_flat)]
                        huss_hist_l_flat = huss_hist_l_flat[~np.isnan(huss_hist_l_flat)]
                        huss_hist_o_flat = huss_hist_o_flat[~np.isnan(huss_hist_o_flat)]
                        huss_rcp_flat = huss_rcp_flat[~np.isnan(huss_rcp_flat)]
                        huss_rcp_l_flat = huss_rcp_l_flat[~np.isnan(huss_rcp_l_flat)]
                        huss_rcp_o_flat = huss_rcp_o_flat[~np.isnan(huss_rcp_o_flat)]

                        # compute the percentile values
                        pts_hist = np.percentile(ts_hist_flat, prct, interpolation=imethod)
                        pts_hist_l = np.percentile(ts_hist_l_flat, prct, interpolation=imethod)
                        pts_hist_o = np.percentile(ts_hist_o_flat, prct, interpolation=imethod)
                        pts_rcp = np.percentile(ts_rcp_flat, prct, interpolation=imethod)
                        pts_rcp_l = np.percentile(ts_rcp_l_flat, prct, interpolation=imethod)
                        pts_rcp_o = np.percentile(ts_rcp_o_flat, prct, interpolation=imethod)
                        phuss_hist = np.percentile(huss_hist_flat, prct, interpolation=imethod)
                        phuss_hist_l = np.percentile(huss_hist_l_flat, prct, interpolation=imethod)
                        phuss_hist_o = np.percentile(huss_hist_o_flat, prct, interpolation=imethod)
                        phuss_rcp = np.percentile(huss_rcp_flat, prct, interpolation=imethod)
                        phuss_rcp_l = np.percentile(huss_rcp_l_flat, prct, interpolation=imethod)
                        phuss_rcp_o = np.percentile(huss_rcp_o_flat, prct, interpolation=imethod)

                       # take the cumulative mean (i.e. 0th cumulative percentile is the average of all days because any day exceeds the 0th percentile
                        for iprct in range(len(prct)):
                            cts_hist[ilat,iprct] = np.sum(ts_hist_flat[ts_hist_flat>pts_hist[iprct]])/len(ts_hist_flat[ts_hist_flat>pts_hist[iprct]])
                            cts_hist_l[ilat,iprct] = np.sum(ts_hist_l_flat[ts_hist_l_flat>pts_hist_l[iprct]])/len(ts_hist_l_flat[ts_hist_l_flat>pts_hist_l[iprct]])
                            cts_hist_o[ilat,iprct] = np.sum(ts_hist_o_flat[ts_hist_o_flat>pts_hist_o[iprct]])/len(ts_hist_o_flat[ts_hist_o_flat>pts_hist_o[iprct]])
                            cts_rcp[ilat,iprct] = np.sum(ts_rcp_flat[ts_rcp_flat>pts_rcp[iprct]])/len(ts_rcp_flat[ts_rcp_flat>pts_rcp[iprct]])
                            cts_rcp_l[ilat,iprct] = np.sum(ts_rcp_l_flat[ts_rcp_l_flat>pts_rcp_l[iprct]])/len(ts_rcp_l_flat[ts_rcp_l_flat>pts_rcp_l[iprct]])
                            cts_rcp_o[ilat,iprct] = np.sum(ts_rcp_o_flat[ts_rcp_o_flat>pts_rcp_o[iprct]])/len(ts_rcp_o_flat[ts_rcp_o_flat>pts_rcp_o[iprct]])
                            chuss_hist[ilat,iprct] = np.sum(huss_hist_flat[huss_hist_flat>phuss_hist[iprct]])/len(huss_hist_flat[huss_hist_flat>phuss_hist[iprct]])
                            chuss_hist_l[ilat,iprct] = np.sum(huss_hist_l_flat[huss_hist_l_flat>phuss_hist_l[iprct]])/len(huss_hist_l_flat[huss_hist_l_flat>phuss_hist_l[iprct]])
                            chuss_hist_o[ilat,iprct] = np.sum(huss_hist_o_flat[huss_hist_o_flat>phuss_hist_o[iprct]])/len(huss_hist_o_flat[huss_hist_o_flat>phuss_hist_o[iprct]])
                            chuss_rcp[ilat,iprct] = np.sum(huss_rcp_flat[huss_rcp_flat>phuss_rcp[iprct]])/len(huss_rcp_flat[huss_rcp_flat>phuss_rcp[iprct]])
                            chuss_rcp_l[ilat,iprct] = np.sum(huss_rcp_l_flat[huss_rcp_l_flat>phuss_rcp_l[iprct]])/len(huss_rcp_l_flat[huss_rcp_l_flat>phuss_rcp_l[iprct]])
                            chuss_rcp_o[ilat,iprct] = np.sum(huss_rcp_o_flat[huss_rcp_o_flat>phuss_rcp_o[iprct]])/len(huss_rcp_o_flat[huss_rcp_o_flat>phuss_rcp_o[iprct]])

                elif sample_method == 'gridpt':
                    ###########################
                    # sample percentiles for each latitude and longitude
                    ###########################
                    cts_hist = np.empty(  [ts_hist_trop.shape[1],  ts_hist_trop.shape[2],  len(prct)])
                    cts_hist_l = np.empty([ts_hist_l_trop.shape[1],ts_hist_l_trop.shape[2],len(prct)])
                    cts_hist_o = np.empty([ts_hist_o_trop.shape[1],ts_hist_o_trop.shape[2],len(prct)])
                    cts_rcp = np.empty(   [ts_rcp_trop.shape[1],   ts_rcp_trop.shape[2],   len(prct)])
                    cts_rcp_l = np.empty( [ts_rcp_l_trop.shape[1], ts_rcp_l_trop.shape[2], len(prct)])
                    cts_rcp_o = np.empty( [ts_rcp_o_trop.shape[1], ts_rcp_o_trop.shape[2], len(prct)])
                    chuss_hist = np.empty(  [huss_hist_trop.shape[1],  huss_hist_trop.shape[2],  len(prct)])
                    chuss_hist_l = np.empty([huss_hist_l_trop.shape[1],huss_hist_l_trop.shape[2],len(prct)])
                    chuss_hist_o = np.empty([huss_hist_o_trop.shape[1],huss_hist_o_trop.shape[2],len(prct)])
                    chuss_rcp = np.empty(   [huss_rcp_trop.shape[1],   huss_rcp_trop.shape[2],   len(prct)])
                    chuss_rcp_l = np.empty( [huss_rcp_l_trop.shape[1], huss_rcp_l_trop.shape[2], len(prct)])
                    chuss_rcp_o = np.empty( [huss_rcp_o_trop.shape[1], huss_rcp_o_trop.shape[2], len(prct)])
                    for ilat in tqdm(range(ts_hist_l_trop.shape[1])):
                        for ilon in range(ts_hist_l_trop.shape[2]):
                            # determine if this is a land or ocean grid
                            island_hist = ( lmask_hist_trop[ilat,ilon]==1 )
                            island_rcp = ( lmask_rcp_trop[ilat,ilon]==1 )

                            # select data at grid point 
                            ts_hist_ll   = ts_hist_trop[:,  ilat, ilon]
                            ts_rcp_ll    = ts_rcp_trop[:   ,ilat, ilon]
                            huss_hist_ll   = huss_hist_trop[:,  ilat, ilon]
                            huss_rcp_ll    = huss_rcp_trop[:   ,ilat, ilon]

                            if island_hist:
                                ts_hist_l_ll = ts_hist_l_trop[:,ilat, ilon]
                                cts_hist_o[ilat, ilon, :] = np.nan*np.ones([len(prct)])
                                huss_hist_l_ll = huss_hist_l_trop[:,ilat, ilon]
                                chuss_hist_o[ilat, ilon, :] = np.nan*np.ones([len(prct)])
                            else:
                                ts_hist_o_ll = ts_hist_o_trop[:,ilat, ilon]
                                cts_hist_l[ilat, ilon, :] = np.nan*np.ones([len(prct)])
                                huss_hist_o_ll = huss_hist_o_trop[:,ilat, ilon]
                                chuss_hist_l[ilat, ilon, :] = np.nan*np.ones([len(prct)])

                            if island_rcp:
                                ts_rcp_l_ll  = ts_rcp_l_trop[: ,ilat, ilon]
                                cts_rcp_o[ilat, ilon, :] = np.nan*np.ones([len(prct)])
                                huss_rcp_l_ll  = huss_rcp_l_trop[: ,ilat, ilon]
                                chuss_rcp_o[ilat, ilon, :] = np.nan*np.ones([len(prct)])
                            else:
                                ts_rcp_o_ll  = ts_rcp_o_trop[: ,ilat, ilon]
                                cts_rcp_l[ilat, ilon, :] = np.nan*np.ones([len(prct)])
                                huss_rcp_o_ll  = huss_rcp_o_trop[: ,ilat, ilon]
                                chuss_rcp_l[ilat, ilon, :] = np.nan*np.ones([len(prct)])

                            # remove nans
                            ts_hist_ll = ts_hist_ll[~np.isnan(ts_hist_ll)]
                            ts_rcp_ll = ts_rcp_ll[~np.isnan(ts_rcp_ll)]
                            huss_hist_ll = huss_hist_ll[~np.isnan(huss_hist_ll)]
                            huss_rcp_ll = huss_rcp_ll[~np.isnan(huss_rcp_ll)]
                            if island_hist:
                                ts_hist_l_ll = ts_hist_l_ll[~np.isnan(ts_hist_l_ll)]
                                huss_hist_l_ll = huss_hist_l_ll[~np.isnan(huss_hist_l_ll)]
                            else:
                                ts_hist_o_ll = ts_hist_o_ll[~np.isnan(ts_hist_o_ll)]
                                huss_hist_o_ll = huss_hist_o_ll[~np.isnan(huss_hist_o_ll)]
                            if island_rcp:
                                ts_rcp_l_ll = ts_rcp_l_ll[~np.isnan(ts_rcp_l_ll)]
                                huss_rcp_l_ll = huss_rcp_l_ll[~np.isnan(huss_rcp_l_ll)]
                            else:
                                ts_rcp_o_ll = ts_rcp_o_ll[~np.isnan(ts_rcp_o_ll)]
                                huss_rcp_o_ll = huss_rcp_o_ll[~np.isnan(huss_rcp_o_ll)]


                            # compute the percentile values
                            pts_hist = np.percentile(ts_hist_ll, prct, interpolation=imethod)
                            pts_rcp = np.percentile(ts_rcp_ll, prct, interpolation=imethod)
                            phuss_hist = np.percentile(huss_hist_ll, prct, interpolation=imethod)
                            phuss_rcp = np.percentile(huss_rcp_ll, prct, interpolation=imethod)

                            if island_hist:
                                pts_hist_l = np.percentile(ts_hist_l_ll, prct, interpolation=imethod)
                                phuss_hist_l = np.percentile(huss_hist_l_ll, prct, interpolation=imethod)
                            else:
                                pts_hist_o = np.percentile(ts_hist_o_ll, prct, interpolation=imethod)
                                phuss_hist_o = np.percentile(huss_hist_o_ll, prct, interpolation=imethod)
                            if island_rcp:
                                pts_rcp_l = np.percentile(ts_rcp_l_ll, prct, interpolation=imethod)
                                phuss_rcp_l = np.percentile(huss_rcp_l_ll, prct, interpolation=imethod)
                            else:
                                pts_rcp_o = np.percentile(ts_rcp_o_ll, prct, interpolation=imethod)
                                phuss_rcp_o = np.percentile(huss_rcp_o_ll, prct, interpolation=imethod)

                           # take the cumulative mean (i.e. 0th cumulative percentile is the average of all days because any day exceeds the 0th percentile
                            for iprct in range(len(prct)):
                                cts_hist[  ilat,ilon,iprct] = np.sum(ts_hist_ll[ts_hist_ll>pts_hist[iprct]])/len(ts_hist_ll[ts_hist_ll>pts_hist[iprct]])
                                cts_rcp[   ilat,ilon,iprct] = np.sum(ts_rcp_ll[ts_rcp_ll>pts_rcp[iprct]])/len(ts_rcp_ll[ts_rcp_ll>pts_rcp[iprct]])
                                chuss_hist[  ilat,ilon,iprct] = np.sum(huss_hist_ll[huss_hist_ll>phuss_hist[iprct]])/len(huss_hist_ll[huss_hist_ll>phuss_hist[iprct]])
                                chuss_rcp[   ilat,ilon,iprct] = np.sum(huss_rcp_ll[huss_rcp_ll>phuss_rcp[iprct]])/len(huss_rcp_ll[huss_rcp_ll>phuss_rcp[iprct]])

                            if island_hist:
                                for iprct in range(len(prct)):
                                    cts_hist_l[ilat,ilon,iprct] = np.sum(ts_hist_l_ll[ts_hist_l_ll>pts_hist_l[iprct]])/len(ts_hist_l_ll[ts_hist_l_ll>pts_hist_l[iprct]])
                                    chuss_hist_l[ilat,ilon,iprct] = np.sum(huss_hist_l_ll[huss_hist_l_ll>phuss_hist_l[iprct]])/len(huss_hist_l_ll[huss_hist_l_ll>phuss_hist_l[iprct]])
                            else:
                                for iprct in range(len(prct)):
                                    cts_hist_o[ilat,ilon,iprct] = np.sum(ts_hist_o_ll[ts_hist_o_ll>pts_hist_o[iprct]])/len(ts_hist_o_ll[ts_hist_o_ll>pts_hist_o[iprct]])
                                    chuss_hist_o[ilat,ilon,iprct] = np.sum(huss_hist_o_ll[huss_hist_o_ll>phuss_hist_o[iprct]])/len(huss_hist_o_ll[huss_hist_o_ll>phuss_hist_o[iprct]])
                            if island_rcp:
                                for iprct in range(len(prct)):
                                    cts_rcp_l[ ilat,ilon,iprct] = np.sum(ts_rcp_l_ll[ts_rcp_l_ll>pts_rcp_l[iprct]])/len(ts_rcp_l_ll[ts_rcp_l_ll>pts_rcp_l[iprct]])
                                    chuss_rcp_l[ ilat,ilon,iprct] = np.sum(huss_rcp_l_ll[huss_rcp_l_ll>phuss_rcp_l[iprct]])/len(huss_rcp_l_ll[huss_rcp_l_ll>phuss_rcp_l[iprct]])
                            else:
                                for iprct in range(len(prct)):
                                    cts_rcp_o[ ilat,ilon,iprct] = np.sum(ts_rcp_o_ll[ts_rcp_o_ll>pts_rcp_o[iprct]])/len(ts_rcp_o_ll[ts_rcp_o_ll>pts_rcp_o[iprct]])
                                    chuss_rcp_o[ ilat,ilon,iprct] = np.sum(huss_rcp_o_ll[huss_rcp_o_ll>phuss_rcp_o[iprct]])/len(huss_rcp_o_ll[huss_rcp_o_ll>phuss_rcp_o[iprct]])



                ############################
                ## take area weighted mean of the percentiles
                ############################
                #clat = np.transpose(np.cos(np.deg2rad(grid['lat'][idx_trop])))
                #cts_hist_l_areaavg = np.nansum(clat * cts_hist_l, axis=0) / np.nansum(clat)
                #cts_hist_o_areaavg = np.nansum(clat * cts_hist_o, axis=0) / np.nansum(clat)
                #cts_rcp_l_areaavg = np.nansum(clat * cts_rcp_l, axis=0) / np.nansum(clat)
                #cts_rcp_o_areaavg = np.nansum(clat * cts_rcp_o, axis=0) / np.nansum(clat)

                ############################
                ## warming as a function of percentile
                ############################
                #dcts_l = cts_rcp_l_areaavg - cts_hist_l_areaavg
                #dcts_o = cts_rcp_o_areaavg - cts_hist_o_areaavg

                ############################
                ## compute saturation specific humidity
                ############################
                #cqsat_hist_l = e2q(pref, comp_esat(cts_hist_l))
                #cqsat_hist_o = e2q(pref, comp_esat(cts_hist_o))
                #cqsat_rcp_l = e2q(pref, comp_esat(cts_rcp_l))
                #cqsat_rcp_o = e2q(pref, comp_esat(cts_rcp_o))

                ############################
                ## compute pseudo relative humidity
                ############################
                #crh_hist_l = chuss_hist_l/cqsat_hist_l
                #crh_hist_o = chuss_hist_o/cqsat_hist_o
                #crh_rcp_l = chuss_rcp_l/cqsat_rcp_l
                #crh_rcp_o = chuss_rcp_o/cqsat_rcp_o

                ###########################
                # mean humidity
                ###########################
                if sample_method == 'latband':
                    chuss_hist_l_mean = np.nanmean(huss_hist_l_trop, axis=(0,2)) 
                    chuss_hist_o_mean = np.nanmean(huss_hist_o_trop, axis=(0,2)) 
                elif sample_method == 'gridpt':
                    chuss_hist_l_mean = np.nanmean(huss_hist_l_trop, axis=(0))
                    chuss_hist_o_mean = np.nanmean(huss_hist_o_trop, axis=(0))

                ############################
                ## mean saturation humidity
                ############################
                #if sample_method == 'latband':
                #    cqsat_hist_l_mean = np.nanmean(qsat_hist_l_trop, axis=(0,2)) 
                #    cqsat_hist_o_mean = np.nanmean(qsat_hist_o_trop, axis=(0,2)) 
                #elif sample_method == 'gridpt':
                #    cqsat_hist_l_mean = np.nanmean(qsat_hist_l_trop, axis=(0))
                #    cqsat_hist_o_mean = np.nanmean(qsat_hist_o_trop, axis=(0))

                ###########################
                # warming as a function of percentile
                ###########################
                dcts_l = cts_rcp_l - cts_hist_l
                dcts_o = cts_rcp_o - cts_hist_o

                ############################
                ## moistening as a function of percentile
                ############################
                #dcrh_l = crh_rcp_l - crh_hist_l
                #dcrh_o = crh_rcp_o - crh_hist_o
                #dchuss_l = chuss_rcp_l - chuss_hist_l
                #dchuss_o = chuss_rcp_o - chuss_hist_o

                ###########################
                # mean warming
                ###########################
                if sample_method == 'latband':
                    dcts_l_mean = np.nanmean(ts_rcp_l_trop, axis=(0,2)) - np.nanmean(ts_hist_l_trop, axis=(0,2))
                    dcts_o_mean = np.nanmean(ts_rcp_o_trop, axis=(0,2)) - np.nanmean(ts_hist_o_trop, axis=(0,2))
                elif sample_method == 'gridpt':
                    dcts_l_mean = np.nanmean(ts_rcp_l_trop, axis=(0)) - np.nanmean(ts_hist_l_trop, axis=(0))
                    dcts_o_mean = np.nanmean(ts_rcp_o_trop, axis=(0)) - np.nanmean(ts_hist_o_trop, axis=(0))

                ############################
                ## mean moistening
                ############################
                #if sample_method == 'latband':
                #    dcrh_l_mean = np.nanmean(rh_rcp_l_trop, axis=(0,2)) - np.nanmean(rh_hist_l_trop, axis=(0,2))
                #    dcrh_o_mean = np.nanmean(rh_rcp_o_trop, axis=(0,2)) - np.nanmean(rh_hist_o_trop, axis=(0,2))
                #    dchuss_l_mean = np.nanmean(huss_rcp_l_trop, axis=(0,2)) - np.nanmean(huss_hist_l_trop, axis=(0,2))
                #    dchuss_o_mean = np.nanmean(huss_rcp_o_trop, axis=(0,2)) - np.nanmean(huss_hist_o_trop, axis=(0,2))
                #elif sample_method == 'gridpt':
                #    dcrh_l_mean = np.nanmean(rh_rcp_l_trop, axis=(0)) - np.nanmean(rh_hist_l_trop, axis=(0))
                #    dcrh_o_mean = np.nanmean(rh_rcp_o_trop, axis=(0)) - np.nanmean(rh_hist_o_trop, axis=(0))
                #    dchuss_l_mean = np.nanmean(huss_rcp_l_trop, axis=(0)) - np.nanmean(huss_hist_l_trop, axis=(0))
                #    dchuss_o_mean = np.nanmean(huss_rcp_o_trop, axis=(0)) - np.nanmean(huss_hist_o_trop, axis=(0))

                ############################
                ## theoretical prediction (eq 5 in byrne2021)
                ############################
                #ep = par.Lv * al_l * cqsat_hist_l / ( par.cp + par.Lv * al_l * cqsat_hist_l )
                #ga_to = ( par.cp + par.Lv * al_o * chuss_hist_o_mean ) / ( par.cp + par.Lv * al_l * chuss_hist_l )
                #ga_ro = ( par.Lv * cqsat_hist_o_mean ) / ( par.cp + par.Lv * al_l * chuss_hist_l )
                #et = ( cqsat_hist_l_mean / cqsat_hist_l ) * ( ep / al_l )
                ## dcts_l_pred = 1/(1+ep*dcrh_l_mean)

                ###########################
                # scaling factor
                ###########################
                if sample_method == 'latband':
                    dcsf_l = dcts_l / np.expand_dims(dcts_l_mean, axis=1)
                    dcsf_o = dcts_o / np.expand_dims(dcts_o_mean, axis=1)
                elif sample_method == 'gridpt':
                    dcsf_l = dcts_l / np.expand_dims(dcts_l_mean, axis=2)
                    dcsf_o = dcts_o / np.expand_dims(dcts_o_mean, axis=2)

                ###########################
                # if computed for each grid point, take zonal mean now
                ###########################
                if sample_method == 'gridpt':
                    dcts_l = np.nanmean(dcts_l, axis=1)
                    dcts_o = np.nanmean(dcts_o, axis=1)
                    dcsf_l = np.nanmean(dcsf_l, axis=1)
                    dcsf_o = np.nanmean(dcsf_o, axis=1)

                ###########################
                # take area weighted mean of the percentiles
                ###########################
                clat = np.transpose(np.cos(np.deg2rad(grid['lat'][idx_trop])))
                dcts_l = np.nansum(clat * dcts_l, axis=0) / np.nansum(clat)
                dcts_o = np.nansum(clat * dcts_o, axis=0) / np.nansum(clat)
                dcsf_l = np.nansum(clat * dcsf_l, axis=0) / np.nansum(clat)
                dcsf_o = np.nansum(clat * dcsf_o, axis=0) / np.nansum(clat)

                ###########################
                # save data
                ###########################
                pickle.dump([dcts_l, dcts_o, dcsf_l, dcsf_o], open(dcts_file, 'wb'))

            # load pickled data
            if mmm:
                dcts_l_agg = np.empty([len(prct), len(modellist)])
                dcts_o_agg = np.empty([len(prct), len(modellist)])
                dcsf_l_agg = np.empty([len(prct), len(modellist)])
                dcsf_o_agg = np.empty([len(prct), len(modellist)])
                for imodel in tqdm(range(len(modellist))):
                    currentmodel = modellist[imodel]

                    datadir0 = '/project2/tas1/miyawaki/projects/000_hotdays/data/proc/%s/%s/%s/%s_%g_%g' % (currentmodel, sample_method, varname, region, lat_lo, lat_up)
                    dcts_file0 = '%s/dcts_%s_%s.pickle' % (datadir0, region, seas)

                    [dcts_l_agg[:,imodel], dcts_o_agg[:,imodel], dcsf_l_agg[:,imodel], dcsf_o_agg[:,imodel]] = pickle.load(open(dcts_file0, 'rb'))

                # take multimodel mean
                dcts_l = np.mean(dcts_l_agg,1)
                dcts_o = np.mean(dcts_o_agg,1)
                dcsf_l = np.mean(dcsf_l_agg,1)
                dcsf_o = np.mean(dcsf_o_agg,1)

                # compute 25th and 75th percentiles
                dcts_l_mmm = {}
                dcts_l_mmm['prc25'] = np.percentile(dcts_l_agg, 25, axis=1)
                dcts_l_mmm['prc75'] = np.percentile(dcts_l_agg, 75, axis=1)
                dcts_o_mmm = {}
                dcts_o_mmm['prc25'] = np.percentile(dcts_o_agg, 25, axis=1)
                dcts_o_mmm['prc75'] = np.percentile(dcts_o_agg, 75, axis=1)

                dcsf_l_mmm = {}
                dcsf_l_mmm['prc25'] = np.percentile(dcsf_l_agg, 25, axis=1)
                dcsf_l_mmm['prc75'] = np.percentile(dcsf_l_agg, 75, axis=1)
                dcsf_o_mmm = {}
                dcsf_o_mmm['prc25'] = np.percentile(dcsf_o_agg, 25, axis=1)
                dcsf_o_mmm['prc75'] = np.percentile(dcsf_o_agg, 75, axis=1)

            else:
                [dcts_l, dcts_o, dcsf_l, dcsf_o] = pickle.load(open(dcts_file, 'rb'))

            ###########################
            # plot dcts
            ###########################
            if seas is None:
                plotname = '%s/dcts' % (plotdir)
            else:
                plotname = '%s/dcts.%s' % (plotdir, seas)
            fig, ax = plt.subplots()
            vmin = 0
            vmax = 100
            if model == 'mmm':
                ax.fill_between(prct, dcts_l_mmm['prc25'], dcts_l_mmm['prc75'], facecolor='r', edgecolor=None, alpha=0.1)
                ax.fill_between(prct, dcts_o_mmm['prc25'], dcts_o_mmm['prc75'], facecolor='b', edgecolor=None, alpha=0.1)
            ax.plot(prct, dcts_l, '-r', label='Land')
            ax.plot(prct, dcts_o, '-b', label='ocean')
            ax.set_title(titlestr)
            # make_title_sim_time(ax, sim, model=model, timemean=timemean)
            ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
            # if 'ymonmean' in timemean:
            #     ax.set_xticks(np.arange(0,12,1))
            #     ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            # else:
            ax.set_xlim(0,100)
            ax.set_xlabel('Percentile')
            ax.set_ylabel('$\delta T^x$ (K)')
            # ax.set_yticks(np.arange(-90,91,30))
            # ax.xaxis.set_minor_locator(MultipleLocator(10))
            # ax.yaxis.set_minor_locator(MultipleLocator(10))
            fig.set_size_inches(5, 4)
            plt.savefig('%s.pdf' % (plotname), format='pdf', dpi=300)
            plt.close()

            ###########################
            # plot dcsf
            ###########################
            if seas is None:
                plotname = '%s/sf' % (plotdir)
            else:
                plotname = '%s/sf.%s' % (plotdir, seas)
            fig, ax = plt.subplots()
            vmin = 0
            vmax = 100
            if model == 'mmm':
                ax.fill_between(prct, dcsf_l_mmm['prc25'], dcsf_l_mmm['prc75'], facecolor='r', edgecolor=None, alpha=0.1)
                ax.fill_between(prct, dcsf_o_mmm['prc25'], dcsf_o_mmm['prc75'], facecolor='b', edgecolor=None, alpha=0.1)
            ax.plot(prct, dcsf_l, '-r', label='Land')
            ax.plot(prct, dcsf_o, '-b', label='ocean')
            ax.set_title(titlestr)
            # make_title_sim_time(ax, sim, model=model, timemean=timemean)
            ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
            # if 'ymonmean' in timemean:
            #     ax.set_xticks(np.arange(0,12,1))
            #     ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            # else:
            ax.set_xlim(0,100)
            ax.set_xlabel('Percentile')
            ax.set_ylabel('$\delta T^x / \delta \overline{T}$ (1)')
            # ax.set_yticks(np.arange(-90,91,30))
            # ax.xaxis.set_minor_locator(MultipleLocator(10))
            # ax.yaxis.set_minor_locator(MultipleLocator(10))
            fig.set_size_inches(5, 4)
            plt.savefig('%s.pdf' % (plotname), format='pdf', dpi=300)
            plt.close()

        # end seas loop

    # end region loop

    # stop after one loop if multimodel mean
    if mmm:
        break

# end model loop

