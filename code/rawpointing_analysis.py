#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 11:13:19 2025

@author: mrutala
"""

import pandas as pd
import glob
import astropy
from astropy.time import Time
import sys
import datetime
import numpy as np
import os
sys.path.append(os.getcwd() + '/Chandra_X_Ray_data_processing_pipeline/')
import go_chandra_analysis_tools as gca_tools
import go_chandra as gc

import matplotlib.pyplot as plt
import matplotlib as mpl
try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

# Setup paths for chandra files
rawpath_format = '/Users/mrutala/projects/Jupiter_XUV_Comparison/data/CXO/{0}/primary/hrcf{0}N001_evt2.fits'
ditherpath_format = '/Users/mrutala/projects/Jupiter_XUV_Comparison/data/CXO/{0}/primary/pcadf{0}_???N???_asol1.fits'
fovpath_format = '/Users/mrutala/projects/Jupiter_XUV_Comparison/data/CXO/{0}/primary/hrcf{0}_???N???_fov1.fits'
alleventspath_format = '/Users/mrutala/projects/Jupiter_XUV_Comparison/data/CXO/{0}/primary/{0}_selected_region_ellipse.csv'
obs_ids = ['29673', '29674', '29675', '29676']

# Read chandra photon event lists & fits headers
# Each observation gets one dictionary item, with a DataFrame ()
observations = {}
for obs_id in obs_ids:
    observations[obs_id] = {}
    
    # Read the raw data
    with astropy.io.fits.open(glob.glob(rawpath_format.format(obs_id))[0]) as hdul:
        observations[obs_id]['raw_data'] = hdul[1].data
        observations[obs_id]['raw_header'] =  hdul[1].header
        
    # Read the dither information
    with astropy.io.fits.open(glob.glob(ditherpath_format.format(obs_id))[0]) as hdul:
        observations[obs_id]['dither_header'] =  hdul[1].header
        observations[obs_id]['dither_data'] = hdul[1].data
        
    # Read the FOV information
    with astropy.io.fits.open(glob.glob(fovpath_format.format(obs_id))[0]) as hdul:
        observations[obs_id]['fov_header'] = hdul[1].header
        observations[obs_id]['fov_data'] = hdul[1].data
        
    observations[obs_id]['all_events'] = pd.read_csv(alleventspath_format.format(obs_id), comment='#', sep=',')
        
    # Read ephemerides
    start = observations[obs_id]['raw_header']['MJD-OBS']
    exptime =  (observations[obs_id]['raw_header']['TSTOP'] - \
                observations[obs_id]['raw_header']['TSTART']) / (24 * 3600)
    
    eph = gca_tools.fetch_ephemerides_fromCXO(Time(start, format='mjd'), 
                                              Time(start + exptime, format='mjd'),
                                              '10m')
    observations[obs_id]['full_eph'] = eph

# %%

# Inspect the dither patterns w.r.t. Jupiter's position
for obs_id, data in observations.items():
    
    breakpoint()
    
    # In pixel coords, un-rotate the FOV regions to assign thin/thick locations
    x, y = data['fov_data']['x'], data['fov_data']['y']
    x0, y0 = data['fov_header']['TCRPX1'], data['fov_header']['TCRPX2']
    # Sky angles are measured backwards
    a = - np.deg2rad(data['fov_header']['ROLL_PNT'])
    # But we want to de-rotate, so take the negative of the negative angle
    x_rot = (x-x0)*np.cos(-a) - (y-y0)*np.sin(-a) #+ x0
    y_rot = (x-x0)*np.sin(-a) + (y-y0)*np.cos(-a) #+ y0
    # x_scl = x_rot * data['fov_header']['TCDLT1']
    # y_scl = y_rot * data['fov_header']['TCDLT2']
    
    # So, the 3 detectors are arranged [3, 2, 1], and consist of parts:
    # ___________________________________
    #|_____5_____|___  1  ___|_____6_____|
    #|     3     | 2 |   | 2 |     4     |
    #|___________|___|___|___|___________|
    #      -1        center        +1    
    # 
    height = 18.6 # mm
    width_center = 100 # mm
    # al1_yrange = [height/2 + 4, height]
    
    # dispersion = 1.148 # Dispersion in Angstrom/mm    
    # From the left (negative)
    xbound1 = 0
    xbound2 = 53 # Angstrom 
    xbound3 = (53 + 19.1 + 17.1)
    xbound4 = (53 + 19.1 + 17.1 + 64) 
    
    # In principle, this chip is only 100 mm long... but whatever
    # For the center chip, the aluminum region should span:
    c1_xrange = [np.nanmin(x_rot[1]), np.nanmax(x_rot[1])]
    c1_yrange = [np.nanmin(y_rot[1]), np.nanmax(y_rot[1])]
    
    al1_xrange = [(xbound2 / xbound4) * (c1_xrange[1] - c1_xrange[0]) + c1_xrange[0], 
                  (xbound3 / xbound4) * (c1_xrange[1] - c1_xrange[0]) + c1_xrange[0]]
    al1_yrange = [(height/2 + 4)/height * (c1_yrange[1] - c1_yrange[0]) + c1_yrange[0],
                  1.0 * (c1_yrange[1] - c1_yrange[0]) + c1_yrange[0]]
    
    # Finally, the region should be
    al1_region = np.array([[c1_xrange[0],  al1_yrange[0]],
                           [c1_xrange[0],  al1_yrange[1]],
                           [c1_xrange[1],  al1_yrange[1]],
                           [c1_xrange[1],  al1_yrange[0]],
                           [al1_xrange[1], al1_yrange[0]],
                           [al1_xrange[1], c1_yrange[0]],
                           [al1_xrange[0], c1_yrange[0]],
                           [al1_xrange[0], al1_yrange[0]],
                           [c1_xrange[0], al1_yrange[0]]
                           ])
    
    # Finally, re-rotate the region into chip coords
    xr_inframe = (al1_region[:,0]-0)*np.cos(a) - (al1_region[:,1]-0)*np.sin(a) + x0
    yr_inframe = (al1_region[:,0]-0)*np.sin(a) + (al1_region[:,1]-0)*np.cos(a) + y0
           
    # Now convert all detector coords to sky coords
    def chip_to_sky(X, Y):
        ref_ra, ref_dec = data['fov_header']['TCRVL1'], data['fov_header']['TCRVL2']
        ref_x, ref_y = data['fov_header']['TCRPX1'], data['fov_header']['TCRPX2']
        scale = np.abs(data['fov_header']['TCDLT1'])
        return (X-ref_x)*scale+ref_ra, (Y-ref_y)*scale+ref_dec
    
    # Visualize  
    fig, ax = plt.subplots(figsize=(3,3))
    plt.subplots_adjust(left=0.20, bottom=0.15)
    
    for i, label in zip([0, 1, 2], ['-1', 'Center', '+1']):
        fov_x, fov_y = chip_to_sky(data['fov_data']['x'][i], data['fov_data']['y'][i])
        ax.plot(fov_x, fov_y, color='black', lw=1)
        ax.annotate("{}".format(label), 
                    (np.nanmean(fov_x), np.nanmean(fov_y)),
                    (5, 0), 
                    xycoords='data', textcoords='offset fontsize', 
                    ha='center', va='center')
        
    # Center chip aluminum
    # ax.plot(*chip_to_sky(xr_inframe, yr_inframe), color='xkcd:steel', lw=1)
    al1 = mpl.patches.Polygon(np.array(chip_to_sky(xr_inframe, yr_inframe)).T, color='xkcd:steel', alpha=0.5)
    ax.add_patch(al1)
    
    
    dither_jd = Time(data['dither_data'].time, format='cxcsec').jd
    target_ra = np.interp(dither_jd, data['full_eph']['datetime_jd'], data['full_eph']['RA'])
    target_dec = np.interp(dither_jd, data['full_eph']['datetime_jd'], data['full_eph']['DEC'])
    
    # # What are the SIM offsets in dither_header? Are these the chip offsets in mm?
    ra_cxo, dec_cxo = data['dither_header']['RA_PNT'], data['dither_header']['DEC_PNT']
    
    # ra_offsets = (target_ra - ra_cxo).mean()
    # dec_offsets = (target_dec - dec_cxo).mean()
    
    # # 
    # ax.scatter(data['dither_data'].ra, data['dither_data'].dec, 
    #            label = 'Dither Pattern', marker='.', s=2, color='black')
    
    
    
    
    # ax.plot(target_ra, target_dec, lw=2, color='C0', label = 'Jupiter')
    
    # Normalize the dither, then add it to Jupiter's position
    dither_norm_ra = data['dither_data'].ra - ra_cxo
    dither_norm_dec = data['dither_data'].dec - dec_cxo
    
    target_with_dither_x = target_ra + dither_norm_ra
    target_with_dither_y = target_dec + dither_norm_dec
    
    ax.plot(target_with_dither_x, target_with_dither_y, 
            color='xkcd:red', lw=0.5)
    
    # Make it prety
    ax.set(aspect=1, 
           xlim = [target_with_dither_x.max()+0.1, target_with_dither_x.min()-0.1],
           ylim = [target_with_dither_y.min()-0.1, target_with_dither_y.max()+0.1],
           xlabel = 'RA [deg.]', ylabel = 'Dec. [deg.]', 
           title = "Observation: {}".format(obs_id))
    plt.show()


#%%
# Extract background annulus around Jupiter, normalize, and compare counts

# Segment the image into 5" x 5" bins
for obs_id, obs in observations.items():
    x_lims = np.ceil(obs['all_events']['x'].min()), np.floor(obs['all_events']['x'].max())
    y_lims = np.ceil(obs['all_events']['y'].min()), np.floor(obs['all_events']['y'].max())
    x_bins = np.arange(*x_lims, 5)
    y_bins = np.arange(*y_lims, 5)
    
    hist_2d, _, _ = np.histogram2d(obs['all_events']['x'], obs['all_events']['y'],
                                   bins=(x_bins, y_bins))    
    
    
    # Check each corner of each bin for overlap with Jupiter
    grid_lower_left = np.meshgrid(x_bins[:-1], y_bins[:-1], indexing='ij')
    grid_upper_left = np.meshgrid(x_bins[:-1], y_bins[1:], indexing='ij')
    grid_lower_right = np.meshgrid(x_bins[1:], y_bins[:-1], indexing='ij')
    grid_upper_right = np.meshgrid(x_bins[1:], y_bins[1:], indexing='ij')
    inside_limb = np.zeros((len(x_bins)-1, len(y_bins)-1))
    # Need a new patch, otherwise it's in axis coords
    jupiter_patch = gc.get_JupiterPatch(obs['full_eph'])
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            
            # Check each corner for being inside the limb
            ll = jupiter_patch.contains_point((grid_lower_left[0][i,j], grid_lower_left[1][i,j]))
            ul = jupiter_patch.contains_point((grid_upper_left[0][i,j], grid_upper_left[1][i,j]))
            lr = jupiter_patch.contains_point((grid_lower_right[0][i,j], grid_lower_right[1][i,j]))
            ur = jupiter_patch.contains_point((grid_upper_right[0][i,j], grid_upper_right[1][i,j]))
            
            if any([ll, ul, lr, ur]):
                inside_limb[i,j] = 1
                
    # Get the background statistics from outside the limb
    # Not a Gaussian, but whatever
    mean_total_bg = hist_2d.T[inside_limb == 0].mean()/((5/3600)**2)
    std_total_bg = hist_2d.T[inside_limb == 0].std()/((5/3600)**2)
    
    # Convert to counts per second
    exptime = obs['raw_header']['TSTOP'] - obs['raw_header']['TSTART']
    mean_bg = mean_total_bg / exptime
    std_bg = std_total_bg / exptime
    
    # Show the binned image + jupiter's limb
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(x_bins, y_bins, hist_2d.T/exptime, vmin=0, vmax=0.005)
    fig.colorbar(pcm, ax=ax, label="Counts/s")
    
    jupiter_patch = gc.get_JupiterPatch(obs['full_eph'], color='white', alpha=0.5)
    ax.add_patch(jupiter_patch)
    ax.set(aspect=1)
    title = "Obs ID: {}".format(obs_id) + '\n' + r"Background: {:.02f} $\pm$ {:.02f} counts/sq. deg./s".format(mean_bg, std_bg)
    ax.set_title(title)
    
    plt.show()