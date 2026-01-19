#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 11:13:19 2025

@author: mrutala
"""

import pandas as pd
import tqdm
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
import shapely

from shapely import LineString, Point, Polygon

import matplotlib.pyplot as plt
import matplotlib as mpl
try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

# Setup paths for chandra files
base = '/Users/mrutala/projects/Jupiter_XUV_Comparison/data/CXO'
rawpath_format              = base + '/{0}/primary/hrcf{0}N001_evt2.fits'
ditherpath_format           = base + '/{0}/primary/pcadf{0}_???N???_asol1.fits'
fovpath_format              = base + '/{0}/primary/hrcf{0}_???N???_fov1.fits'
alleventspath_format        = base + '/{0}/primary/{0}_selected_region_ellipse.csv'
filteredeventspath_format   = base + '/{0}/primary/{0}_photonlist_filtered_ellipse.csv'
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
    observations[obs_id]['filtered_events'] = pd.read_csv(filteredeventspath_format.format(obs_id), comment='#', sep=',')    
    
    # Read ephemerides
    start = observations[obs_id]['raw_header']['MJD-OBS']
    exptime =  (observations[obs_id]['raw_header']['TSTOP'] - \
                observations[obs_id]['raw_header']['TSTART']) / (24 * 3600)
    
    eph = gca_tools.fetch_ephemerides_fromCXO(Time(start, format='mjd'), 
                                              Time(start + exptime, format='mjd'),
                                              '10m')
    observations[obs_id]['full_eph'] = eph

# %%

def get_thick_coordinates(roll_angle, chip_coords, chip_center):
    # Define the UVIS boundaries in chip space
    disp = 1.148 # Angstrom/mm
    scale = 0.13175 # arcseconds/pixel
    # fov = 33*60 / 100 # arcseconds/mm
    height_mm = 18.6 # mm
    length_mm = (53+64) / disp # mm
    
    
    # So, the 3 detectors are arranged [3, 2, 1], and consist of parts:
    # ___________________________________
    #|_____5_____|___  1  ___|_____6_____|
    #|     3     | 2 |   | 2 |     4     |
    #|___________|___|___|___|___________|
    #      -1        center        +1    
    # 
    # Center chip T polygon, in optical coordinates
    center_thick_x = np.array([-53.0, +64.0, +64.0, +17.1, +17.1, -19.1, -19.1, -53.0, -53.0]) # in Angstroms
    center_thick_y = np.array([+ 9.3, + 9.3, + 5.3, + 5.3, - 9.3, - 9.3, + 5.3, + 5.3, + 9.3]) # in mm
        
    # Detector centerline is 3.86mm offset toward positive dispersion (+y)
    # But maybe we don't need to consider that in wavelength/dispersion space?
    center_thick_y_fraction = (center_thick_x / disp) / length_mm
    center_thick_z_fraction = (center_thick_y) / height_mm
    
    # y and z now refer to real chip direction
    
    # Get the roll angle and de-roll the chip coords
    # roll_a_deg = data['fov_header']['ROLL_PNT']
    # roll_a_rad = np.deg2rad(roll_a_deg)
    roll_a_rad = np.deg2rad(roll_angle)
    x_chip_derot = (chip_coords[0]-chip_center[0])*np.cos(roll_a_rad) - (chip_coords[1]-chip_center[1])*np.sin(roll_a_rad)
    y_chip_derot = (chip_coords[0]-chip_center[0])*np.sin(roll_a_rad) + (chip_coords[1]-chip_center[1])*np.cos(roll_a_rad)
    
    # Scale thick by min/max of x/y_chip_derot
    x_derot_min, x_derot_max = np.nanmin(x_chip_derot), np.nanmax(x_chip_derot)
    y_derot_min, y_derot_max = np.nanmin(y_chip_derot), np.nanmax(y_chip_derot)
    
    # Re-zeroed to the upper right
    mod_y_ur = center_thick_y_fraction - center_thick_y_fraction[0]
    mod_z_ur = center_thick_z_fraction - center_thick_z_fraction[0]
    
    center_thick_y_chip = mod_y_ur * (x_derot_max - x_derot_min) + x_derot_min
    center_thick_z_chip = mod_z_ur * (y_derot_max - y_derot_min) + y_derot_max
    
    # fig, ax = plt.subplots()
    # ax.plot(x_chip_derot[1], y_chip_derot[1], color='black', lw=1)
    # ax.plot(center_thick_y_chip, center_thick_z_chip, color='grey', lw=1)
    # ax.set(aspect=1, ylim=ax.get_xlim())
    
    
    # # TEST: derot the central chip
    # test_x = x_chip_derot*np.cos(-roll_a_rad) - y_chip_derot*np.sin(-roll_a_rad) + x0_chip
    # test_y = x_chip_derot*np.sin(-roll_a_rad) + y_chip_derot*np.cos(-roll_a_rad) + y0_chip
    # fig, ax = plt.subplots()
    # ax.plot(x_chip.T, y_chip.T)
    # ax.plot(test_x.T, test_y.T, ls=':')
    # plt.show()
    

    # Now go back to sky coordinates
    thick_x_chip = center_thick_y_chip*np.cos(-roll_a_rad) - center_thick_z_chip*np.sin(-roll_a_rad) + x0_chip
    thick_y_chip = center_thick_y_chip*np.sin(-roll_a_rad) + center_thick_z_chip*np.cos(-roll_a_rad) + y0_chip
    
    return thick_x_chip, thick_y_chip

# Inspect the dither patterns w.r.t. Jupiter's position
for obs_id, data in observations.items():
    
    # The FOV file is detector coords, which are convertible to RA/DEC
    x_chip, y_chip = data['fov_data']['x'], data['fov_data']['y']
    x0_chip, y0_chip = data['fov_data'].columns['X'].coord_ref_point, data['fov_data'].columns['Y'].coord_ref_point
    x0_sky, y0_sky = data['fov_data'].columns['X'].coord_ref_value, data['fov_data'].columns['Y'].coord_ref_value
    x_chip_to_sky, y_chip_to_sky = data['fov_data'].columns['X'].coord_inc, data['fov_data'].columns['Y'].coord_inc
    
    def map_chip_to_sky(x, y):
        x_sky = (x - x0_chip) * x_chip_to_sky + x0_sky
        y_sky = (y - y0_chip) * y_chip_to_sky + y0_sky
        
        return x_sky, y_sky
    
    def map_sky_to_chip(x, y):
        x_chip = (x - x0_sky) / x_chip_to_sky + x0_chip
        y_chip = (y - y0_sky) / y_chip_to_sky + y0_chip
        
        return x_chip, y_chip
        
    # Check that everything works as expected
    x_sky, y_sky = map_chip_to_sky(x_chip, y_chip)
    # fig, ax = plt.subplots()
    # ax.plot(x_sky.T * (24/360), y_sky.T, color='black', lw=1)
    # ax.set(xlabel = 'DEC [decimal hours]', ylabel='RA [deg.]',
    #        xlim = [x_sky.max() * (24/360), x_sky.min() * (24/360)])
    
    thick_x_chip, thick_y_chip = get_thick_coordinates(data['fov_header']['ROLL_PNT'], (x_chip[1], y_chip[1]), (x0_chip, y0_chip))
    thick_x_sky, thick_y_sky = map_chip_to_sky(thick_x_chip, thick_y_chip)
    
    # fig, ax = plt.subplots()
    # ax.plot(x_chip.T, y_chip.T, lw=1)
    # ax.plot(thick_x_chip, thick_y_chip, color='gray', lw=1)
    # plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(x_sky.T, y_sky.T, lw=1)
    ax.plot(thick_x_sky, thick_y_sky, color='gray', lw=1)
    
    # ax.plot(data['full_eph']['RA'], data['full_eph']['DEC'], lw=1)
     
    # Interpolate the ephemeris to dither length
    dither_jd = Time(data['dither_data']['time'], format='cxcsec').jd
    eph_dec = np.interp(dither_jd, data['full_eph']['datetime_jd'], data['full_eph']['DEC'])
    eph_ra = np.interp(dither_jd, data['full_eph']['datetime_jd'], data['full_eph']['RA'])
    
    # Dither - original pointing gives the nominal 'offset' of the dither
    offset_ra = (data['dither_data']['ra'] - data['dither_header']['RA_NOM'])
    offset_dec = (data['dither_data']['dec'] - data['dither_header']['DEC_NOM'])
    
    # Extract the RA, DEC of a point along the CML at 50 deg. north
    # to simulate the auroral viewing
    aurora_edge_radius = np.sin(np.deg2rad(50)) * 0.5 * data['full_eph']['ang_width'] / 3600
    aurora_edge_ra = aurora_edge_radius * np.sin(np.deg2rad(data['full_eph']['NPole_ang']))
    aurora_edge_dec = aurora_edge_radius * np.cos(np.deg2rad(data['full_eph']['NPole_ang']))
    
    aurora_edge_ra = np.interp(dither_jd, data['full_eph']['datetime_jd'], aurora_edge_ra)
    aurora_edge_dec = np.interp(dither_jd, data['full_eph']['datetime_jd'], aurora_edge_dec)
    
    # Subtract the offset: when the camera points up without Jupiter moving,
    # it's as if the camera didn't move by Jupiter moved down
    ax.plot(eph_ra - offset_ra + aurora_edge_ra,
            eph_dec - offset_dec + aurora_edge_dec,
            color='red', lw=0.5)
    
    from matplotlib.patches import Ellipse, Rectangle
    ellipse = Ellipse(xy=(eph_ra[0] - offset_ra[0], eph_dec[0] - offset_dec[0]), 
                      width=data['full_eph']['ang_width'].mean() / 3600,
                      height=data['full_eph']['ang_width'].mean() / 3600,
                      angle=0, 
                      color='blue', alpha=0.5, fill=True, lw=0.5)
    ax.add_patch(ellipse)
    
    box = Rectangle(xy=(eph_ra[0] - offset_ra[0] - 0.5*data['full_eph']['ang_width'].mean() / 3600,
                        eph_dec[0] - offset_dec[0] - 0.5*np.sin(np.deg2rad(50))*data['full_eph']['ang_width'].mean() / 3600),
                    width=data['full_eph']['ang_width'].mean() / 3600,
                    height=np.sin(np.deg2rad(50))*data['full_eph']['ang_width'].mean() / 3600, 
                    angle=data['full_eph']['NPole_ang'].mean(),
                    rotation_point='center',
                    color='orange', alpha=0.5, fill=True, lw=0.5)
    ax.add_patch(box)
    
    
    ax.set(aspect=1, 
           xlabel = 'RA', xlim=[eph_ra.mean()+0.1, eph_ra.mean()-0.1],
           ylabel = 'DEC', ylim=[eph_dec.mean()-0.1, eph_dec.mean()+0.1])
    plt.show()
    
    
    # Check when our reference is inside/outside the thick polygon
    thick_chip = shapely.Polygon([(x, y) for x, y in zip(thick_x_sky, thick_y_sky)])
    shapely.prepare(thick_chip)
    
    testpoint_x = eph_ra - offset_ra + aurora_edge_ra
    testpoint_y = eph_dec - offset_dec + aurora_edge_dec
    in_thick = [shapely.within(shapely.Point(x, y), thick_chip) for x, y in zip(testpoint_x, testpoint_y)]
    
    # # Now, loop over each photon and assign it to thick (True) or thin (False)
    # photon_df = data['all_events']
    # photon_in_thick = []
    # for t in data['all_events']['t']:
    #     # Find the closest time in the dither file
    #     closest_indx = np.abs(data['dither_data']['time'] - t).argmin()
        
    #     # Get the thin/thick filter location at that time
    #     photon_in_thick.append(bool(in_thick[closest_indx]))
        
    # Different tack: check where each individual photon is
    in_thick_list = []
    in_thick_fraction_list = []
    for row_index, photon in tqdm.tqdm(data['filtered_events'].iterrows(), total=len(data['filtered_events'])):
        
        # Closest index to dither file times
        closest_indx = np.abs(data['dither_data']['time'] - photon['t']).argmin()
        
        # Location of the planet's center in RA, DEC
        ra_planet_center = eph_ra[closest_indx]
        dec_planet_center = eph_dec[closest_indx]
        
        # The photon location in (x, y) is measured in sky-coord arcsecs from 
        # the center of the planet
        photon_ra_offset = photon['x'] / 3600 # arcseconds to degrees
        photon_dec_offset = photon['y'] / 3600 # arcseconds to degrees
        
        # Get the effective photon position, accounting for dithering
        eff_ra = ra_planet_center + photon_ra_offset - offset_ra[closest_indx]
        eff_dec = dec_planet_center + photon_dec_offset - offset_dec[closest_indx]
        
        # Test this point against the thick_chip
        in_thick = bool(shapely.within(shapely.Point(eff_ra, eff_dec), thick_chip))
        in_thick_list.append(in_thick)
        
        # Find the total amount of 'time' this location spent in the thin/thick filters
        # This is useful for normalizing later
        all_eff_ra = ra_planet_center + photon_ra_offset - offset_ra
        all_eff_dec = dec_planet_center + photon_dec_offset - offset_dec
        _in_thick = [bool(shapely.within(shapely.Point(_ra, _dec), thick_chip)) for _ra, _dec in zip(all_eff_ra, all_eff_dec)]
        in_thick_fraction_list.append(sum(_in_thick)/len(_in_thick))
    
    in_thick_arr = np.array(in_thick_list)
    in_thick_fraction_arr = np.array(in_thick_fraction_list)
    
    # Add these to the dataframe
    photon_df = data['filtered_events']
    photon_df['event_in_thick'] = in_thick_arr
    photon_df['location_in_thick_fraction'] = in_thick_fraction_arr
    
    new_filepath = base + '/{0}/primary/{0}_UVISfiltered_photonlist.csv'.format(obs_id)
    photon_df.to_csv(new_filepath, sep=',', index=False, header=True)
    
    fig, ax = plt.subplots()
    ax.scatter(data['filtered_events']['x'][in_thick_arr], data['filtered_events']['y'][in_thick_arr], color='black', marker='.', s=8)
    ax.scatter(data['filtered_events']['x'][~in_thick_arr], data['filtered_events']['y'][~in_thick_arr], color='xkcd:violet', marker='o', s=16)
    plt.show()
    
    # breakpoint()
    

# #%%
# # Extract background annulus around Jupiter, normalize, and compare counts

# # Segment the image into 5" x 5" bins
# for obs_id, obs in observations.items():
#     x_lims = np.ceil(obs['all_events']['x'].min()), np.floor(obs['all_events']['x'].max())
#     y_lims = np.ceil(obs['all_events']['y'].min()), np.floor(obs['all_events']['y'].max())
#     x_bins = np.arange(*x_lims, 5)
#     y_bins = np.arange(*y_lims, 5)
    
#     hist_2d, _, _ = np.histogram2d(obs['all_events']['x'], obs['all_events']['y'],
#                                    bins=(x_bins, y_bins))    
    
    
#     # Check each corner of each bin for overlap with Jupiter
#     grid_lower_left = np.meshgrid(x_bins[:-1], y_bins[:-1], indexing='ij')
#     grid_upper_left = np.meshgrid(x_bins[:-1], y_bins[1:], indexing='ij')
#     grid_lower_right = np.meshgrid(x_bins[1:], y_bins[:-1], indexing='ij')
#     grid_upper_right = np.meshgrid(x_bins[1:], y_bins[1:], indexing='ij')
#     inside_limb = np.zeros((len(x_bins)-1, len(y_bins)-1))
#     # Need a new patch, otherwise it's in axis coords
#     jupiter_patch = gc.get_JupiterPatch(obs['full_eph'])
#     for i in range(len(x_bins)-1):
#         for j in range(len(y_bins)-1):
            
#             # Check each corner for being inside the limb
#             ll = jupiter_patch.contains_point((grid_lower_left[0][i,j], grid_lower_left[1][i,j]))
#             ul = jupiter_patch.contains_point((grid_upper_left[0][i,j], grid_upper_left[1][i,j]))
#             lr = jupiter_patch.contains_point((grid_lower_right[0][i,j], grid_lower_right[1][i,j]))
#             ur = jupiter_patch.contains_point((grid_upper_right[0][i,j], grid_upper_right[1][i,j]))
            
#             if any([ll, ul, lr, ur]):
#                 inside_limb[i,j] = 1
                
#     # Get the background statistics from outside the limb
#     # Not a Gaussian, but whatever
#     mean_total_bg = hist_2d.T[inside_limb == 0].mean()/((5/3600)**2)
#     std_total_bg = hist_2d.T[inside_limb == 0].std()/((5/3600)**2)
    
#     # Convert to counts per second
#     exptime = obs['raw_header']['TSTOP'] - obs['raw_header']['TSTART']
#     mean_bg = mean_total_bg / exptime
#     std_bg = std_total_bg / exptime
    
#     # Show the binned image + jupiter's limb
#     fig, ax = plt.subplots()
#     pcm = ax.pcolormesh(x_bins, y_bins, hist_2d.T/exptime, vmin=0, vmax=0.005)
#     fig.colorbar(pcm, ax=ax, label="Counts/s")
    
#     jupiter_patch = gc.get_JupiterPatch(obs['full_eph'], color='white', alpha=0.5)
#     ax.add_patch(jupiter_patch)
#     ax.set(aspect=1)
#     title = "Obs ID: {}".format(obs_id) + '\n' + r"Background: {:.02f} $\pm$ {:.02f} counts/sq. deg./s".format(mean_bg, std_bg)
#     ax.set_title(title)
    
#     plt.show()