#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:05:15 2024

@author: mrutala
"""

import spiceypy as spice
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

#   Essential solar system and timekeeping kernels
#   From:
#   https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls   
#   https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls
#   https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc
#   https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp
#   https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup365.bsp
spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de440s.bsp')
spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/satellites/jup365.bsp')

#   Juno perijove dates, copied from 
#   https://lasp.colorado.edu/mop/missions/juno/trajectory-information/
PJs = dict(
    PJ69 = dt.datetime(2025,  1, 28, 23,  5,  7, 905),
    PJ70 = dt.datetime(2025,  3,  2, 16,  4, 27, 901),
    PJ71 = dt.datetime(2025,  4,  4,  9, 30, 50, 237),
    PJ72 = dt.datetime(2025,  5,  7,  3,  1, 28, 831),
    PJ73 = dt.datetime(2025,  6,  8, 20, 30, 45, 482),
    PJ74 = dt.datetime(2025,  7, 11, 13, 40, 11, 272),
    PJ75 = dt.datetime(2025,  8, 13,  7,  6, 45, 868),
    PJ76 = dt.datetime(2025,  9, 14, 23, 42, 30, 219),
    )

#   Set the start and end dates to be the full Chandra Cycle 26
#   Alternatively, comment this out and set a smaller window based on perijoves
# start_date, end_date = dt.datetime(2025, 1, 1), dt.datetime(2025, 10, 31)
half_window_in_days = 2
start_date, end_date = np.array([dt.timedelta(days=-half_window_in_days), dt.timedelta(days=+half_window_in_days)]) + PJs['PJ76']

#   Get an array of datetimes from the start and end dates
#   Then convert from datetime to SPICE et
datetimes = np.arange(start_date, end_date, dt.timedelta(minutes=1), dtype=dt.datetime)
ets = spice.datetime2et(datetimes)

#   Add sub observer lon and lat to a dictionary by looping over each time step
sub_observer = dict(datetime=np.array([]), 
                    lon=np.array([]), 
                    lat=np.array([]))
for datetime, et in zip(datetimes, ets):
    subpoints, trgepc, sfvec = spice.subpnt('NEAR POINT/ELLIPSOID', 'Jupiter', et, 'IAU_JUPITER', 'LT+S', 'EARTH')
    lon, lat, alt = spice.recpgr('Jupiter', subpoints, 71492, 0.06487)
    
    sub_observer['datetime'] = np.append(sub_observer['datetime'], datetime)
    sub_observer['lon'] = np.append(sub_observer['lon'], lon * 180/np.pi)
    sub_observer['lat'] = np.append(sub_observer['lat'], lat * 180/np.pi)
    
#   Initialize figure with extra axis for overplotting lines
fig, axs = plt.subplots(nrows=2, sharex=True)
plt.subplots_adjust(hspace=0)

axs[1].set_xlim(start_date, end_date)
axs[1].set_ylim(0, 360)
axs[0].set_ylim(0,1)

#   Plot the CML
axs[1].plot(sub_observer['datetime'], sub_observer['lon'])

#   Highlight XRay Hot Spot (110-220) padded to make a 14.5ks obs. 
#   (or (65-280) for longer padding, as previously used)
hotspot_lonrange = [110, 220]
chandra_exptime = 14.5e3  #  seconds
jupiter_day = 9.926*3600  #  seconds
chandra_lonrange = (chandra_exptime/jupiter_day) * 360  #  degrees
buffer_window = (chandra_lonrange - (hotspot_lonrange[1] - hotspot_lonrange[0]))/2.
hotspot_windows = np.where((sub_observer['lon'] > hotspot_lonrange[0] - buffer_window) & 
                           (sub_observer['lon'] < hotspot_lonrange[1] + buffer_window),
                           0.5, np.nan)

axs[0].plot(sub_observer['datetime'], hotspot_windows,
              color='orange', linewidth=5)


#   Juno PJ +/- 10 hours to PJ (maybe +/- 5 hours?)
juno_uvs_window = [dt.timedelta(hours=-10) , dt.timedelta(hours=+10)]
for PJ, PJ_datetime in PJs.items():
    axs[0].plot(PJ_datetime + np.array(juno_uvs_window), [0.25,0.25], 
                  color='blue', linewidth=5)


axs[0].set_yticks([0.25, 0.5], ['Perijove', 'Hotspot'])

