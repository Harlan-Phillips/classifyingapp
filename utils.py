# Astropy modules
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from astropy.io import fits
from astropy import units as u
from astropy.time import Time

# Data handling modules
import pandas as pd
import numpy as np

import plotly.graph_objs as go
import plotly.io as pio

# Compression and file handling modules
import gzip
import io
import os
from collections import Counter, defaultdict
import random
import time
import os

# Image handling modules
from PIL import Image
import matplotlib.pyplot as plt
import mpld3
from mpld3 import fig_to_html, plugins

# Network and API modules
import requests
from penquins import Kowalski
from dl import queryClient as qc
from dl.helpers.utils import convert

# JSON handling
import json

# External utility modules
from ztfquery.utils import stamps

import mastcasjobs
from celery  import Celery, shared_task

from flask import session
import logging
from models import db, User, Transient, Classification

basedir = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(level=logging.DEBUG,  # or INFO
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler()])

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    return celery

def read_secrets():
    secrets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'secrets.txt')
    with open(secrets_file, 'r') as f:
        secrets = f.read().splitlines()
    return secrets

secrets = read_secrets()

username_kowalski = secrets[0]
password_kowalski = secrets[1]
wsid_mastcasjobs = secrets[2]
password_mastcasjobs = secrets[3]

# Reading data from CSV
column_names = ['stars_x', 'stars_y', 'ellipticals_x', 'ellipticals_y', 'spirals_x', 'spirals_y',
                'LIRGs_x', 'LIRGs_y', 'qsos_x', 'qsos_y']
data = pd.read_csv('wpd_datasets.csv', names=column_names, skiprows=1)

# Extracting the data for each group
stars_x = data['stars_x'].dropna().tolist()
stars_y = data['stars_y'].dropna().tolist()

ellipticals_x = data['ellipticals_x'].dropna().tolist()
ellipticals_y = data['ellipticals_y'].dropna().tolist()

spirals_x = data['spirals_x'].dropna().tolist()
spirals_y = data['spirals_y'].dropna().tolist()

LIRGs_x = data['LIRGs_x'].dropna().tolist()
LIRGs_y = data['LIRGs_y'].dropna().tolist()

qsos_x = data['qsos_x'].dropna().tolist()
qsos_y = data['qsos_y'].dropna().tolist()

def logon():
    """ Log onto Kowalski """
    s = Kowalski(
        protocol='https', host='kowalski.caltech.edu', port=443,
        verbose=False, username=username_kowalski, password=password_kowalski)
    return s

def get_dets(s, name):
    """ Fetch detection alerts from Kowalski """
    q = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_alerts",
            "filter": {
                'objectId': {'$eq': name},
                'candidate.isdiffpos': {'$in': ['1', 't']},
            },
            "projection": {
                "_id": 0,
                "candidate.jd": 1,
                "candidate.magpsf": 1,
                "candidate.exptime": 1,
                "candidate.sigmapsf": 1,
                "candidate.fid": 1,
                "candidate.programid": 1,
                "candidate.field": 1,
                "candidate.ra": 1,
                "candidate.dec": 1,
                "candidate.ssdistnr": 1,
                "candidate.ssmagnr": 1,
                "candidate.distpsnr1": 1,
                "candidate.sgscore1": 1,
                "candidate.drb": 1
            }
        }
    }
    query_result = s.query(query=q)
    try:
        out = query_result['default']['data']
        return out
    except:
        return []
    
def alert_table(detections):
    flattened_data = [item['candidate'] for item in detections]
    df = pd.DataFrame(flattened_data)

    # If you need to convert to CSV for any reason
    csv_data = df.to_csv(index=False)

    print(df)
    # Display the DataFrame
    return df
        
def get_drb(s,name,dets):
    """ Calculate the median position from alerts, and the scatter """
    det_alerts = dets
    if not det_alerts:
        return None, None, None, None
    
    #det_prv = get_prv_dets(s, name)
    
    drbs = [det['candidate']['drb'] for det in det_alerts if 'drb' in det['candidate']]
    
    if not drbs:
        return None, None, None, None
    
    # Calculate the median position
    med = np.median(drbs)
    mini = np.min(drbs)
    mx = np.max(drbs)
    avg = np.mean(drbs)

    return med,mini,mx,avg

def get_span(s,name,dets):
    """ Calculate the median position from alerts, and the scatter """
    det_alerts = dets
    if not det_alerts:
        return None
    
    det_prv = get_prv_dets(s, name)
    
    detects = [det['candidate']['jd'] for det in det_alerts]

    if det_prv:
        for det in det_prv:
            if len(det)>50:
                detects.append(det['jd'])

    if not detects:
        return None
    
    return max(detects) - min(detects)

def get_pos(s,name):
    """ Calculate the median position from alerts, and the scatter """
    det_alerts = get_dets(s, name)
    if not det_alerts:
        return None, None, None
    det_prv = get_prv_dets(s, name)
    ras = [det['candidate']['ra'] for det in det_alerts]
    decs = [det['candidate']['dec'] for det in det_alerts]

    # Calculate the median position
    ra = np.median(ras)
    dec = np.median(decs)

    if det_prv is not None:
        for det in det_prv:
            if len(det)>50:
                ras.append(det['ra'])
                decs.append(det['dec'])

    scat_sep = 0
    if len(ras)>1:
        # Calculate the separations between each pair
        seps = []
        for i,raval in enumerate(ras[:-1]):
            c1 = SkyCoord(raval, decs[i], unit='deg')
            c2 = SkyCoord(ras[i+1], decs[i+1], unit='deg')
            seps.append(c1.separation(c2).arcsec)
        # Calculate the median separation
        scat_sep = np.median(seps)

    return ra,dec,scat_sep


def get_galactic(ra,dec):
    """ Convert to galactic coordinates, ra and dec given in decimal deg """
    c = SkyCoord(ra,dec,unit='deg')
    galactic_l = c.galactic.l.deg
    galactic_b = c.galactic.b.deg
    return galactic_l, galactic_b

def get_ecliptic(ra, dec):
    """ Convert to ecliptic coordinates, ra and dec given in decimal degrees """
    # Create a SkyCoord object with RA and Dec
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    
    obstime = Time(58000, format='mjd')

    # Convert to Geocentric True Ecliptic coordinates with obstime
    ecliptic_coord = c.transform_to(GeocentricTrueEcliptic(obstime=obstime))
    
    # Extract ecliptic longitude and latitude
    ecliptic_lon = ecliptic_coord.lon.deg
    ecliptic_lat = ecliptic_coord.lat.deg
    
    return ecliptic_lon, ecliptic_lat


def get_lc(s, name):
    """ Retrieve LC for object """
    # The alerts
    dets = get_dets(s, name)
    if not dets:
        # If no alerts, try getting forced photometry directly
        logging.warning(f"No detection alerts found for {name}. Checking forced photometry.")
        det_prv_forced = get_prv_dets_forced(s, name)
        if det_prv_forced:
            lc = pd.DataFrame(det_prv_forced)
            # Rename forced phot columns to match expected final columns if possible
            if 'mag' in lc.columns: lc['mag_final'] = lc['mag']
            if 'magerr' in lc.columns: lc['emag_final'] = lc['magerr']
            if 'limmag5sig' in lc.columns: lc['maglim'] = lc['limmag5sig']
            if 'forcediffimflux' in lc.columns and 'forcediffimfluxunc' in lc.columns:
                 lc['snr'] = lc['forcediffimflux'] / lc['forcediffimfluxunc']
                 lc['isdet'] = lc['snr'] > 3
            else:
                 lc['isdet'] = False # Assume non-detection if flux/snr info missing
            # Add required columns if missing
            for col in ['jd', 'fid', 'ra', 'dec', 'programid', 'field', 'isalert']:
                 if col not in lc.columns:
                      lc[col] = None # Or appropriate default
            lc['isalert'] = False # Mark as not alert
            lc['origin'] = 'forced_photometry' # Add origin
            logging.debug(f"Created LC DataFrame solely from forced photometry for {name}.")
        else:
            logging.error(f"No alert or forced photometry data found for {name}.")
            return pd.DataFrame() # Return empty DataFrame if no data at all
    else:
        df_alerts = pd.DataFrame([val['candidate'] for val in dets])
        df_alerts['isalert'] = True
        df_alerts['origin'] = 'alert' # Add origin
        lc = df_alerts

    # Get 30-day history from forced photometry
    det_prv_forced = get_prv_dets_forced(s, name)
    if det_prv_forced is not None:
        df_forced = pd.DataFrame(det_prv_forced)
        if not df_forced.empty:
            df_forced['isalert'] = False
            df_forced['origin'] = 'forced_photometry' # Add origin

            # Merge the two dataframes
            lc = lc.merge(
                df_forced, on='jd', how='outer',
                suffixes=['_alerts', '_forced30d']).sort_values('jd').reset_index()

            # Define columns to drop CAREFULLY - Keep essential ones like ra, dec, fid, mag, err
            cols_to_drop = ['index', 'rcid', 'rfid', 'sciinpseeing', 'scibckgnd',
                'scisigpix', 'magzpsci', 'magzpsciunc', 'magzpscirms', 'clrcoeff',
                'clrcounc', 'exptime', 'adpctdif1', 'adpctdif2', 'procstatus',
                'distnr', 'ranr', 'decnr', 'magnr', 'sigmagnr', 'chinr',
                'sharpnr', 'forcediffimflux', 'forcediffimfluxunc', 'limmag3sig'] # REMOVED ra, dec, alert_ra, alert_dec etc.
                # Also keep 'ra', 'dec' potentially coming from forced photometry if available
            cols_to_drop_existing = [col for col in cols_to_drop if col in lc.columns]
            lc = lc.drop(cols_to_drop_existing, axis=1, errors='ignore')

            # --- Combine essential columns --- Keep original names if possible
            # RA/Dec: Prioritize alert coords, fall back to forced phot coords if they exist
            lc['ra'] = lc['ra_alerts'].combine_first(lc.get('ra_forced30d', pd.Series(index=lc.index)))
            lc['dec'] = lc['dec_alerts'].combine_first(lc.get('dec_forced30d', pd.Series(index=lc.index)))

            lc['fid'] = lc['fid_alerts'].combine_first(lc['fid_forced30d'])
            lc['programid'] = lc['programid_alerts'].combine_first(lc['programid_forced30d'])
            lc['field'] = lc['field_alerts'].combine_first(lc['field_forced30d'])
            lc['isalert'] = lc['isalert_alerts'].combine_first(lc['isalert_forced30d'])
            lc['origin'] = lc['origin_alerts'].combine_first(lc['origin_forced30d'])

            # Select magnitudes. Options: magpsf/sigmapsf (alert), mag/magerr (30d)
            lc['mag_final'] = lc['magpsf']  # alert value
            lc['emag_final'] = lc['sigmapsf']  # alert value
            if 'mag' in lc.columns and 'snr' in lc.columns:
                 # Ensure 'snr' is numeric before comparison
                 lc['snr'] = pd.to_numeric(lc['snr'], errors='coerce')
                 valid_snr_mask = lc['snr'] > 3
                 lc.loc[valid_snr_mask, 'mag_final'] = lc.loc[valid_snr_mask, 'mag']
                 if 'magerr' in lc.columns:
                      lc.loc[valid_snr_mask, 'emag_final'] = lc.loc[valid_snr_mask, 'magerr']

            # Select limits. Use limmag5sig if available
            lc['maglim'] = lc.get('limmag5sig', pd.Series(index=lc.index))

            # Define whether detection or not based on SNR or alert status
            if 'snr' in lc.columns:
                 lc['isdet'] = np.logical_or(lc['isalert'] == True, lc['snr'] > 3)
            else:
                 # Fallback if SNR is missing (e.g., only alerts were present)
                 lc['isdet'] = lc['isalert'] == True

            # Drop merged/intermediate columns
            cols_to_drop_final = [
                'ra_alerts', 'dec_alerts', 'ra_forced30d', 'dec_forced30d',
                'fid_alerts', 'fid_forced30d', 'field_alerts', 'field_forced30d',
                'programid_alerts', 'programid_forced30d', 'isalert_alerts',
                'isalert_forced30d', 'origin_alerts', 'origin_forced30d',
                'magpsf', 'sigmapsf', 'mag', 'magerr', 'snr', 'limmag5sig',
                'diffmaglim', # Often redundant or less useful than limmag5sig
                'pid' # Process ID, usually not needed
                # Keep 'ssdistnr', 'ssmagnr', 'sgscore1', 'distpsnr1' if they exist
            ]
            cols_to_drop_final_existing = [col for col in cols_to_drop_final if col in lc.columns]
            lc = lc.drop(cols_to_drop_final_existing, axis=1, errors='ignore')

            # Drop rows where essential data (mag or limit) is missing
            # Keep rows that are non-detections but have a limit
            essential_missing_mask = lc['mag_final'].isna() & lc['maglim'].isna()
            lc = lc[~essential_missing_mask]
        else:
            # Case where only alerts exist, ensure columns match final schema
            if 'magpsf' in lc.columns: lc['mag_final'] = lc['magpsf']
            if 'sigmapsf' in lc.columns: lc['emag_final'] = lc['sigmapsf']
            # Ensure other essential columns exist
            if 'maglim' not in lc.columns: lc['maglim'] = np.nan
            if 'isdet' not in lc.columns: lc['isdet'] = True # Assume alerts are detections
            if 'ra' not in lc.columns: lc['ra'] = np.nan # Should exist from alerts
            if 'dec' not in lc.columns: lc['dec'] = np.nan # Should exist from alerts

            # Drop original mag/err if renamed
            lc = lc.drop(['magpsf', 'sigmapsf'], axis=1, errors='ignore')

    # --- Handle prv_candidates (older detections, often without forced phot info) ---
    df_prv = pd.DataFrame(get_prv_dets(s, name))
    if not df_prv.empty:
        logging.debug(f"Found {len(df_prv)} prv_candidates for {name}. Merging.")
        df_prv['isalert'] = False
        df_prv['origin'] = 'prv_candidate'

        # Select/rename relevant columns from prv_candidates before merge
        # Prioritize magpsf/sigmapsf if available
        prv_cols_map = {
            'jd': 'jd',
            'fid': 'fid',
            'ra': 'ra',
            'dec': 'dec',
            'magpsf': 'mag_final',
            'sigmapsf': 'emag_final',
            'diffmaglim': 'maglim'
            # Add other potentially useful columns? ssdistnr, ssmagnr, sgscore1, distpsnr1?
        }
        cols_to_keep = [col for col in prv_cols_map.keys() if col in df_prv.columns]
        df_prv_renamed = df_prv[cols_to_keep].rename(columns=prv_cols_map)

        # Define prv_candidates as detections if they have mag_final
        df_prv_renamed['isdet'] = df_prv_renamed['mag_final'].notna()

        # Merge with existing lc data, avoiding duplicate columns from previous merge
        # Use suffixes specific to this merge step
        lc = lc.merge(df_prv_renamed, on='jd', how='outer', suffixes=['_main', '_prv'])

        # --- Combine columns after prv_candidate merge --- #
        # Iterate through columns that might have _main and _prv versions
        for col_base in ['fid', 'ra', 'dec', 'mag_final', 'emag_final', 'maglim', 'isdet', 'isalert', 'origin']:
            col_main = f'{col_base}_main'
            col_prv = f'{col_base}_prv'
            if col_main in lc.columns and col_prv in lc.columns:
                lc[col_base] = lc[col_main].combine_first(lc[col_prv])
                lc = lc.drop([col_main, col_prv], axis=1)
            elif col_main in lc.columns: # Only main exists (no overlap or prv didn't have it)
                lc[col_base] = lc[col_main]
                lc = lc.drop(col_main, axis=1)
            elif col_prv in lc.columns: # Only prv exists
                lc[col_base] = lc[col_prv]
                lc = lc.drop(col_prv, axis=1)
            # If neither exists, the column won't be created

        # Ensure essential columns exist after all merges, fill with NaN if necessary
        final_expected_cols = ['jd', 'ra', 'dec', 'fid', 'mag_final', 'emag_final', 'maglim', 'isdet', 'isalert', 'origin',
                               'programid', 'field', 'ssdistnr', 'ssmagnr', 'sgscore1', 'distpsnr1']
        for col in final_expected_cols:
            if col not in lc.columns:
                lc[col] = np.nan

        # Sort final DataFrame
        lc = lc.sort_values('jd').reset_index(drop=True)

    # Final check for isdet - should be True if mag_final is not NaN
    # This corrects cases where only prv_candidates were merged without explicit isdet
    if 'isdet' in lc.columns and 'mag_final' in lc.columns:
         lc['isdet'] = lc['isdet'].fillna(lc['mag_final'].notna())
    elif 'mag_final' in lc.columns:
         lc['isdet'] = lc['mag_final'].notna()

    # Final cleanup - drop rows where mag_final and maglim are both NaN
    final_drop_mask = lc['mag_final'].isna() & lc['maglim'].isna()
    lc = lc[~final_drop_mask]

    # --- Final Column Cleanup BEFORE returning --- #
    # Ensure only the intended final columns exist, drop any intermediate ones
    # that might have survived the merge logic (like original magpsf/sigmapsf)
    final_columns_to_keep = [
        'jd', 'ra', 'dec', 'fid', 'mag_final', 'emag_final', 'maglim',
        'isdet', 'isalert', 'origin', 'programid', 'field', 'ssdistnr',
        'ssmagnr', 'sgscore1', 'distpsnr1'
        # Add any other columns that are *intentionally* generated and needed downstream
    ]
    # Select only the columns that exist in the DataFrame from the desired list
    actual_columns_to_keep = [col for col in final_columns_to_keep if col in lc.columns]
    lc = lc[actual_columns_to_keep]

    logging.debug(f"Final LC for {name} has {len(lc)} rows. Detections: {lc['isdet'].sum() if 'isdet' in lc.columns else 'N/A'}.")
    logging.debug(f"Final LC columns (after final cleanup): {lc.columns.tolist()}")

    return lc

def get_prv_dets(s, name):
    """
    Query previous detections of a given source from the ZTF_alerts_aux catalog.

    Parameters:
    s (object): The Kowalski session object used to query the catalog.
    name (str): The name of the source to query.

    Returns:
    list or None: A list of previous candidates if found, None otherwise.
    """

    q = {"query_type": "find",
         "query": {
             "catalog": "ZTF_alerts_aux",
             "filter": {
                     '_id': {'$eq': name},
             },
             "projection": {
                     "_id": 0,
                     "prv_candidates": 1,
             }
         }
         }
    query_result = s.query(query=q)
    if len(query_result['default']['data'])>0:
        out = query_result['default']['data'][0]['prv_candidates']
        return out
    return None



def get_prv_dets_forced(s, name):
    """
    Query forced photometry history of a given source from the ZTF_alerts_aux catalog.

    Parameters:
    s (object): The Kowalski session object used to query the catalog.
    name (str): The name of the source to query.

    Returns:
    list or None: A list of forced photometry histories if found, None otherwise.
    """

    q = {"query_type": "find",
         "query": {
             "catalog": "ZTF_alerts_aux",
             "filter": {
                     '_id': {'$eq': name},
             },
             "projection": {
                     "_id": 0,
                     "fp_hists": 1,
             }
         }
         }
    query_result = s.query(query=q)
    if len(query_result['default']['data'])>0:
        if 'fp_hists' in query_result['default']['data'][0]:
            return query_result['default']['data'][0]['fp_hists']
    return None

def make_triplet(alert, normalize=False):
    """
    Get the science, reference, and difference image for a given alert
    Takes in an alert packet
    """
    cutout_dict = dict()

    for cutout in ('science', 'template', 'difference'):
        tmpstr = 'cutout' + cutout.capitalize()
        cutout_data = alert[tmpstr]['stampData']

        # unzip
        with gzip.open(io.BytesIO(cutout_data), 'rb') as f:
            with fits.open(io.BytesIO(f.read()), ignore_missing_simple=True) as hdu:
                data = hdu[0].data
                # replace nans with zeros
                cutout_dict[cutout] = np.nan_to_num(data)
                # normalize
                if normalize:
                    cutout_dict[cutout] /= np.linalg.norm(cutout_dict[cutout])

        # pad to 63x63 if smaller
        shape = cutout_dict[cutout].shape
        if shape != (63, 63):
            cutout_dict[cutout] = np.pad(cutout_dict[cutout], [(0, 63 - shape[0]), (0, 63 - shape[1])],
                                         mode='constant', constant_values=1e-9)

    triplet = np.zeros((63, 63, 3))
    triplet[:, :, 0] = cutout_dict['science']
    triplet[:, :, 1] = cutout_dict['template']
    triplet[:, :, 2] = cutout_dict['difference']
    return triplet


def plot_triplet(tr):
    """ From Dima's Kowalski tutorial """
    fig,axarr = plt.subplots(1,3,figsize=(5.5, 2.1), dpi=120)
    titles = ['Science', 'Reference', 'Difference']
    u_scale_factor = [40, 40, 10]
    l_scale_factor = [30, 40, 1]
    for ii,ax in enumerate(axarr):
        ax.axis('off')
        data = tr[:,:,ii]
        dat = data.flatten()
        sig = np.median(np.abs(dat-np.median(dat)))
        median = np.median(data)
        ax.imshow(
            data, origin='upper', cmap=plt.cm.bone,
            vmin=median-l_scale_factor[ii]*sig,
            vmax=median+u_scale_factor[ii]*sig) # Corrected vmax index
        #norm=LogNorm())
        ax.set_title(titles[ii], fontsize = 12)

    fig.subplots_adjust(wspace=0)
    return fig  # Return the figure to allow saving


def plot_ztf_cutout(s, alert, cutout_type='science'):
    """ Plot the ZTF cutouts: science, reference, difference """
    cutout_data = alert['cutout' + cutout_type.capitalize()]['stampData']
    
    # unzip
    with gzip.open(io.BytesIO(cutout_data), 'rb') as f:
        with fits.open(io.BytesIO(f.read()), ignore_missing_simple=True) as hdu:
            data = hdu[0].data
            # replace nans with zeros
            data = np.nan_to_num(data)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(data, cmap='gray', origin='lower')
    ax.set_title(f'{cutout_type.capitalize()} Cutout')
    plt.axis('off')
    plt.tight_layout()
    return fig

def filter_and_plot_alerts(s, output_dir, object_id):
    
    # Define the desired order and keys for plots
    plot_keys_definitions = {
        'first': 'First Detection',
        'last': 'Last Detection',
        'median': 'Median Detection',
        'highest_snr': 'Highest S/N',
        'highest_drb': 'Highest DRB',
        'lowest_drb': 'Lowest DRB',
        'brightest_g': 'Brightest g-band',
        'brightest_r': 'Brightest r-band'
    }
    plot_keys_ordered = list(plot_keys_definitions.keys())
    # Initialize results with None placeholders for each key in order
    plot_files_map = {key: None for key in plot_keys_ordered}

    # Check if any files are missing to determine if querying/plotting is needed
    need_query_or_plot = False
    potential_filenames = [os.path.join(output_dir, f"{object_id}_{key}.png") for key in plot_keys_ordered]
    for fname in potential_filenames:
        if not os.path.isfile(fname):
            need_query_or_plot = True
            break

    if not need_query_or_plot:
        # All potential files exist, return their basenames in the correct order
        return [os.path.basename(fname) for fname in potential_filenames if os.path.exists(fname)]

    # Proceed with querying and plotting
    q0 = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_alerts",
            "filter": {"objectId": object_id}
        },
        "kwargs": {
            "limit": 1000, # Consider if this limit is sufficient
        }
    }
    out = s.query(q0)
    alerts = out["default"]["data"]
    
    if len(alerts) == 0:
        print("No alerts found for the given object ID.")
        return [] # Return empty list if no alerts

    # Convert alerts to DataFrame
    df = pd.DataFrame([alert['candidate'] for alert in alerts])
    
    # Ensure 'drb' column exists and handle potential non-numeric values if necessary
    if 'drb' in df.columns:
        df['drb'] = pd.to_numeric(df['drb'], errors='coerce') # Convert to numeric, coerce errors to NaN
    else:
        df['drb'] = np.nan # Add NaN column if 'drb' doesn't exist

    if 'scorr' not in df.columns: # Ensure scorr exists for S/N
        df['scorr'] = np.nan 

    # --- Select Detections ---
    key_detections = {}
    df_sorted = df.sort_values(by='jd')

    if not df_sorted.empty:
        key_detections['first'] = df_sorted.iloc[0]
        key_detections['last'] = df_sorted.iloc[-1]
        key_detections['median'] = df_sorted.iloc[len(df_sorted) // 2]
    
    if not df['scorr'].isna().all():
         key_detections['highest_snr'] = df.loc[df['scorr'].idxmax()] if not df['scorr'].isnull().all() else None

    # DRB checks - ensure column exists and has valid values
    if 'drb' in df.columns and not df['drb'].isna().all():
         key_detections['highest_drb'] = df.loc[df['drb'].idxmax()] if not df['drb'].isnull().all() else None
         key_detections['lowest_drb'] = df.loc[df['drb'].idxmin()] if not df['drb'].isnull().all() else None

    # Brightest g/r checks
    df_g = df[(df['fid'] == 1) & df['magpsf'].notna()]
    if not df_g.empty:
        key_detections['brightest_g'] = df_g.loc[df_g['magpsf'].idxmin()]
    
    df_r = df[(df['fid'] == 2) & df['magpsf'].notna()]
    if not df_r.empty:
        key_detections['brightest_r'] = df_r.loc[df_r['magpsf'].idxmin()]

    # --- Plotting Loop (in predefined order) ---
    for key in plot_keys_ordered:
        if key in key_detections and key_detections[key] is not None:
            detection = key_detections[key]
            fname = os.path.join(output_dir, f"{object_id}_{key}.png")
            
            # Only plot if the file doesn't exist - CHECK RE-ADDED
            if not os.path.isfile(fname):
                try:
                    # Find the corresponding full alert packet
                    alert = next(alert for alert in alerts if alert['candidate']['candid'] == detection['candid'])
                    triplet = make_triplet(alert)
                    fig = plot_triplet(triplet)
                    fig.savefig(fname, bbox_inches="tight")
                    plt.close(fig)
                except StopIteration:
                    print(f"Warning: Could not find full alert packet for {key} detection (candid: {detection.get('candid', 'N/A')}). Skipping plot.")
                except Exception as e:
                    print(f"Error plotting '{key}' for {object_id}: {e}")
                    plt.close('all') # Close any potentially lingering plots

            # Check if file exists (might have existed before or was just created) 
            # and update the map if it does.
            if os.path.isfile(fname):
                 plot_files_map[key] = os.path.basename(fname)
        # No need for an else clause, the map entry remains None if detection missing or plot failed
             
    # Return the list of filenames (or None) in the fixed order
    return [plot_files_map[key] for key in plot_keys_ordered]

def plot_ps1_cutout(s,ddir,name,ra,dec):
    """ Plot cutout from PS1 """
    if dec>0:
        decsign = "+"
    else:
        decsign = "-"

        
    fnames = []
    need_query = False
    filters = ["first", "last", "median", "highest_snr", "highest_drb"]
    
    for filter in filters:
        fname = "%s/%s_%s.png" % (output_dir, object_id, filter)
        if not os.path.isfile(fname):
            need_query = True
        fnames.append(fname)

    if need_query:
        q0 = {
            "query_type": "find",
            "query": {
                "catalog": "ZTF_alerts",
                "filter": {"objectId": object_id}
            },
            "kwargs": {
                "limit": 1000,
            }
        }
        out = s.query(q0)
        alerts = out["default"]["data"]
        
        if len(alerts) == 0:
            print("No alerts found for the given object ID.")
            return

        # Convert alerts to DataFrame
        df = pd.DataFrame([alert['candidate'] for alert in alerts])
        
        # Extract the desired detections
        # Ensure the DataFrame is sorted by the 'jd' column in ascending order
        df_sorted = df.sort_values(by='jd')

        # Select the first, last, and median detections
        first_detection = df_sorted.iloc[0]
        last_detection = df_sorted.iloc[-1]
        median_detection = df_sorted.iloc[len(df_sorted) // 2]
        highest_sn_detection = df.loc[df['scorr'].idxmax()]
        highest_drb_detection = df.loc[df['drb'].idxmax()] if 'drb' in df else None
        lowest_drb_detection = df.loc[df['drb'].idxmin()] if 'drb' in df else None

        # Select brightest g-band and r-band
        brightest_g_index = df.loc[(df['fid'] == 1) & (df['magpsf'].notna())]['magpsf'].idxmin() if not df.loc[df['fid'] == 1].empty else None
        brightest_g_detection = df.loc[brightest_g_index] if brightest_g_index is not None else None
        brightest_r_index= df.loc[(df['fid'] == 2) & (df['magpsf'].notna())]['magpsf'].idxmin() if not df.loc[df['fid'] == 2].empty else None
        brightest_r_detection = df.loc[brightest_r_index] if brightest_r_index is not None else None

        # Plot cutouts for each detection
        key_detections = {
            'first': first_detection,
            'last': last_detection,
            'median': median_detection,
            'highest_snr': highest_sn_detection,
        }

        if highest_drb_detection is not None:
            key_detections['highest_drb'] = highest_drb_detection
            key_detections['lowest_drb'] = lowest_drb_detection
        
        if brightest_g_detection is not None:
           key_detections['brightest_g'] = brightest_g_detection
        
        if brightest_r_detection is not None:
            key_detections['brightest_r'] = brightest_r_detection

        for key, detection in key_detections.items():
            print(f"Key: {key}")
            print(f"JD: {detection['jd']}")
            print(f"FID: {detection['fid']}")
            print(f"DRB: {detection.get('drb', 'N/A')}")  # Safely get 'drb' in case it's not present
            print("-" * 40)
            alert = next(alert for alert in alerts if alert['candidate']['candid'] == detection['candid'])
            triplet = make_triplet(alert)
            fig = plot_triplet(triplet)
            fig.savefig(os.path.join(output_dir, f"{object_id}_{key}.png"), bbox_inches="tight")
            plt.close(fig)
    
    return fnames

def plot_ps1_cutout(s,ddir,name,ra,dec):
    """ Plot cutout from PS1 """
    if dec>0:
        decsign = "+"
    else:
        decsign = "-"

    fname = ddir + "/%s_ps1.png" %name
    if os.path.isfile(fname)==False:
        img = stamps.get_ps_stamp(ra, dec, size=240, color=["y","g","i"])
        plt.figure(figsize=(2.1,2.1), dpi=120)
        img_array = np.asarray(img)
        plt.imshow(img_array)
        center_x = img_array.shape[1] // 2
        center_y = img_array.shape[0] // 2
        plt.title("PS1 (y/g/i)", fontsize = 12)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(fname, bbox_inches = "tight")
        plt.close()
    return fname


def plot_ls_cutout(s,ddir,name,ra,dec):
    """ Plot cutout from Legacy Survey """
    fname = ddir + "/%s_ls.png"%name
    if os.path.isfile(fname)==False:
        url = "http://legacysurvey.org/viewer/cutout.jpg?ra=%s&dec=%s&layer=ls-dr9&pixscale=0.27&bands=grz" %(ra,dec)
        plt.figure(figsize=(2.1,2.1), dpi=120)
        try:
            r = requests.get(url)
            plt.imshow(Image.open(io.BytesIO(r.content)))
            image = io.BytesIO(r.content)
            img_array = np.asarray(Image.open(io.BytesIO(r.content)))
            center_x = img_array.shape[1] // 2
            center_y = img_array.shape[0] // 2
            plt.title("LegSurv DR9", fontsize = 12)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(fname, bbox_inches="tight")
            lslinkstr = "http://legacysurvey.org/viewer?" +\
                        "ra=%.6f&dec=%s%.6f"%(ra, decsign, abs(dec))+\
                        "&zoom=16&layer=dr9"
            outputf.write("<a href = %s>"%lslinkstr)
            outputf.write('<img src="%s_ls.png" height="200">'%(name))
            outputf.write("</a>")
            outputf.write('</br>')
        except:
            # not in footprint
            return None
        # you want to save it anyway so you don't do this over and over again
        plt.close()
    return fname

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

def plot_light_curve(lc, source_id, span=None):
    # Preserve existing data prep logic
    non_dets = lc[(lc['isdet'] == False) & (lc['maglim'] > 1)]
    lc = lc.dropna(subset=['mag_final']) if lc['mag_final'].isna().sum() > 0 else lc
    non_dets = non_dets.dropna(subset=['maglim']) if non_dets['maglim'].isna().sum() > 0 else non_dets

    # Convert JD to MJD
    from astropy.time import Time
    lc['mjd'] = Time(lc['jd'], format='jd').mjd - 58000
    non_dets['mjd'] = Time(non_dets['jd'], format='jd').mjd - 58000

    fig = go.Figure()
    color_map = {1: 'seagreen', 2: 'crimson', 3: 'goldenrod'}
    symbol_map = {1: 'square', 2: 'circle', 3: 'diamond'}

    # Upper limits
    for band in non_dets['fid'].unique():
        band_data = non_dets[non_dets['fid'] == band]
        fig.add_trace(go.Scatter(
            x=band_data['mjd'],
            y=band_data['maglim'],
            mode='markers',
            marker=dict(color=color_map.get(band, 'gray'), symbol='triangle-down', opacity=0.7, size=10),
            name=f'Upper Limit'
        ))
    
    # Detections
    for band in lc['fid'].unique():
        band_data = lc[lc['fid'] == band]
        # Determine filter name based on band (fid)
        if band == 1:
            filter_name = 'g-band'
        elif band == 2:
            filter_name = 'r-band'
        elif band == 3:
            filter_name = 'i-band' # Assuming fid 3 is i-band
        else:
            filter_name = f'band {band}' # Fallback for unknown bands

        fig.add_trace(go.Scatter(
            x=band_data['mjd'],
            y=band_data['mag_final'],
            mode='markers',
            error_y=dict(type='data', array=band_data['emag_final'], visible=True),
            marker=dict(color=color_map.get(band, 'gray'), symbol=symbol_map.get(band, 'circle'), size=10),
            name=filter_name # Use the determined filter name
        ))

    # Flip y-axis
    y_min = lc['mag_final'].min() - 0.5 if not lc['mag_final'].empty else 0
    y_max = lc['mag_final'].max() + 0.5 if not lc['mag_final'].empty else 0

    if span == 'detections' and not lc.empty:
        diff = lc['mjd'].max() - lc['mjd'].min()
        if diff < 1:
            x_min = lc['mjd'].min() - (diff * 1.2)
            x_max = lc['mjd'].max() + (diff * 1.2)
        else:
            x_min = lc['mjd'].min() - 0.5
            x_max = lc['mjd'].max() + 0.5
        fig.update_xaxes(range=[x_min, x_max])
    
    fig.update_layout(
        width=800,
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=f"Light Curve for {source_id}",
        xaxis_title="MJD - 58000",
        yaxis_title="Magnitude",
        xaxis=dict(showgrid=True, gridcolor="lightgray", linecolor="black"),
        yaxis=dict(showgrid=True, gridcolor="lightgray", linecolor="black", range=[y_max, y_min])
    )

    plot_filename = f'static/light_curves/{source_id}_light_curve.html'
    if span == "detections":
        plot_filename = f'static/light_curves/{source_id}_light_curve_zoomed.html'
    pio.write_html(fig, file=plot_filename, auto_open=False, full_html=False)
    return plot_filename

def plot_big_light_curve(lc, source_id, span=None):
    # Preserve existing data prep logic
    non_dets = lc[(lc['isdet'] == False) & (lc['maglim'] > 1)]
    lc = lc.dropna(subset=['mag_final']) if lc['mag_final'].isna().sum() > 0 else lc
    non_dets = non_dets.dropna(subset=['maglim']) if non_dets['maglim'].isna().sum() > 0 else non_dets

    from astropy.time import Time
    lc['mjd'] = Time(lc['jd'], format='jd').mjd - 58000
    non_dets['mjd'] = Time(non_dets['jd'], format='jd').mjd - 58000

    fig = go.Figure()
    color_map = {1: 'seagreen', 2: 'crimson', 3: 'goldenrod'}
    symbol_map = {1: 'square', 2: 'circle', 3: 'square'}

    # Upper limits
    for band in non_dets['fid'].unique():
        band_data = non_dets[non_dets['fid'] == band]
        fig.add_trace(go.Scatter(
            x=band_data['mjd'],
            y=band_data['maglim'],
            mode='markers',
            marker=dict(color=color_map.get(band, 'gray'), symbol='triangle-down', opacity=0.7, size=12),
            name=f'Upper Limit'
        ))

    # Detections
    for band in lc['fid'].unique():
        band_data = lc[lc['fid'] == band]
        # Determine filter name based on band (fid)
        if band == 1:
            filter_name = 'g-band'
        elif band == 2:
            filter_name = 'r-band'
        elif band == 3:
            filter_name = 'i-band' # Assuming fid 3 is i-band
        else:
            filter_name = f'band {band}' # Fallback for unknown bands

        fig.add_trace(go.Scatter(
            x=band_data['mjd'],
            y=band_data['mag_final'],
            mode='markers',
            error_y=dict(type='data', array=band_data['emag_final'], visible=True),
            marker=dict(color=color_map.get(band, 'gray'), symbol=symbol_map.get(band, 'circle'), size=12),
            name=filter_name # Use the determined filter name
        ))

    y_min = lc['mag_final'].min() - 0.5 if not lc['mag_final'].empty else 0
    y_max = lc['mag_final'].max() + 0.5 if not lc['mag_final'].empty else 0

    if span == 'detections' and not lc.empty:
        diff = lc['mjd'].max() - lc['mjd'].min()
        if diff < 1:
            x_min = lc['mjd'].min() - (diff * 1.2)
            x_max = lc['mjd'].max() + (diff * 1.2)
        else:
            x_min = lc['mjd'].min() - 0.5
            x_max = lc['mjd'].max() + 0.5
        fig.update_xaxes(range=[x_min, x_max])

    fig.update_layout(
        width=950,
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=f"Light Curve for {source_id}",
        xaxis_title="MJD - 58000",
        yaxis_title="Magnitude",
        xaxis=dict(showgrid=True, gridcolor="lightgray", linecolor="black"),
        yaxis=dict(showgrid=True, gridcolor="lightgray", linecolor="black", range=[y_max, y_min])
    )

    plot_filename = f'static/light_curves/{source_id}_big_light_curve.html'
    if span == "detections":
        plot_filename = f'static/light_curves/{source_id}_big_light_curve_zoomed.html'
    pio.write_html(fig, file=plot_filename, auto_open=False, full_html=False)
    return plot_filename


def xmatch_ls(ra, dec, radius=5):
    """ Query Legacy Survey """
    # Run the query
    columns = "ra,dec,type,ls_id"
    query = """
    SELECT %s
    FROM ls_dr9.tractor
    WHERE q3c_radial_query(ra, dec, %.6f, %.6f, %.2f/3600)
    """ % (columns, ra, dec, radius)
    try:
        result = qc.query(sql=query)
        df = convert(result)

        nmatch = len(df)
        if nmatch >= 1:
            # Create table of values sorted by separation
            c = SkyCoord(ra, dec, frame='icrs', unit='deg')
            coos = SkyCoord(df["ra"].values, df["dec"].values, frame='icrs', unit='deg')
            sep = c.separation(coos)
            sep_arcsec = sep.arcsec  # in arcsec
            pa = c.position_angle(coos)  # positive angles East of North (match wrt science)
            pa_degree = pa.deg
            df["sep_arcsec"] = sep_arcsec
            df["pa_degree"] = pa_degree
            df = df.sort_values(by=["sep_arcsec"])

            # Get photo-z info
            my_ls_id = df["ls_id"].values[0]
            columns = "ls_id,z_phot_median,z_phot_l68,z_phot_u68"
            query = """
                    SELECT %s
                    FROM ls_dr9.photo_z
                    WHERE ls_id=%d
                    """ % (columns, my_ls_id)
            result = qc.query(sql=query)
            df2 = convert(result)
            out = df.merge(df2)
            return out
    except Exception as e:
        print(f"Error in xmatch_ls: {e}")
        return pd.DataFrame()
    return pd.DataFrame()


def filter_ztf_alerts(ztf_alerts):
    filtered_data = []
    seen_sgscore1 = set()
    for _, row in ztf_alerts.iterrows():
        if 'distpsnr1' in row and row['distpsnr1'] != -999.0 and 'sgscore1' in row and row['sgscore1'] != -999.0:
            if row['sgscore1'] not in seen_sgscore1:
                filtered_data.append(row)
                seen_sgscore1.add(row['sgscore1'])
    return pd.DataFrame(filtered_data)

def plot_polar_coordinates(ztf_alerts, ra_ps1, dec_ps1, legacy_survey_data, source_ra, source_dec, output_path, xlim, ylim, point_size):
    """
    Plots the polar coordinates of nearby sources with the transient at the center.

    Parameters:
    - ztf_alerts (pd.DataFrame): DataFrame containing ZTF alert data with columns 'ra', 'dec', 'fid', and other relevant data.
    - legacy_survey_data (pd.DataFrame): DataFrame containing Legacy Survey data with columns 'ra', 'dec', and other relevant data.
    - source_ra (float): Right Ascension of the transient source.
    - source_dec (float): Declination of the transient source.
    - output_path (str): Path to save the output plot.
    """
    
    
    # Filter ZTF alerts 
    #ztf_alerts = filter_ztf_alerts(ztf_alerts)

    # Create a SkyCoord object for the transient source
    central_coord = SkyCoord(ra=source_ra, dec=source_dec, unit='deg')

    # Process ZTF alerts
    ztf_coords = SkyCoord(ra=ztf_alerts['ra'], dec=ztf_alerts['dec'], unit='deg')

    # Calculate offsets in arcseconds
    ztf_ra_offset = (ztf_coords.ra - central_coord.ra).arcsec
    ztf_dec_offset = (ztf_coords.dec - central_coord.dec).arcsec

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')

    # Plot ZTF alerts
    filters = {1: 'green', 2: 'red'}
    for fid, color in filters.items():
        mask = ztf_alerts['fid'] == fid
        #if ztf_ra_offset[mask] and ztf_dec_offset > 0.5:
        scatter = ax.scatter(ztf_ra_offset[mask], ztf_dec_offset[mask], color=color, label=f'ztf{color[0]}', s=point_size)  # Reduce marker size to 50
        labels = [f"<div>RA: {ra:.6f}, Dec: {dec:.6f}</div>" for ra, dec in zip(ztf_alerts['ra'], ztf_alerts['dec'])]
        plugins.connect(fig, plugins.PointHTMLTooltip(scatter, labels=labels))

    # Plot Legacy Survey data if available
    if not legacy_survey_data.empty:
        legacy_coords = SkyCoord(ra=legacy_survey_data['ra'], dec=legacy_survey_data['dec'], unit='deg')
        legacy_ra_offset = (legacy_coords.ra - central_coord.ra).arcsec
        legacy_dec_offset = (legacy_coords.dec - central_coord.dec).arcsec
        legacy_scatter = ax.scatter(legacy_ra_offset, legacy_dec_offset, color='blue', marker='*', s=point_size*10, label='Legacy Survey')
        
        # Simplify HTML Tooltip content
        labels = [f"Legacy Survey<br><div>RA: {ra:.6f}<br> Dec: {dec:.6f}</div>" for ra, dec in zip(legacy_survey_data['ra'], legacy_survey_data['dec'])]
        plugins.connect(fig, plugins.PointHTMLTooltip(legacy_scatter, labels=labels))

    # Plot Legacy Survey data if available
    if ra_ps1 and dec_ps1:
        # Create a SkyCoord object for the PS1 source
        ps1_coord = SkyCoord(ra=ra_ps1, dec=dec_ps1, unit='deg')
        ps1_ra_offset = (ps1_coord.ra - central_coord.ra).arcsec
        ps1_dec_offset = (ps1_coord.dec - central_coord.dec).arcsec
        # Plot PS1 source
        ps1_scatter = ax.scatter(ps1_ra_offset, ps1_dec_offset, color='purple', marker='*', s=point_size*10, label='PS1')
        plugins.connect(fig, plugins.PointHTMLTooltip(ps1_scatter, labels=[f'PS1 Source<br>RA: {ra_ps1:.6f}<br>Dec: {dec_ps1:.6f}']))
            
        
    # Central source
    central_scatter = ax.scatter(0, 0, color='black', marker='o', s=point_size, label='Transient Avg.')  # Reduce marker size to 100
    plugins.connect(fig, plugins.PointHTMLTooltip(central_scatter, labels=['Transient Avg.']))

    # Add concentric circles
    circle1 = plt.Circle((0, 0), 3, color='blue', fill=False, alpha=0.1)  # Increase radius to 3 arcseconds
    circle2 = plt.Circle((0, 0), 1.5, color='blue', fill=False, alpha=0.3)  # Increase radius to 1.5 arcseconds
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    ax.set_xlabel('RA (arcsec)', fontsize=16)
    ax.set_ylabel(r'Dec (arcsec)', fontsize=16)
    ax.legend(title='Filter/Catalog')
    ax.set_title('Coordinates of Nearby Sources', fontsize=18)
    #ax.legend(loc='upper right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), fancybox=True, shadow=True, ncol=2)
    ax.grid(alpha =.1)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()
    #fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.2, wspace=0.2)

    # Save as HTML
    html_str = mpld3.fig_to_html(fig)
    with open(output_path, 'w') as f:
        f.write(html_str)

    plt.close(fig)

def plot_big_polar_coordinates(ztf_alerts, ra_ps1, dec_ps1, legacy_survey_data, source_ra, source_dec, output_path, xlim, ylim, point_size):
    """
    Plots the polar coordinates of nearby sources with the transient at the center.

    Parameters:
    - ztf_alerts (pd.DataFrame): DataFrame containing ZTF alert data with columns 'ra', 'dec', 'fid', and other relevant data.
    - legacy_survey_data (pd.DataFrame): DataFrame containing Legacy Survey data with columns 'ra', 'dec', and other relevant data.
    - source_ra (float): Right Ascension of the transient source.
    - source_dec (float): Declination of the transient source.
    - output_path (str): Path to save the output plot.
    """
    
    
    # Filter ZTF alerts 
    #ztf_alerts = filter_ztf_alerts(ztf_alerts)

    # Create a SkyCoord object for the transient source
    central_coord = SkyCoord(ra=source_ra, dec=source_dec, unit='deg')

    # Process ZTF alerts
    ztf_coords = SkyCoord(ra=ztf_alerts['ra'], dec=ztf_alerts['dec'], unit='deg')

    # Calculate offsets in arcseconds
    ztf_ra_offset = (ztf_coords.ra - central_coord.ra).arcsec
    ztf_dec_offset = (ztf_coords.dec - central_coord.dec).arcsec

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # Plot ZTF alerts
    filters = {1: 'green', 2: 'red'}
    for fid, color in filters.items():
        mask = ztf_alerts['fid'] == fid
        #if ztf_ra_offset[mask] and ztf_dec_offset > 0.5:
        scatter = ax.scatter(ztf_ra_offset[mask], ztf_dec_offset[mask], color=color, label=f'ztf{color[0]}', s=point_size * 1.1)  # Reduce marker size to 50
        labels = [f"<div>RA: {ra:.6f}, Dec: {dec:.6f}</div>" for ra, dec in zip(ztf_alerts['ra'], ztf_alerts['dec'])]
        plugins.connect(fig, plugins.PointHTMLTooltip(scatter, labels=labels))

    # Plot Legacy Survey data if available
    if not legacy_survey_data.empty:
        legacy_coords = SkyCoord(ra=legacy_survey_data['ra'], dec=legacy_survey_data['dec'], unit='deg')
        legacy_ra_offset = (legacy_coords.ra - central_coord.ra).arcsec
        legacy_dec_offset = (legacy_coords.dec - central_coord.dec).arcsec
        legacy_scatter = ax.scatter(legacy_ra_offset, legacy_dec_offset, color='blue', marker='*', s=point_size *10, label='Legacy Survey')
        
        # Simplify HTML Tooltip content
        labels = [f"Legacy Survey<br><div>RA: {ra:.6f}<br> Dec: {dec:.6f}</div>" for ra, dec in zip(legacy_survey_data['ra'], legacy_survey_data['dec'])]
        plugins.connect(fig, plugins.PointHTMLTooltip(legacy_scatter, labels=labels))

    # Plot Legacy Survey data if available
    if ra_ps1 and dec_ps1:
        # Create a SkyCoord object for the PS1 source
        ps1_coord = SkyCoord(ra=ra_ps1, dec=dec_ps1, unit='deg')
        ps1_ra_offset = (ps1_coord.ra - central_coord.ra).arcsec
        ps1_dec_offset = (ps1_coord.dec - central_coord.dec).arcsec
        # Plot PS1 source
        ps1_scatter = ax.scatter(ps1_ra_offset, ps1_dec_offset, color='purple', marker='*', s=point_size*10, label='PS1')
        plugins.connect(fig, plugins.PointHTMLTooltip(ps1_scatter, labels=[f'PS1 Source<br>RA: {ra_ps1:.6f}<br>Dec: {dec_ps1:.6f}']))
            
        
    # Central source
    central_scatter = ax.scatter(0, 0, color='black', marker='o', s=point_size, label='Transient Avg.')  # Reduce marker size to 100
    plugins.connect(fig, plugins.PointHTMLTooltip(central_scatter, labels=['Transient Avg.']))

    # Add concentric circles
    circle1 = plt.Circle((0, 0), 3, color='blue', fill=False, alpha=0.1)  # Increase radius to 3 arcseconds
    circle2 = plt.Circle((0, 0), 1.5, color='blue', fill=False, alpha=0.3)  # Increase radius to 1.5 arcseconds
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    ax.set_xlabel('RA (arcsec)', fontsize=18)
    ax.set_ylabel(r'Dec (arcsec)', fontsize=18)
    ax.legend(title='Filter/Catalog')
    ax.set_title('Coordinates of Nearby Sources', fontsize=20)
    #ax.legend(loc='upper right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), fancybox=True, shadow=True, ncol=2)

    ax.grid(alpha =.1)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()
    #fig.subplots_adjust(top=.98, bottom=0.2, left=0.1, right=.75, hspace=0, wspace=0)

    # Save as HTML
    html_str = mpld3.fig_to_html(fig)
    with open(output_path, 'w') as f:
        f.write(html_str)

    plt.close(fig)

def get_most_confident_classification(classifications):
    """Determine the most confident classification."""
    classification_counts = defaultdict(lambda: {'count': 0, 'confidence': 0})
    for classification in classifications:
        classification_counts[classification.classification]['count'] += 1
        if classification.confidence == 'Not confident':
            classification_counts[classification.classification]['confidence'] += 1
        elif classification.confidence == 'Confident':
            classification_counts[classification.classification]['confidence'] += 2
        elif classification.confidence == 'Certain':
            classification_counts[classification.classification]['confidence'] += 3

    if classification_counts:
        return max(
            classification_counts.items(),
            key=lambda x: (x[1]['confidence'], x[1]['count'])
        )[0]
    return None


def get_ps1_host(s, name):
    """Retrieve the distance to the nearest PS1 host"""
    q = {
        "query_type": "find_one",
        "query": {
            "catalog": "ZTF_alerts",
            "filter": {
                'objectId': {'$eq': name},
            },
            "projection": {
                "_id": 0,
                "candidate.distpsnr1": 1
            }
        }
    }
    query_result = s.query(query=q)
    out = query_result['default']['data']
    return out['candidate']['distpsnr1']

def get_ps1_photoz(ra, dec, radius=10):
    """ Find the photoz for a PS1 object within 10 arcseconds """
    jobs = mastcasjobs.MastCasJobs(
            userid=wsid_mastcasjobs, password=password_mastcasjobs, context="HLSP_PS1_STRM")
    query = """select o.objID, o.uniquePspsOBid, o.raMean, o.decMean,
            o.class, o.prob_Galaxy, o.prob_Star, o.prob_QSO,
            o.extrapolation_Class, o.cellDistance_Class, o.cellID_Class,
            o.z_phot, o.z_photErr, o.z_phot0,
            o.extrapolation_Photoz, cellDistance_Photoz, o.cellID_Photoz
            from fGetNearbyObjEq({},{},{}) nb
            inner join catalogRecordRowStore o on o.objID=nb.objID
            """.format(ra, dec, radius/60)
    max_retries = 5
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            tab = jobs.quick(query, task_name="python cone search")
            time.sleep(3)
            return tab
        except Exception as e:
            print(f"Attempt {attempt+1} failed with exception: {e}")
            if 'deadlocked on lock resources' in str(e):
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return []
    print("PS1 query failed after multiple retries")
    return []

def analyze_ps1_photoz(s, name, ra, dec, radius=5):
    nearest_ps1_dist = get_ps1_host(s, name)
    
    if nearest_ps1_dist < radius:
        print("Getting PS1 photo-z")
        tab = get_ps1_photoz(ra, dec, radius)
        
        if nearest_ps1_dist < radius:
            print("Getting PS1 photo-z")
            tab = get_ps1_photoz(ra, dec, radius)
            
            if len(tab) > 0:
                print("PS1 photo-z values obtained")
                ramatch = tab['raMean'].data
                decmatch = tab['decMean'].data
                cmatch = SkyCoord(ramatch, decmatch, unit='deg')
                seps = cmatch.separation(SkyCoord(ra, dec, unit='deg')).arcsec
                ind = np.argmin(seps)
                print(ramatch[ind], decmatch[ind])
                return ramatch[ind], decmatch[ind]
            else:
                print("No PS1 photo-z values found")
                return None, None
        else:
            print("No nearby PS1 host found within the specified radius")
            return None, None
    
    return None, None

def wise_xmatch(s, ra, dec, radius=3):
    print(f"Querying WISE for coordinates: RA={ra}, Dec={dec}, Radius={radius}")
    qu = {
        "query_type": "cone_search",
        "query": {
            "object_coordinates": {
                "radec": "[(%.5f, %.5f)]" % (ra, dec),
                "cone_search_radius": "%.2f" % radius,
                "cone_search_unit": "arcsec"
            },
            "kwargs": {},
            "catalogs": {
                "AllWISE": {
                    "filter": "{}",
                    "projection": "{}"
                }
            }
        }
    }
    r = s.query(query=qu)
    out = r['default']['data']
    key = list(out['AllWISE'].keys())[0]
    print(f"Output from WISE query: {out}")
    if len(out['AllWISE'][key]) > 0:
        dat = out['AllWISE'][key][0]
        if np.logical_and.reduce(('w1mpro' in dat.keys(), 'w3mpro' in dat.keys(), 'w2mpro' in dat.keys())):
            wmag = [dat['w1mpro'], dat['w2mpro'], dat['w3mpro']]
            return dat['ra'], dat['dec'], wmag
        else:
            print("WISE data does not contain all required magnitudes (w1mpro, w2mpro, w3mpro).")
            return None, None, None
    else:
        print("No matching WISE data found within the specified radius.")
        return None, None, None

def plot_wise(s, name, ra, dec, output_path):
    # WISE data (RA, Dec, WISE magnitudes)
    ra, dec, wmag = wise_xmatch(s, ra, dec)

    if not ra or not dec or not wmag:
        print(f"No WISE data available for source {name}.")
        return None
    
    if len(wmag) != 3:
        print("WISE magnitudes are incomplete.")
        return None

    w1_w2 = wmag[0] - wmag[1]
    w2_w3 = wmag[1] - wmag[2]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the WISE source data
    scatter = ax.scatter(w1_w2, w2_w3, color='blue', label='WISE Source')

    # Plot the data from the CSV
    plt.fill(stars_x, stars_y, label='Stars', alpha=0.3)
    plt.fill(ellipticals_x, ellipticals_y, label='Ellipticals', alpha=0.3)
    plt.fill(spirals_x, spirals_y, label='Spirals', alpha=0.3)
    plt.fill(LIRGs_x, LIRGs_y, label='LIRGs', alpha=0.3)
    plt.fill(qsos_x, qsos_y, label='QSOs/Seyferts', alpha=0.3)

    # Adding text labels for each group
    plt.text(0.376, 0.376, 'Stars', fontsize=16, color='black', ha='center')
    plt.text(0.696, 0.114, 'Ellipticals', fontsize=16, color='black', ha='center')
    plt.text(2.036, 0.196, 'Spirals', fontsize=16, color='black', ha='center')
    plt.text(4.895, 0.456, 'LIRGs', fontsize=16, color='black', ha='center')
    plt.text(3.07, 1.276, 'QSOs/Seyferts', fontsize=16, color='black', ha='center')

    # Set axis labels and title
    ax.set_xlabel('W1 - W2', fontsize=14)
    ax.set_ylabel('W2 - W3', fontsize=14)
    ax.set_title(f'WISE Color-Color Plot\n(RA: {ra:.5f}, Dec: {dec:.5f})', fontsize=16)
    ax.legend()
    ax.grid(alpha =.1)

    # Tooltips and interactive plot
    tooltip = plugins.PointHTMLTooltip(scatter, labels=[f"RA: {ra:.5f}, Dec: {dec:.5f}<br>W1-W2: {w1_w2:.2f}, W2-W3: {w2_w3:.2f}"], css="background-color: white; color: black; font-size: 14px;")
    plugins.connect(fig, tooltip)

    html_str = mpld3.fig_to_html(fig)
    
    # Save the HTML to a file
    with open(output_path, 'w') as f:
        f.write(html_str)

    plt.close(fig)
    return output_path

def fetch_transient_data(kowalski_session, source_id):
    """Fetch all the required data for rendering a classification page for a given source."""
    try:
        # Fetch positional and galactic data
        ra, dec, scat_sep = get_pos(kowalski_session, source_id)
        if ra is None or dec is None:
            logging.error(f"Could not determine position for {source_id}")
            return None # Cannot proceed without position
        logging.debug(f"RA: {ra}, Dec: {dec}, Scatter Separation: {scat_sep}")

        # Fetch galactic coordinates
        galactic_l, galactic_b = get_galactic(ra, dec)
        logging.debug(f"Galactic Coordinates - l: {galactic_l}, b: {galactic_b}")

        # Fetch ecliptic coordinates
        ecliptic_lon, ecliptic_lat = get_ecliptic(ra, dec)
        logging.debug(f"Ecliptic Coordinates - lon: {ecliptic_lon}, lat: {ecliptic_lat}")

        # Fetch the original public alerts (needed for DRB and span)
        # We will NOT use this for the main table or alert count anymore
        original_dets_packets = get_dets(kowalski_session, source_id)
        logging.debug(f"Original Public Alerts (dets packets): {len(original_dets_packets) if original_dets_packets else 0} found")

        # Fetch comprehensive light curve data (includes alerts, forced phot, prv dets)
        # THIS WILL BE THE PRIMARY SOURCE FOR TABLE AND COUNT
        light_curve_df = get_lc(kowalski_session, source_id)
        if light_curve_df is None or light_curve_df.empty:
             logging.warning(f"Comprehensive light curve (get_lc) returned empty or None for {source_id}. Falling back to original dets packets.")
             # Fallback: use original alert packets if get_lc fails
             if original_dets_packets:
                 light_curve_df = pd.DataFrame([p['candidate'] for p in original_dets_packets])
                 # Add necessary columns if missing from original packets
                 if 'mag_final' not in light_curve_df.columns and 'magpsf' in light_curve_df.columns:
                     light_curve_df['mag_final'] = light_curve_df['magpsf']
                 if 'emag_final' not in light_curve_df.columns and 'sigmapsf' in light_curve_df.columns:
                     light_curve_df['emag_final'] = light_curve_df['sigmapsf']
                 if 'isdet' not in light_curve_df.columns:
                      # Assume all original packets are detections if 'isdet' is missing
                      light_curve_df['isdet'] = True 
                 if 'origin' not in light_curve_df.columns:
                      light_curve_df['origin'] = 'alert_packet'
             else:
                 # If both fail, create an empty DataFrame to avoid errors later
                 light_curve_df = pd.DataFrame()
                 logging.error(f"No alert data could be retrieved for {source_id} from get_lc or get_dets.")

        logging.debug(f"Comprehensive Light Curve DataFrame head:\n{light_curve_df.head() if not light_curve_df.empty else 'Empty LC DataFrame'}")

        # Calculate alert count based on actual detections in the comprehensive LC DataFrame
        detections_lc_df = pd.DataFrame() # Initialize empty DataFrame
        if not light_curve_df.empty and 'isdet' in light_curve_df.columns:
             # Ensure 'isdet' is boolean or comparable to True
             try:
                 detections_lc_df = light_curve_df[light_curve_df['isdet'] == True].copy()
             except Exception as e:
                 logging.error(f"Error filtering light_curve_df by 'isdet': {e}. LC columns: {light_curve_df.columns}")
                 # Attempt recovery if possible, or default to empty
                 if 'isdet' in light_curve_df.columns: # Check again if column exists
                     detections_lc_df = light_curve_df[light_curve_df['isdet'].notna()].copy() # Fallback: Count non-null isdet
                 else:
                    detections_lc_df = light_curve_df.copy() # Fallback: Count all if isdet is missing

        alert_count = detections_lc_df.shape[0] # THIS IS THE CORRECT COUNT
        logging.debug(f"Final Alert Count (from LC detections): {alert_count}")

        # Add 'origin' column if it doesn't exist in the detections_lc_df
        if 'origin' not in detections_lc_df.columns:
            # Add the column before copying
            light_curve_df['origin'] = 'unknown' # Add to original df if filtering failed
            if not detections_lc_df.empty:
                detections_lc_df['origin'] = 'unknown' # Add to filtered df

        # Prepare data for the alert table from the comprehensive LC detections
        table_columns_map = {
            # Source Col Name -> Target Col Name for Table
            'jd': 'jd',
            'fid': 'fid',
            'programid': 'programid',
            'field': 'field',
            'ra': 'ra',
            'dec': 'dec',
            'mag_final': 'magpsf',   # Use mag_final as the magnitude for the table
            'emag_final': 'sigmapsf', # Use emag_final as the error for the table
            'ssdistnr': 'ssdistnr',
            'ssmagnr': 'ssmagnr',
            'sgscore1': 'sgscore1',
            'distpsnr1': 'distpsnr1',
            'origin': 'origin'
        }
        raw_alerts_for_table = pd.DataFrame()
        expected_table_cols = list(table_columns_map.values())

        if not detections_lc_df.empty:
            # 1. Copy the filtered detections DataFrame
            df_for_table = detections_lc_df.copy()
            logging.debug(f"Step 1 (Copy): df_for_table shape = {df_for_table.shape}, columns = {df_for_table.columns.tolist()}")

            # 2. Ensure all SOURCE columns needed for the map exist
            for source_col in table_columns_map.keys():
                if source_col not in df_for_table.columns:
                    logging.warning(f"Source column '{source_col}' missing in detections_lc_df for table prep. Filling with NaN.")
                    df_for_table[source_col] = np.nan
            logging.debug(f"Step 2 (Ensure Source Cols): df_for_table shape = {df_for_table.shape}, columns = {df_for_table.columns.tolist()}")

            # 3. Rename columns based on the map
            df_for_table = df_for_table.rename(columns=table_columns_map)
            logging.debug(f"Step 3 (Rename): df_for_table shape = {df_for_table.shape}, columns = {df_for_table.columns.tolist()}")

            # 4. Select only the TARGET columns expected by the table
            # Ensure target columns exist after renaming, fill if necessary (though renaming should handle this)
            final_table_cols = []
            missing_target_cols = []
            for target_col in expected_table_cols:
                if target_col in df_for_table.columns:
                    final_table_cols.append(target_col)
                else:
                    # This case should be less likely now with the rename first approach
                    logging.warning(f"Target column '{target_col}' missing after rename.")
                    missing_target_cols.append(target_col)

            logging.debug(f"Step 4 (Target Cols Check): Expected = {expected_table_cols}, Found = {final_table_cols}, Missing = {missing_target_cols}")

            # Assign the processed DataFrame using only the found target columns
            if final_table_cols: # Only select if we have columns to select
                raw_alerts_for_table = df_for_table[final_table_cols]
                logging.debug(f"Step 4 (Select Target Cols): raw_alerts_for_table shape = {raw_alerts_for_table.shape}")
            else:
                logging.error("No target columns found after processing! Resulting table data will be empty.")
                raw_alerts_for_table = pd.DataFrame() # Ensure it's an empty DF

        else:
            logging.warning("detections_lc_df was empty, cannot prepare table data.")


        # Convert NaN to None for JSON/template compatibility
        raw_alerts_for_table = raw_alerts_for_table.replace({pd.NaT: None, np.nan: None})
        raw_alerts = raw_alerts_for_table.to_dict(orient='records')
        logging.debug(f"Step 5 (To Dict): raw_alerts list length = {len(raw_alerts)}")

        # Fetch DRB stats based on original alerts (dets packets) - DRB is specific to alert packets
        med_drb, min_drb, max_drb, avg_drb = get_drb(kowalski_session, source_id, original_dets_packets)
        logging.debug(f"DRB - Med: {med_drb}, Min: {min_drb}, Max: {max_drb}, Avg: {avg_drb}")

        # Calculate span using original dets packets as well
        span = get_span(kowalski_session, source_id, original_dets_packets)
        logging.debug(f"Span (days): {span}")

        # Directories for cutouts and plots
        cutout_dir = os.path.join(basedir, 'static', 'cutouts')
        light_cur = os.path.join(basedir, 'static', 'light_curves')
        wise_dir = os.path.join(basedir, 'static', 'wise_plots')

        # WISE plot
        wise_plot_path = os.path.join(wise_dir, f"{source_id}_wise_plot.html")
        wise_filename = None # Initialize as None
        if os.path.exists(wise_plot_path):
            wise_filename = os.path.basename(wise_plot_path) # Just use basename if exists
            logging.debug(f"WISE plot found: {wise_filename}")
        else:
            # Check if RA/Dec are valid before plotting
            if ra is not None and dec is not None:
                 wise_plot_result = plot_wise(kowalski_session, source_id, ra, dec, wise_plot_path)
                 if wise_plot_result:
                     wise_filename = os.path.basename(wise_plot_result)
                     logging.debug(f"WISE plot generated: {wise_filename}")
                 else:
                     logging.debug(f"WISE plot generation failed or returned None for {source_id}.")
            else:
                logging.warning(f"Skipping WISE plot for {source_id} due to missing RA/Dec.")

        # Light curves (use the comprehensive light_curve_df fetched earlier)
        light_curve_path = os.path.join(light_cur, f"{source_id}_light_curve.html")
        big_light_curve_path = os.path.join(light_cur, f"{source_id}_big_light_curve.html")
        light_curve_zoomed_path = os.path.join(light_cur, f"{source_id}_light_curve_zoomed.html")
        big_light_curve_zoomed_path = os.path.join(light_cur, f"{source_id}_big_light_curve_zoomed.html")

        # Check and generate light curve plots using the comprehensive 'light_curve_df'
        plot_filename_rel = os.path.join('static', 'light_curves', os.path.basename(light_curve_path))
        plot_filename_zoomed_rel = os.path.join('static', 'light_curves', os.path.basename(light_curve_zoomed_path))
        if not os.path.exists(light_curve_path) or not os.path.exists(light_curve_zoomed_path):
             if not light_curve_df.empty:
                 logging.debug(f"Generating light curve plots for {source_id}")
                 plot_light_curve(light_curve_df.copy(), source_id) # Pass a copy to avoid modifying original df
                 plot_light_curve(light_curve_df.copy(), source_id, "detections")
             else:
                 logging.warning(f"Skipping light curve plot generation for {source_id} due to empty light_curve_df.")


        plot_big_filename_rel = os.path.join('static', 'light_curves', os.path.basename(big_light_curve_path))
        plot_big_filename_zoomed_rel = os.path.join('static', 'light_curves', os.path.basename(big_light_curve_zoomed_path))
        if not os.path.exists(big_light_curve_path) or not os.path.exists(big_light_curve_zoomed_path):
             if not light_curve_df.empty:
                 logging.debug(f"Generating big light curve plots for {source_id}")
                 plot_big_light_curve(light_curve_df.copy(), source_id) # Pass a copy
                 plot_big_light_curve(light_curve_df.copy(), source_id, "detections")
             else:
                  logging.warning(f"Skipping big light curve plot generation for {source_id} due to empty light_curve_df.")

        # ZTF cutouts - Still uses original alert packets to pick specific moments.
        ztf_cutout_filenames_or_none = filter_and_plot_alerts(kowalski_session, cutout_dir, source_id)
        ztf_cutout_basenames_for_template = ztf_cutout_filenames_or_none # Pass the list with Nones
        logging.debug(f"ZTF Cutout list (with Nones): {ztf_cutout_basenames_for_template}")

        # Pan-STARRS (PS1) cutouts
        ps1_cutout_path = os.path.join(cutout_dir, f"{source_id}_ps1.png")
        ps1_cutout_basename = None
        if os.path.exists(ps1_cutout_path):
            ps1_cutout_basename = os.path.basename(ps1_cutout_path)
        else:
            if ra is not None and dec is not None:
                ps1_cutout = plot_ps1_cutout(kowalski_session, cutout_dir, source_id, ra, dec)
                ps1_cutout_basename = os.path.basename(ps1_cutout) if ps1_cutout else None
            else:
                 logging.warning(f"Skipping PS1 cutout for {source_id} due to missing RA/Dec.")
        logging.debug(f"PS1 Cutout Basename: {ps1_cutout_basename}")

        # Legacy Survey (LS) cutouts
        ls_cutout_path = os.path.join(cutout_dir, f"{source_id}_ls.png")
        ls_cutout_basename = None
        if os.path.exists(ls_cutout_path):
             ls_cutout_basename = os.path.basename(ls_cutout_path)
        else:
             if ra is not None and dec is not None:
                 # Try to plot/fetch with retries if it doesn't exist
                 for attempt in range(3): # Reduced retries slightly
                      ls_cutout = plot_ls_cutout(kowalski_session, cutout_dir, source_id, ra, dec)
                      if ls_cutout and os.path.exists(ls_cutout):
                          ls_cutout_basename = os.path.basename(ls_cutout)
                          break # Success
                      time.sleep(1) # Wait 1 second before retrying
             else:
                 logging.warning(f"Skipping LS cutout for {source_id} due to missing RA/Dec.")
        logging.debug(f"LS Cutout Basename: {ls_cutout_basename}")


        # Fetch Legacy Survey crossmatch data
        legacy_survey_data = pd.DataFrame() # Default to empty
        if ra is not None and dec is not None:
             legacy_survey_data = xmatch_ls(ra, dec)
             logging.debug(f"Legacy Survey Data: {legacy_survey_data.shape[0]} sources found.")
        else:
             logging.warning(f"Skipping Legacy Survey crossmatch for {source_id} due to missing RA/Dec.")

        legacy_amount = legacy_survey_data.shape[0]
        legacy_data = []
        if legacy_amount > 0:
            legacy_closest = legacy_survey_data.iloc[0]
            legacy_data = [
                legacy_closest.get('sep_arcsec', 'N/A'), # Use .get for safety
                legacy_closest.get('pa_degree', 'N/A'),
                legacy_closest.get('z_phot_median', 'N/A'),
                legacy_closest.get('z_phot_l68', 'N/A'),
                legacy_closest.get('z_phot_u68', 'N/A'),
                legacy_closest.get('type', 'N/A')
            ]
            # Round numeric values if they are not 'N/A'
            legacy_data = [round(v, 2) if isinstance(v, (int, float)) else v for v in legacy_data]


        # SDSS crossmatch data (check original packets)
        sdss_data = None
        # Use the original_dets_packets list here
        if original_dets_packets and 'candidate' in original_dets_packets[0]:
            first_candidate = original_dets_packets[0]['candidate']
            # Check for existence and valid values before creating dict
            if first_candidate.get('ssdistnr') is not None and first_candidate.get('ssdistnr') > -999 and \
               first_candidate.get('ssmagnr') is not None and first_candidate.get('ssmagnr') > -999:
                sdss_data = {
                    'ssdistnr': first_candidate['ssdistnr'],
                    'ssmagnr': first_candidate['ssmagnr']
                }
        logging.debug(f"SDSS Data (from first packet): {sdss_data}")


        # PS1 crossmatch summary (use comprehensive LC detections)
        ps1_dist = None
        ps1_sgs = None
        if not detections_lc_df.empty and 'distpsnr1' in detections_lc_df.columns and 'sgscore1' in detections_lc_df.columns:
             # Filter out invalid values (-999) before finding the minimum distance
             valid_ps1_df = detections_lc_df[
                 (detections_lc_df['distpsnr1'].notna()) & (detections_lc_df['distpsnr1'] > -999) &
                 (detections_lc_df['sgscore1'].notna()) & (detections_lc_df['sgscore1'] > -999) &
                 (detections_lc_df['distpsnr1'] <= 3) # Apply 3 arcsec filter
             ].copy()

             if not valid_ps1_df.empty:
                 # Find the row with the minimum distpsnr1
                 closest_ps1_row = valid_ps1_df.loc[valid_ps1_df['distpsnr1'].idxmin()]
                 ps1_dist = closest_ps1_row['distpsnr1']
                 ps1_sgs = closest_ps1_row['sgscore1']
        logging.debug(f"PS1 Crossmatch (closest within 3\"): dist={ps1_dist}, sgs={ps1_sgs}")


        # Create the polar plot using the comprehensive detections_lc_df
        # Ensure detections_lc_df has 'ra' and 'dec' columns
        ztf_alerts_for_polar = pd.DataFrame()
        if not detections_lc_df.empty and 'ra' in detections_lc_df.columns and 'dec' in detections_lc_df.columns:
             ztf_alerts_for_polar = detections_lc_df.copy()
        else:
            logging.warning(f"Cannot generate polar plot data for {source_id} due to missing RA/Dec in detections_lc_df.")

        polar_plot_rel_path = os.path.join('static', 'light_curves', f'{source_id}_polar_plot.html')
        polar_big_plot_rel_path = os.path.join('static', 'light_curves', f'{source_id}_big_polar_plot.html')
        polar_plot_out_rel_path = os.path.join('static', 'light_curves', f'{source_id}_polar_plot_out.html')
        polar_big_plot_out_rel_path = os.path.join('static', 'light_curves', f'{source_id}_big_polar_plot_out.html')

        # Check if plots need generating
        polar_plot_abs_path = os.path.join(basedir, polar_plot_rel_path)
        polar_big_plot_abs_path = os.path.join(basedir, polar_big_plot_rel_path)
        polar_plot_out_abs_path = os.path.join(basedir, polar_plot_out_rel_path)
        polar_big_plot_out_abs_path = os.path.join(basedir, polar_big_plot_out_rel_path)

        if not os.path.exists(polar_plot_abs_path) or not os.path.exists(polar_plot_out_abs_path):
             if not ztf_alerts_for_polar.empty and ra is not None and dec is not None:
                 logging.debug(f"Generating polar plots for {source_id} using {len(ztf_alerts_for_polar)} detections.")
                 # Analyze PS1 photoz needs valid RA/Dec
                 ra_ps1, dec_ps1 = analyze_ps1_photoz(kowalski_session, source_id, ra, dec, 3)

                 # Pass the DataFrame derived from the comprehensive light curve
                 plot_polar_coordinates(ztf_alerts_for_polar, ra_ps1, dec_ps1, legacy_survey_data, ra, dec, polar_plot_abs_path, xlim=(-2, 2), ylim=(-2, 2), point_size=15)
                 plot_polar_coordinates(ztf_alerts_for_polar, ra_ps1, dec_ps1, legacy_survey_data, ra, dec, polar_plot_out_abs_path, xlim=(-10, 10), ylim=(-10, 10), point_size=15)
                 plot_big_polar_coordinates(ztf_alerts_for_polar, ra_ps1, dec_ps1, legacy_survey_data, ra, dec, polar_big_plot_abs_path, xlim=(-2, 2), ylim=(-2, 2), point_size=17)
                 plot_big_polar_coordinates(ztf_alerts_for_polar, ra_ps1, dec_ps1, legacy_survey_data, ra, dec, polar_big_plot_out_abs_path, xlim=(-10, 10), ylim=(-10, 10), point_size=17)
             else:
                 logging.warning(f"Skipping polar plot generation for {source_id} due to missing data (alerts or RA/Dec).")

        # Retrieve classifications and determine the most confident classification
        classifications = Classification.query.filter_by(source_id=source_id).all()
        classification_counts = defaultdict(lambda: {'count': 0, 'confidence': 0})
        classified_by_users = []

        for classification in classifications:
            user = User.query.get(classification.user_id)
            if user: # Check if user exists
                classified_by_users.append(user.username)
                classification_counts[classification.classification]['count'] += 1
                # Simplified confidence score mapping
                confidence_map = {'Uncertain': 1, 'Probable': 2, 'Confident': 3}
                classification_counts[classification.classification]['confidence'] += confidence_map.get(classification.confidence, 0)


        most_confident_classification = None
        if classification_counts:
            # Sort first by confidence score (desc), then by count (desc)
            sorted_classifications = sorted(
                classification_counts.items(),
                key=lambda item: (item[1]['confidence'], item[1]['count']),
                reverse=True
            )
            most_confident_classification = sorted_classifications[0][0]

        logging.debug(f"Most Confident Classification: {most_confident_classification}")

        # Convert RA/Dec to sexagesimal strings
        ra_str = 'N/A'
        dec_str = 'N/A'
        if ra is not None and dec is not None:
            try:
                 coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                 ra_str = coord.ra.to_string(unit=u.hour, sep=':', precision=4, pad=True)
                 dec_str = coord.dec.to_string(unit=u.degree, sep=':', precision=4, pad=True, alwayssign=True)
            except Exception as coord_err:
                 logging.error(f"Error converting RA/Dec to string for {source_id}: {coord_err}")


        logging.debug(f"RA (string): {ra_str}, Dec (string): {dec_str}")

        # Return the core data needed
        data = {
            "source_id": source_id,
            "ra": ra_str, # Use formatted string
            "dec": dec_str, # Use formatted string
            "scat_sep": scat_sep,
            "galactic_l": galactic_l,
            "galactic_b": galactic_b,
            "span": span, # Use span calculated from original packets
            "ecliptic_lon": ecliptic_lon,
            "ecliptic_lat": ecliptic_lat,
            # "dets": original_dets_packets, # No longer needed directly by template
            "alert_count": alert_count,  # Use the count from LC detections
            "med_drb": med_drb,
            "min_drb": min_drb,
            "max_drb": max_drb,
            "avg_drb": avg_drb,
            "ps1_dist": ps1_dist, # Use closest from LC
            "ps1_sgs": ps1_sgs,   # Use closest from LC
            "wise_plot": wise_filename, # Use basename
            "plot_filename": plot_filename_rel, # Use relative path for template
            "plot_filename_zoomed": plot_filename_zoomed_rel, # Use relative path
            "plot_big_filename": plot_big_filename_rel, # Use relative path
            "plot_big_filename_zoomed": plot_big_filename_zoomed_rel, # Use relative path
            "ztf_cutout": ztf_cutout_basenames_for_template, # Pass the list with Nones
            "ps1_cutout": ps1_cutout_basename, # Use basename
            "ls_cutout": ls_cutout_basename, # Use basename
            "legacy_amount": legacy_amount,
            "legacy_data": legacy_data,
            "sdss_data": sdss_data,
            "polar_plot": polar_plot_rel_path, # Use relative path
            "polar_big_plot": polar_big_plot_rel_path, # Use relative path
            "polar_plot_out": polar_plot_out_rel_path, # Use relative path
            "polar_big_plot_out": polar_big_plot_out_rel_path, # Use relative path
            "classifications": classifications,
            "classified_by_users": classified_by_users,
            "most_confident_classification": most_confident_classification,
            # "ra_str": ra_str, # Included in 'ra' key now
            # "dec_str": dec_str, # Included in 'dec' key now
            # Pass the processed raw_alerts list derived from the comprehensive LC
            "raw_alerts": raw_alerts
        }

        return data

    except Exception as e:
        logging.error(f"Error during fetch_transient_data for {source_id}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None # Return None on error

# ... rest of utils.py ...