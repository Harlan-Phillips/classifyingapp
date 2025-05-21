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
                "candidate.drb": 1,
                "candidate.diffmaglim": 1
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
    lc = pd.DataFrame() # Initialize an empty DataFrame
    df_alerts = pd.DataFrame()
    df_forced = pd.DataFrame()
    df_prv_candidates = pd.DataFrame()

    try:
        # The alerts
        dets = get_dets(s, name)
        if dets:
            df_alerts = pd.DataFrame(dets)
            # Ensure essential alert columns exist, even if empty
            for col in ['ra', 'dec', 'magpsf', 'sigmapsf', 'fid', 'jd', 'programid', 'field', 'ssdistnr', 'ssmagnr', 'sgscore1', 'distpsnr1', 'isdiffpos', 'diffmaglim']:
                if col not in df_alerts.columns:
                    df_alerts[col] = np.nan 
            df_alerts['origin'] = 'alert'
            logging.debug(f"Alerts for {name}: {len(df_alerts)} rows. Columns: {df_alerts.columns.tolist()}")

        # Forced photometry for previous detections
        det_prv_forced = get_prv_dets_forced(s, name)
        if det_prv_forced:
            df_forced = pd.DataFrame(det_prv_forced)
            # Ensure essential forced phot columns exist
            for col in ['jd', 'fid', 'mag', 'magerr', 'maglim', 'ra', 'dec', 'programid', 'field', 'forcediffimflux', 'forcediffimfluxunc']:
                if col not in df_forced.columns:
                    df_forced[col] = np.nan
            df_forced['origin'] = 'forced'
            logging.debug(f"Forced photometry for {name}: {len(df_forced)} rows. Columns: {df_forced.columns.tolist()}")

        # Previous candidates (includes non-detections/upper limits from alerts)
        prv_candidates = get_prv_dets(s, name)
        if prv_candidates:
            df_prv_candidates = pd.DataFrame(prv_candidates)
            # Ensure essential prv_candidate columns exist
            for col in ['jd', 'fid', 'magpsf', 'sigmapsf', 'maglim', 'programid', 'field', 'ra', 'dec', 'isdiffpos']:
                 if col not in df_prv_candidates.columns:
                    df_prv_candidates[col] = np.nan
            df_prv_candidates['origin'] = 'prv_candidate'
            logging.debug(f"Previous candidates for {name}: {len(df_prv_candidates)} rows. Columns: {df_prv_candidates.columns.tolist()}")

        # --- Start Merging --- 
        if not df_alerts.empty:
            lc = df_alerts.copy()
            # Rename alert columns to intermediate standard names if they exist
            if 'magpsf' in lc.columns: lc['mag_intermediate'] = lc['magpsf']
            if 'sigmapsf' in lc.columns: lc['magerr_intermediate'] = lc['sigmapsf']
            if 'ra' in lc.columns: lc['ra_intermediate'] = lc['ra']
            if 'dec' in lc.columns: lc['dec_intermediate'] = lc['dec']
            if 'diffmaglim' in lc.columns: lc['maglim_intermediate'] = lc['diffmaglim'] # Added this line for alert maglim
        
        if not df_forced.empty:
            # Rename forced phot columns to intermediate standard names if they exist
            if 'mag' in df_forced.columns: df_forced['mag_intermediate'] = df_forced['mag']
            if 'magerr' in df_forced.columns: df_forced['magerr_intermediate'] = df_forced['magerr']
            if 'maglim' in df_forced.columns: df_forced['maglim_intermediate'] = df_forced['maglim'] # maglim specific to forced/prv_candidates
            if 'ra' in df_forced.columns: df_forced['ra_intermediate'] = df_forced['ra']
            if 'dec' in df_forced.columns: df_forced['dec_intermediate'] = df_forced['dec']

            if lc.empty:
                lc = df_forced.copy()
            else:
                # Merge based on JD, prioritizing alert data for conflicts
                lc = pd.merge(lc, df_forced, on=['jd', 'fid'], how='outer', suffixes=('_alert', '_forced'))
                # Coalesce common columns, prioritizing alert data if not already done by suffix handling
                for col_base in ['mag_intermediate', 'magerr_intermediate', 'ra_intermediate', 'dec_intermediate', 'programid', 'field', 'origin']:
                    if f'{col_base}_alert' in lc.columns and f'{col_base}_forced' in lc.columns:
                        lc[col_base] = lc[f'{col_base}_alert'].combine_first(lc[f'{col_base}_forced'])
                        lc.drop(columns=[f'{col_base}_alert', f'{col_base}_forced'], inplace=True)
                    elif f'{col_base}_alert' in lc.columns: # Only alert data existed
                        lc.rename(columns={f'{col_base}_alert': col_base}, inplace=True)
                    elif f'{col_base}_forced' in lc.columns: # Only forced data existed
                        lc.rename(columns={f'{col_base}_forced': col_base}, inplace=True)
                # Handle maglim from forced photometry
                if 'maglim_intermediate_forced' in lc.columns:
                    lc['maglim'] = lc['maglim_intermediate_forced']
                    if 'maglim_intermediate_alert' in lc.columns and 'maglim' in lc.columns:
                         lc['maglim'] = lc['maglim_intermediate_alert'].combine_first(lc['maglim'])
                    # If maglim_intermediate (from alerts) exists and maglim_intermediate_forced also, prioritize alert
                    elif 'maglim_intermediate' in lc.columns and 'maglim_intermediate_forced' in lc.columns: # Check if this intermediate is from alert
                        lc['maglim'] = lc['maglim_intermediate'].combine_first(lc['maglim_intermediate_forced'])
                    elif 'maglim_intermediate' in lc.columns: # Only alert maglim exists at this stage
                        lc['maglim'] = lc['maglim_intermediate']
                    lc.drop(columns=[col for col in ['maglim_intermediate_alert', 'maglim_intermediate_forced', 'maglim_intermediate'] if col in lc.columns and col != 'maglim'], inplace=True) # ensure intermediate from alert is also dropped if used
                elif 'maglim_intermediate' in df_forced.columns: # if lc was empty and df_forced was copied
                    lc['maglim'] = df_forced['maglim_intermediate']

        if not df_prv_candidates.empty:
            # Rename prv_candidate columns to intermediate standard names
            if 'magpsf' in df_prv_candidates.columns: df_prv_candidates['mag_intermediate'] = df_prv_candidates['magpsf']
            if 'sigmapsf' in df_prv_candidates.columns: df_prv_candidates['magerr_intermediate'] = df_prv_candidates['sigmapsf']
            if 'maglim' in df_prv_candidates.columns: df_prv_candidates['maglim_intermediate'] = df_prv_candidates['maglim']
            if 'ra' in df_prv_candidates.columns: df_prv_candidates['ra_intermediate'] = df_prv_candidates['ra']
            if 'dec' in df_prv_candidates.columns: df_prv_candidates['dec_intermediate'] = df_prv_candidates['dec']

            if lc.empty:
                lc = df_prv_candidates.copy()
            else:
                # Merge prv_candidates, treating them mostly as potential upper limits or context
                lc = pd.merge(lc, df_prv_candidates, on=['jd', 'fid'], how='outer', suffixes=('_current', '_prv'))
                for col_base in ['mag_intermediate', 'magerr_intermediate', 'maglim_intermediate', 'ra_intermediate', 'dec_intermediate', 'programid', 'field', 'origin', 'isdiffpos']:
                    if f'{col_base}_current' in lc.columns and f'{col_base}_prv' in lc.columns:
                        lc[col_base] = lc[f'{col_base}_current'].combine_first(lc[f'{col_base}_prv'])
                        lc.drop(columns=[f'{col_base}_current', f'{col_base}_prv'], inplace=True)
                    elif f'{col_base}_current' in lc.columns:
                        lc.rename(columns={f'{col_base}_current': col_base}, inplace=True)
                    elif f'{col_base}_prv' in lc.columns:
                        lc.rename(columns={f'{col_base}_prv': col_base}, inplace=True)
                # Ensure maglim is properly coalesced
                if 'maglim_intermediate_prv' in lc.columns and 'maglim' not in lc.columns:
                     lc['maglim'] = lc['maglim_intermediate_prv']
                elif 'maglim_intermediate_prv' in lc.columns and 'maglim' in lc.columns:
                     lc['maglim'] = lc['maglim_intermediate_prv'].combine_first(lc['maglim'])
                if 'maglim_intermediate_current' in lc.columns and 'maglim' not in lc.columns:
                     lc['maglim'] = lc['maglim_intermediate_current']
                elif 'maglim_intermediate_current' in lc.columns and 'maglim' in lc.columns:
                     lc['maglim'] = lc['maglim_intermediate_current'].combine_first(lc['maglim'])
                # Drop intermediate maglim columns from prv_candidates merge
                lc.drop(columns=[col for col in ['maglim_intermediate_current', 'maglim_intermediate_prv', 'maglim_intermediate'] if col in lc.columns and col != 'maglim'], inplace=True)
        
        # At this point, lc should have 'mag_intermediate', 'magerr_intermediate', 'ra_intermediate', 'dec_intermediate', and potentially 'maglim'
        # Create final 'mag_final' and 'emag_final' etc.
        if 'mag_intermediate' in lc.columns:
            lc['mag_final'] = lc['mag_intermediate']
        else:
            lc['mag_final'] = np.nan # Ensure column exists

        if 'magerr_intermediate' in lc.columns:
            lc['emag_final'] = lc['magerr_intermediate']
        else:
            lc['emag_final'] = np.nan # Ensure column exists

        if 'ra_intermediate' in lc.columns:
            lc['ra'] = lc['ra_intermediate'] # Final RA column
        elif 'ra' not in lc.columns: # If no intermediate and no original 'ra' (e.g. from empty alerts only)
            lc['ra'] = np.nan

        if 'dec_intermediate' in lc.columns:
            lc['dec'] = lc['dec_intermediate'] # Final Dec column
        elif 'dec' not in lc.columns:
            lc['dec'] = np.nan

        # Ensure other essential columns for the final list exist, even if they are all NaN
        # These are columns expected by `fetch_transient_data` for the final_columns_to_keep list
        # or for other calculations like `isdet`
        ensure_cols = ['jd', 'fid', 'programid', 'field', 'ssdistnr', 'ssmagnr', 'sgscore1', 'distpsnr1', 'isdiffpos', 'origin', 'maglim']
        for col in ensure_cols:
            if col not in lc.columns:
                lc[col] = np.nan

        # Determine detections based on mag_final (actual measurement) vs maglim (upper limit)
        if 'mag_final' in lc.columns and 'maglim' in lc.columns:
            # A row is a detection if mag_final is not NaN
            # A row is an upper limit if mag_final is NaN but maglim is not NaN
            lc['isdet'] = lc['mag_final'].notna()
        elif 'mag_final' in lc.columns: # Only mag_final exists (e.g. no upper limits data)
             lc['isdet'] = lc['mag_final'].notna()
        elif not lc.empty: # lc is not empty but lacks mag_final (should be rare if defaults are set)
             lc['isdet'] = False 
        elif lc.empty: # lc is truly empty, create isdet for schema consistency if needed by final_columns_to_keep
            if 'isdet' in final_columns_to_keep:
                 lc['isdet'] = pd.Series(dtype=bool) # Empty boolean series
        
        # Final cleanup - drop rows where both mag_final and maglim are NaN, if both columns exist
        if 'mag_final' in lc.columns and 'maglim' in lc.columns:
            final_drop_mask = lc['mag_final'].isna() & lc['maglim'].isna()
            if not lc.empty: lc = lc[~final_drop_mask]
        elif 'mag_final' in lc.columns: # Only mag_final exists, drop if it's NaN
            if not lc.empty: lc = lc[lc['mag_final'].notna()]

        # --- Final Column Selection and Cleanup BEFORE returning --- #
        # Define the columns that are absolutely expected in the output of get_lc
        final_columns_to_keep = [
            'jd', 'ra', 'dec', 'fid', 'mag_final', 'emag_final', 'maglim', 
            'isdiffpos', 'programid', 'field', 'ssdistnr', 'ssmagnr', 'sgscore1', 'distpsnr1', 'origin', 'isdet'
        ]
        
        # Create a new DataFrame with only these columns, filling missing ones with NaN
        # This ensures consistency in the returned DataFrame structure.
        if not lc.empty:
            # First, ensure all columns to keep exist in lc, adding them with NaNs if not.
            for col in final_columns_to_keep:
                if col not in lc.columns:
                    lc[col] = np.nan
            # Then, select only these columns in the specified order.
            lc = lc[final_columns_to_keep]
        else:
            # If lc became empty, return an empty DataFrame but with the correct columns for schema consistency
            lc = pd.DataFrame(columns=final_columns_to_keep)

        logging.debug(f"get_lc for {name} returning DataFrame with shape {lc.shape} and columns {lc.columns.tolist()}")
        if lc.empty:
            logging.warning(f"get_lc for {name} is returning an empty DataFrame.")

    except Exception as e:
        logging.error(f"Critical error in get_lc for {name}: {e}", exc_info=True)
        # In case of any unexpected error, return an empty DataFrame with the correct schema
        lc = pd.DataFrame(columns=final_columns_to_keep)
        logging.warning(f"get_lc for {name} returning empty DataFrame due to exception.")

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
    logging.debug(f"plot_light_curve for {source_id} (span={span}) received lc with {len(lc)} rows. Columns: {lc.columns.tolist()}")
    if not lc.empty:
        logging.debug(f"Initial lc head:\n{lc.head()}")
        logging.debug(f"Value counts for 'isdet' in initial lc:\n{lc['isdet'].value_counts(dropna=False)}")
        if 'maglim' in lc.columns:
            logging.debug(f"Number of non-NaN maglim in initial lc: {lc['maglim'].notna().sum()}")
            logging.debug(f"Number of NaN maglim in initial lc: {lc['maglim'].isna().sum()}")
        else:
            logging.debug("'maglim' column not present in initial lc")
        if 'mag_final' in lc.columns:
            logging.debug(f"Number of non-NaN mag_final in initial lc: {lc['mag_final'].notna().sum()}")
        else:
            logging.debug("'mag_final' column not present in initial lc")


    # Preserve existing data prep logic
    non_dets = lc[lc['isdet'] == False].copy() # Removed & (lc['maglim'] > 1)
    logging.debug(f"After filtering for isdet == False, non_dets has {len(non_dets)} rows. Columns: {non_dets.columns.tolist()}")
    if not non_dets.empty:
        logging.debug(f"non_dets head (after isdet == False):\n{non_dets.head()}")
        if 'maglim' in non_dets.columns:
            logging.debug(f"Number of non-NaN maglim in non_dets (after isdet == False): {non_dets['maglim'].notna().sum()}")
            logging.debug(f"Number of NaN maglim in non_dets (after isdet == False): {non_dets['maglim'].isna().sum()}")


    lc_detections = lc.dropna(subset=['mag_final']) if 'mag_final' in lc.columns and lc['mag_final'].isna().sum() > 0 else lc.copy() # Changed to lc_detections
    logging.debug(f"After dropna on 'mag_final' for detections, lc_detections has {len(lc_detections)} rows.")
    if not lc_detections.empty:
        logging.debug(f"lc_detections head:\n{lc_detections.head()}")

    non_dets = non_dets.dropna(subset=['maglim']) if 'maglim' in non_dets.columns and non_dets['maglim'].isna().sum() > 0 else non_dets
    logging.debug(f"After dropna on 'maglim' for non_dets, non_dets has {len(non_dets)} rows.")
    if not non_dets.empty:
        logging.debug(f"non_dets head (after maglim dropna):\n{non_dets.head()}")
        logging.debug(f"Non-detections to be plotted: {len(non_dets)}")
        if 'maglim' in non_dets.columns:
            logging.debug(f"Unique maglim values in final non_dets: {non_dets['maglim'].unique()}")


    # Convert JD to MJD
    from astropy.time import Time
    if not lc_detections.empty and 'jd' in lc_detections.columns: # Use lc_detections
        lc_detections['mjd'] = Time(lc_detections['jd'], format='jd').mjd - 58000
    if not non_dets.empty and 'jd' in non_dets.columns:
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
    for band in lc_detections['fid'].unique(): # Use lc_detections
        band_data = lc_detections[lc_detections['fid'] == band] # Use lc_detections
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
    y_min_val = lc_detections['mag_final'].min() if not lc_detections.empty and 'mag_final' in lc_detections.columns and lc_detections['mag_final'].notna().any() else 25  # Default if empty or all NaN
    y_max_val = lc_detections['mag_final'].max() if not lc_detections.empty and 'mag_final' in lc_detections.columns and lc_detections['mag_final'].notna().any() else 15 # Default if empty or all NaN
    
    # Consider non_dets for y-axis range as well
    if not non_dets.empty and 'maglim' in non_dets.columns and non_dets['maglim'].notna().any():
        y_min_val = min(y_min_val, non_dets['maglim'].min())
        y_max_val = max(y_max_val, non_dets['maglim'].max())

    y_min = y_min_val - 0.5
    y_max = y_max_val + 0.5


    if span == 'detections' and not lc_detections.empty and 'mjd' in lc_detections.columns: # Use lc_detections
        diff = lc_detections['mjd'].max() - lc_detections['mjd'].min() # Use lc_detections
        if diff < 1:
            x_min = lc_detections['mjd'].min() - (diff * 1.2) # Use lc_detections
            x_max = lc_detections['mjd'].max() + (diff * 1.2) # Use lc_detections
        else:
            x_min = lc_detections['mjd'].min() - 0.5 # Use lc_detections
            x_max = lc_detections['mjd'].max() + 0.5 # Use lc_detections
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
    logging.debug(f"plot_big_light_curve for {source_id} (span={span}) received lc with {len(lc)} rows. Columns: {lc.columns.tolist()}")
    if not lc.empty:
        logging.debug(f"Initial lc head (big plot):\n{lc.head()}")
        logging.debug(f"Value counts for 'isdet' in initial lc (big plot):\n{lc['isdet'].value_counts(dropna=False)}")
        if 'maglim' in lc.columns:
            logging.debug(f"Number of non-NaN maglim in initial lc (big plot): {lc['maglim'].notna().sum()}")
        else:
            logging.debug("'maglim' column not present in initial lc (big plot)")
        if 'mag_final' in lc.columns:
            logging.debug(f"Number of non-NaN mag_final in initial lc (big plot): {lc['mag_final'].notna().sum()}")
        else:
            logging.debug("'mag_final' column not present in initial lc (big plot)")

    # Preserve existing data prep logic
    non_dets = lc[lc['isdet'] == False].copy() # Removed & (lc['maglim'] > 1)
    logging.debug(f"After filtering for isdet == False, non_dets (big plot) has {len(non_dets)} rows. Columns: {non_dets.columns.tolist()}")
    if not non_dets.empty:
        logging.debug(f"non_dets head (big plot, after isdet == False):\n{non_dets.head()}")
        if 'maglim' in non_dets.columns:
             logging.debug(f"Number of non-NaN maglim in non_dets (big plot, after isdet == False): {non_dets['maglim'].notna().sum()}")

    lc_detections = lc.dropna(subset=['mag_final']) if 'mag_final' in lc.columns and lc['mag_final'].isna().sum() > 0 else lc.copy() # Changed to lc_detections
    logging.debug(f"After dropna on 'mag_final' for detections, lc_detections (big plot) has {len(lc_detections)} rows.")

    non_dets = non_dets.dropna(subset=['maglim']) if 'maglim' in non_dets.columns and non_dets['maglim'].isna().sum() > 0 else non_dets
    logging.debug(f"After dropna on 'maglim' for non_dets, non_dets (big plot) has {len(non_dets)} rows.")
    if not non_dets.empty:
        logging.debug(f"non_dets head (big plot, after maglim dropna):\n{non_dets.head()}")
        logging.debug(f"Non-detections to be plotted (big plot): {len(non_dets)}")


    from astropy.time import Time
    if not lc_detections.empty and 'jd' in lc_detections.columns: # Use lc_detections
        lc_detections['mjd'] = Time(lc_detections['jd'], format='jd').mjd - 58000
    if not non_dets.empty and 'jd' in non_dets.columns:
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
    for band in lc_detections['fid'].unique(): # Use lc_detections
        band_data = lc_detections[lc_detections['fid'] == band] # Use lc_detections
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

    y_min_val = lc_detections['mag_final'].min() if not lc_detections.empty and 'mag_final' in lc_detections.columns and lc_detections['mag_final'].notna().any() else 25
    y_max_val = lc_detections['mag_final'].max() if not lc_detections.empty and 'mag_final' in lc_detections.columns and lc_detections['mag_final'].notna().any() else 15

    if not non_dets.empty and 'maglim' in non_dets.columns and non_dets['maglim'].notna().any():
        y_min_val = min(y_min_val, non_dets['maglim'].min())
        y_max_val = max(y_max_val, non_dets['maglim'].max())
    
    y_min = y_min_val - 0.5
    y_max = y_max_val + 0.5


    if span == 'detections' and not lc_detections.empty and 'mjd' in lc_detections.columns: # Use lc_detections
        diff = lc_detections['mjd'].max() - lc_detections['mjd'].min() # Use lc_detections
        if diff < 1:
            x_min = lc_detections['mjd'].min() - (diff * 1.2) # Use lc_detections
            x_max = lc_detections['mjd'].max() + (diff * 1.2) # Use lc_detections
        else:
            x_min = lc_detections['mjd'].min() - 0.5 # Use lc_detections
            x_max = lc_detections['mjd'].max() + 0.5 # Use lc_detections
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
    ax.invert_xaxis()  

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
    ax.invert_xaxis() 

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
    # Initialize all potential data fields to None or default values
    # This ensures the data dictionary always has a consistent structure.
    ra, dec, scat_sep = None, None, None
    ra_str, dec_str = "N/A", "N/A"
    galactic_l, galactic_b = "N/A", "N/A"
    ecliptic_lon, ecliptic_lat = "N/A", "N/A"
    original_dets_packets = []
    light_curve_df = pd.DataFrame()
    detections_lc_df = pd.DataFrame()
    alert_count = 0
    raw_alerts = []
    med_drb, min_drb, max_drb, avg_drb = "N/A", "N/A", "N/A", "N/A"
    span = "N/A"
    wise_filename = None
    plot_filename_rel, plot_filename_zoomed_rel = None, None
    plot_big_filename_rel, plot_big_filename_zoomed_rel = None, None
    ztf_cutout_basenames_for_template = [None] * 8 # Assuming 8 potential cutouts
    ps1_cutout_basename = None
    ls_cutout_basename = None
    legacy_survey_data = pd.DataFrame()
    legacy_amount = 0
    legacy_data = []
    sdss_data = None
    ps1_dist, ps1_sgs = "N/A", "N/A"
    polar_plot_rel_path, polar_big_plot_rel_path = None, None
    polar_plot_out_rel_path, polar_big_plot_out_rel_path = None, None
    classifications = []
    classified_by_users = []
    most_confident_classification = None
    general_error_message = None # To store any top-level error

    try:
        # Attempt to fetch primary positional data
        try:
            ra, dec, scat_sep = get_pos(kowalski_session, source_id)
            if ra is None or dec is None:
                logging.warning(f"Could not determine position for {source_id} via get_pos. Some features will be unavailable.")
                # general_error_message = "Primary position not found. Some data may be unavailable."
                # Do not return yet, try to get other data
            else:
                logging.debug(f"RA: {ra}, Dec: {dec}, Scatter Separation: {scat_sep}")
                # Convert RA/Dec to sexagesimal strings if available
                try:
                    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                    ra_str = coord.ra.to_string(unit=u.hour, sep=':', precision=4, pad=True)
                    dec_str = coord.dec.to_string(unit=u.degree, sep=':', precision=4, pad=True, alwayssign=True)
                except Exception as coord_err:
                    logging.error(f"Error converting RA/Dec to string for {source_id}: {coord_err}")
                    ra_str, dec_str = "Error", "Error"
                
                # Fetch galactic coordinates
                galactic_l, galactic_b = get_galactic(ra, dec)
                logging.debug(f"Galactic Coordinates - l: {galactic_l}, b: {galactic_b}")

                # Fetch ecliptic coordinates
                ecliptic_lon, ecliptic_lat = get_ecliptic(ra, dec)
                logging.debug(f"Ecliptic Coordinates - lon: {ecliptic_lon}, lat: {ecliptic_lat}")

        except Exception as pos_exc:
            logging.error(f"Critical error fetching position for {source_id}: {pos_exc}")
            general_error_message = "Error fetching primary position. Most data will be unavailable."
            # Still proceed to create the data dict with what we have (mostly Nones)

        # Fetch the original public alerts (needed for DRB and span, even if RA/Dec failed)
        try:
            original_dets_packets = get_dets(kowalski_session, source_id) or []
            logging.debug(f"Original Public Alerts (dets packets): {len(original_dets_packets)} found")
        except Exception as dets_exc:
            logging.error(f"Error fetching original alert packets for {source_id}: {dets_exc}")
            # general_error_message = general_error_message or "Error fetching alert packets."


        # Fetch comprehensive light curve data (includes alerts, forced phot, prv dets)
        try:
            light_curve_df = get_lc(kowalski_session, source_id)
            if light_curve_df is None or light_curve_df.empty:
                 logging.warning(f"Comprehensive light curve (get_lc) returned empty or None for {source_id}.")
                 if not original_dets_packets: # Only if get_lc failed AND no original packets
                     general_error_message = general_error_message or "No light curve or alert data found."
                 # Fallback: use original alert packets if get_lc fails and packets exist
                 if original_dets_packets:
                     light_curve_df = pd.DataFrame([p['candidate'] for p in original_dets_packets])
                     # Add necessary columns if missing from original packets
                     if 'mag_final' not in light_curve_df.columns and 'magpsf' in light_curve_df.columns:
                         light_curve_df['mag_final'] = light_curve_df['magpsf']
                     if 'emag_final' not in light_curve_df.columns and 'sigmapsf' in light_curve_df.columns:
                         light_curve_df['emag_final'] = light_curve_df['sigmapsf']
                     if 'maglim' not in light_curve_df.columns and 'diffmaglim' in light_curve_df.columns: # Added this block
                         light_curve_df['maglim'] = light_curve_df['diffmaglim']
                     if 'isdet' not in light_curve_df.columns:
                          light_curve_df['isdet'] = True 
                     if 'origin' not in light_curve_df.columns:
                          light_curve_df['origin'] = 'alert_packet'
                 else:
                     light_curve_df = pd.DataFrame() # Ensure it's an empty DF
            logging.debug(f"Comprehensive Light Curve DataFrame head:\n{light_curve_df.head() if not light_curve_df.empty else 'Empty LC DataFrame'}")
        except Exception as lc_exc:
            logging.error(f"Error fetching comprehensive light curve for {source_id}: {lc_exc}")
            general_error_message = general_error_message or "Error processing light curve data."
            light_curve_df = pd.DataFrame() # Ensure it's an empty DF on error


        # Calculate alert count based on actual detections in the comprehensive LC DataFrame
        if not light_curve_df.empty and 'isdet' in light_curve_df.columns:
             try:
                 detections_lc_df = light_curve_df[light_curve_df['isdet'] == True].copy()
             except Exception as e:
                 logging.error(f"Error filtering light_curve_df by 'isdet': {e}. LC columns: {light_curve_df.columns}")
                 if 'isdet' in light_curve_df.columns:
                     detections_lc_df = light_curve_df[light_curve_df['isdet'].notna()].copy() 
                 else:
                    detections_lc_df = light_curve_df.copy() 
        else:
            detections_lc_df = pd.DataFrame()


        alert_count = detections_lc_df.shape[0]
        logging.debug(f"Final Alert Count (from LC detections): {alert_count}")

        if 'origin' not in detections_lc_df.columns and not detections_lc_df.empty:
            detections_lc_df['origin'] = 'unknown'
        elif 'origin' not in light_curve_df.columns and not light_curve_df.empty : # if detections_lc_df is empty but light_curve_df is not
             light_curve_df['origin'] = 'unknown'


        # Prepare data for the alert table
        table_columns_map = {
            'jd': 'jd', 'fid': 'fid', 'programid': 'programid', 'field': 'field',
            'ra': 'ra', 'dec': 'dec', 'mag_final': 'magpsf', 'emag_final': 'sigmapsf',
            'ssdistnr': 'ssdistnr', 'ssmagnr': 'ssmagnr', 'sgscore1': 'sgscore1',
            'distpsnr1': 'distpsnr1', 'origin': 'origin'
        }
        raw_alerts_for_table = pd.DataFrame()
        expected_table_cols = list(table_columns_map.values())

        if not detections_lc_df.empty:
            df_for_table = detections_lc_df.copy()
            for source_col in table_columns_map.keys():
                if source_col not in df_for_table.columns:
                    df_for_table[source_col] = np.nan
            df_for_table = df_for_table.rename(columns=table_columns_map)
            final_table_cols = [col for col in expected_table_cols if col in df_for_table.columns]
            if final_table_cols:
                raw_alerts_for_table = df_for_table[final_table_cols]
        
        raw_alerts_for_table = raw_alerts_for_table.replace({pd.NaT: None, np.nan: None})
        raw_alerts = raw_alerts_for_table.to_dict(orient='records')
        logging.debug(f"Final raw_alerts list length = {len(raw_alerts)}")

        # DRB stats and Span (from original packets, as DRB is alert-specific)
        if original_dets_packets:
            try:
                med_drb, min_drb, max_drb, avg_drb = get_drb(kowalski_session, source_id, original_dets_packets)
                span = get_span(kowalski_session, source_id, original_dets_packets)
                logging.debug(f"DRB - Med: {med_drb}, Min: {min_drb}, Max: {max_drb}, Avg: {avg_drb}")
                logging.debug(f"Span (days): {span}")
            except Exception as drb_span_exc:
                logging.error(f"Error getting DRB/Span for {source_id}: {drb_span_exc}")

        # --- Position-dependent data fetching ---
        # Directories for cutouts and plots
        cutout_dir = os.path.join(basedir, 'static', 'cutouts')
        light_cur_dir = os.path.join(basedir, 'static', 'light_curves') # Corrected variable name
        wise_dir = os.path.join(basedir, 'static', 'wise_plots')

        if ra is not None and dec is not None: # Only proceed if we have coordinates
            # WISE plot
            try:
                wise_plot_path = os.path.join(wise_dir, f"{source_id}_wise_plot.html")
                if os.path.exists(wise_plot_path):
                    wise_filename = os.path.basename(wise_plot_path)
                else:
                    wise_plot_result = plot_wise(kowalski_session, source_id, ra, dec, wise_plot_path)
                    if wise_plot_result: wise_filename = os.path.basename(wise_plot_result)
                logging.debug(f"WISE plot: {wise_filename}")
            except Exception as wise_exc:
                logging.error(f"Error with WISE plot for {source_id}: {wise_exc}")

            # Pan-STARRS (PS1) cutouts
            try:
                ps1_cutout_path = os.path.join(cutout_dir, f"{source_id}_ps1.png")
                if os.path.exists(ps1_cutout_path):
                    ps1_cutout_basename = os.path.basename(ps1_cutout_path)
                else:
                    ps1_cutout_obj = plot_ps1_cutout(kowalski_session, cutout_dir, source_id, ra, dec) # Corrected function name
                    ps1_cutout_basename = os.path.basename(ps1_cutout_obj) if ps1_cutout_obj else None
                logging.debug(f"PS1 Cutout: {ps1_cutout_basename}")
            except Exception as ps1_exc:
                logging.error(f"Error with PS1 cutout for {source_id}: {ps1_exc}")

            # Legacy Survey (LS) cutouts
            try:
                ls_cutout_path = os.path.join(cutout_dir, f"{source_id}_ls.png")
                if os.path.exists(ls_cutout_path):
                    ls_cutout_basename = os.path.basename(ls_cutout_path)
                else:
                    for attempt in range(3):
                        ls_cutout_obj = plot_ls_cutout(kowalski_session, cutout_dir, source_id, ra, dec) # Corrected var name
                        if ls_cutout_obj and os.path.exists(ls_cutout_obj):
                            ls_cutout_basename = os.path.basename(ls_cutout_obj)
                            break
                        time.sleep(1)
                logging.debug(f"LS Cutout: {ls_cutout_basename}")
            except Exception as ls_exc:
                logging.error(f"Error with LS cutout for {source_id}: {ls_exc}")
            
            # Legacy Survey crossmatch data
            try:
                legacy_survey_data = xmatch_ls(ra, dec)
                legacy_amount = legacy_survey_data.shape[0]
                if legacy_amount > 0:
                    legacy_closest = legacy_survey_data.iloc[0]
                    legacy_data = [
                        legacy_closest.get('sep_arcsec', 'N/A'), legacy_closest.get('pa_degree', 'N/A'),
                        legacy_closest.get('z_phot_median', 'N/A'), legacy_closest.get('z_phot_l68', 'N/A'),
                        legacy_closest.get('z_phot_u68', 'N/A'), legacy_closest.get('type', 'N/A')
                    ]
                    legacy_data = [round(v, 2) if isinstance(v, (int, float)) else v for v in legacy_data]
                logging.debug(f"Legacy Survey Data: {legacy_amount} sources found.")
            except Exception as ls_xmatch_exc:
                logging.error(f"Error with Legacy Survey xmatch for {source_id}: {ls_xmatch_exc}")

            # Polar plots
            try:
                polar_plot_rel_path = os.path.join('static', 'light_curves', f'{source_id}_polar_plot.html')
                polar_big_plot_rel_path = os.path.join('static', 'light_curves', f'{source_id}_big_polar_plot.html')
                polar_plot_out_rel_path = os.path.join('static', 'light_curves', f'{source_id}_polar_plot_out.html')
                polar_big_plot_out_rel_path = os.path.join('static', 'light_curves', f'{source_id}_big_polar_plot_out.html')

                polar_plot_abs_path = os.path.join(basedir, polar_plot_rel_path)
                # ... (similar for other polar plot paths)
                polar_big_plot_abs_path = os.path.join(basedir, polar_big_plot_rel_path)
                polar_plot_out_abs_path = os.path.join(basedir, polar_plot_out_rel_path)
                polar_big_plot_out_abs_path = os.path.join(basedir, polar_big_plot_out_rel_path)


                if not os.path.exists(polar_plot_abs_path) or not os.path.exists(polar_plot_out_abs_path): # Check one pair suffices
                    if not detections_lc_df.empty : # Use detections_lc_df for polar plot
                        ra_ps1_polar, dec_ps1_polar = analyze_ps1_photoz(kowalski_session, source_id, ra, dec, 3) # Renamed vars
                        plot_polar_coordinates(detections_lc_df, ra_ps1_polar, dec_ps1_polar, legacy_survey_data, ra, dec, polar_plot_abs_path, xlim=(-2, 2), ylim=(-2, 2), point_size=15)
                        plot_polar_coordinates(detections_lc_df, ra_ps1_polar, dec_ps1_polar, legacy_survey_data, ra, dec, polar_plot_out_abs_path, xlim=(-10, 10), ylim=(-10, 10), point_size=15)
                        plot_big_polar_coordinates(detections_lc_df, ra_ps1_polar, dec_ps1_polar, legacy_survey_data, ra, dec, polar_big_plot_abs_path, xlim=(-2, 2), ylim=(-2, 2), point_size=17)
                        plot_big_polar_coordinates(detections_lc_df, ra_ps1_polar, dec_ps1_polar, legacy_survey_data, ra, dec, polar_big_plot_out_abs_path, xlim=(-10, 10), ylim=(-10, 10), point_size=17)
                        logging.debug(f"Polar plots generated for {source_id}")
                    else:
                        logging.warning(f"Skipping polar plot generation for {source_id} - no detections for polar plot.")
            except Exception as polar_exc:
                logging.error(f"Error generating polar plots for {source_id}: {polar_exc}")
        else: # RA or Dec is None
            logging.warning(f"Skipping position-dependent data for {source_id} due to missing RA/Dec.")
            general_error_message = general_error_message or "RA/Dec not found; some plots and crossmatches are unavailable."
            # Ensure plot paths are None if not generated
            polar_plot_rel_path, polar_big_plot_rel_path = None, None
            polar_plot_out_rel_path, polar_big_plot_out_rel_path = None, None


        # Light curves (can be generated even if RA/Dec is missing, if LC data exists)
        try:
            if not light_curve_df.empty:
                light_curve_path = os.path.join(light_cur_dir, f"{source_id}_light_curve.html") # Use corrected var
                # ... (similar for other light curve paths)
                big_light_curve_path = os.path.join(light_cur_dir, f"{source_id}_big_light_curve.html")
                light_curve_zoomed_path = os.path.join(light_cur_dir, f"{source_id}_light_curve_zoomed.html")
                big_light_curve_zoomed_path = os.path.join(light_cur_dir, f"{source_id}_big_light_curve_zoomed.html")


                plot_filename_rel = os.path.join('static', 'light_curves', os.path.basename(light_curve_path))
                plot_filename_zoomed_rel = os.path.join('static', 'light_curves', os.path.basename(light_curve_zoomed_path))
                if not os.path.exists(light_curve_path) or not os.path.exists(light_curve_zoomed_path):
                    plot_light_curve(light_curve_df.copy(), source_id)
                    plot_light_curve(light_curve_df.copy(), source_id, "detections")

                plot_big_filename_rel = os.path.join('static', 'light_curves', os.path.basename(big_light_curve_path))
                plot_big_filename_zoomed_rel = os.path.join('static', 'light_curves', os.path.basename(big_light_curve_zoomed_path))
                if not os.path.exists(big_light_curve_path) or not os.path.exists(big_light_curve_zoomed_path):
                    plot_big_light_curve(light_curve_df.copy(), source_id)
                    plot_big_light_curve(light_curve_df.copy(), source_id, "detections")
                logging.debug(f"Light curve plots processed for {source_id}")
            else:
                logging.warning(f"Skipping light curve plot generation for {source_id} due to empty light_curve_df.")
                plot_filename_rel, plot_filename_zoomed_rel = None, None # Ensure None if not generated
                plot_big_filename_rel, plot_big_filename_zoomed_rel = None, None
        except Exception as lc_plot_exc:
            logging.error(f"Error generating light curve plots for {source_id}: {lc_plot_exc}")
            plot_filename_rel, plot_filename_zoomed_rel = None, None
            plot_big_filename_rel, plot_big_filename_zoomed_rel = None, None


        # ZTF cutouts (uses original alert packets)
        try:
            if original_dets_packets: # Only if we have original alerts to pick from
                ztf_cutout_filenames_or_none = filter_and_plot_alerts(kowalski_session, cutout_dir, source_id)
                ztf_cutout_basenames_for_template = ztf_cutout_filenames_or_none
            else:
                ztf_cutout_basenames_for_template = [None] * 8 # Default if no original alerts
            logging.debug(f"ZTF Cutout list: {ztf_cutout_basenames_for_template}")
        except Exception as ztf_cutout_exc:
            logging.error(f"Error with ZTF cutouts for {source_id}: {ztf_cutout_exc}")


        # SDSS crossmatch data (from original packets)
        try:
            if original_dets_packets and original_dets_packets[0].get('candidate'):
                first_candidate = original_dets_packets[0]['candidate']
                if first_candidate.get('ssdistnr') is not None and first_candidate.get('ssdistnr') > -999 and \
                   first_candidate.get('ssmagnr') is not None and first_candidate.get('ssmagnr') > -999:
                    sdss_data = {'ssdistnr': first_candidate['ssdistnr'], 'ssmagnr': first_candidate['ssmagnr']}
            logging.debug(f"SDSS Data: {sdss_data}")
        except Exception as sdss_exc:
            logging.error(f"Error with SDSS data for {source_id}: {sdss_exc}")

        # PS1 crossmatch summary (from comprehensive LC detections)
        try:
            if not detections_lc_df.empty and 'distpsnr1' in detections_lc_df.columns and 'sgscore1' in detections_lc_df.columns:
                 valid_ps1_df = detections_lc_df[
                     (detections_lc_df['distpsnr1'].notna()) & (detections_lc_df['distpsnr1'] > -999) &
                     (detections_lc_df['sgscore1'].notna()) & (detections_lc_df['sgscore1'] > -999) &
                     (detections_lc_df['distpsnr1'] <= 3)
                 ].copy()
                 if not valid_ps1_df.empty:
                     closest_ps1_row = valid_ps1_df.loc[valid_ps1_df['distpsnr1'].idxmin()]
                     ps1_dist = closest_ps1_row['distpsnr1']
                     ps1_sgs = closest_ps1_row['sgscore1']
            logging.debug(f"PS1 Crossmatch (closest): dist={ps1_dist}, sgs={ps1_sgs}")
        except Exception as ps1_xmatch_exc:
            logging.error(f"Error with PS1 xmatch for {source_id}: {ps1_xmatch_exc}")
            ps1_dist, ps1_sgs = None, None # Ensure None on error too

        # Convert "N/A" for ps1_dist and ps1_sgs to None for template compatibility
        if ps1_dist == "N/A":
            ps1_dist = None
        if ps1_sgs == "N/A":
            ps1_sgs = None
            
        # Retrieve classifications
        try:
            classifications = Classification.query.filter_by(source_id=source_id).all()
            classification_counts = defaultdict(lambda: {'count': 0, 'confidence': 0})
            for classification in classifications:
                user = User.query.get(classification.user_id)
                if user:
                    classified_by_users.append(user.username)
                    classification_counts[classification.classification]['count'] += 1
                    confidence_map = {'Uncertain': 1, 'Probable': 2, 'Confident': 3}
                    classification_counts[classification.classification]['confidence'] += confidence_map.get(classification.confidence, 0)
            if classification_counts:
                sorted_classifications = sorted(classification_counts.items(), key=lambda item: (item[1]['confidence'], item[1]['count']), reverse=True)
                most_confident_classification = sorted_classifications[0][0]
            logging.debug(f"Most Confident Classification: {most_confident_classification}")
        except Exception as class_exc:
            logging.error(f"Error retrieving classifications for {source_id}: {class_exc}")
        
        # Final data dictionary construction
        data = {
            "source_id": source_id, "ra": ra_str, "dec": dec_str, "scat_sep": scat_sep,
            "galactic_l": galactic_l, "galactic_b": galactic_b, "span": span,
            "ecliptic_lon": ecliptic_lon, "ecliptic_lat": ecliptic_lat,
            "alert_count": alert_count, "med_drb": med_drb, "min_drb": min_drb,
            "max_drb": max_drb, "avg_drb": avg_drb, "ps1_dist": ps1_dist, "ps1_sgs": ps1_sgs,
            "wise_plot": wise_filename, "plot_filename": plot_filename_rel,
            "plot_filename_zoomed": plot_filename_zoomed_rel,
            "plot_big_filename": plot_big_filename_rel,
            "plot_big_filename_zoomed": plot_big_filename_zoomed_rel,
            "ztf_cutout": ztf_cutout_basenames_for_template,
            "ps1_cutout": ps1_cutout_basename, "ls_cutout": ls_cutout_basename,
            "legacy_amount": legacy_amount, "legacy_data": legacy_data, "sdss_data": sdss_data,
            "polar_plot": polar_plot_rel_path, "polar_big_plot": polar_big_plot_rel_path,
            "polar_plot_out": polar_plot_out_rel_path,
            "polar_big_plot_out": polar_big_plot_out_rel_path,
            "classifications": classifications, "classified_by_users": classified_by_users,
            "most_confident_classification": most_confident_classification,
            "raw_alerts": raw_alerts,
            "general_error_message": general_error_message # Pass any general error
        }
        return data

    except Exception as e: # Catch-all for truly unexpected errors during the process
        logging.error(f"Outer error during fetch_transient_data for {source_id}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # Return a minimal data structure with the error message
        return {
            "source_id": source_id, "ra": "Error", "dec": "Error", "scat_sep": "N/A",
            "galactic_l": "N/A", "galactic_b": "N/A", "span": "N/A",
            "ecliptic_lon": "N/A", "ecliptic_lat": "N/A", "alert_count": 0,
            "med_drb": "N/A", "min_drb": "N/A", "max_drb": "N/A", "avg_drb": "N/A",
            "ps1_dist": "N/A", "ps1_sgs": "N/A", "wise_plot": None, "plot_filename": None,
            "plot_filename_zoomed": None, "plot_big_filename": None,
            "plot_big_filename_zoomed": None, "ztf_cutout": [None]*8,
            "ps1_cutout": None, "ls_cutout": None, "legacy_amount": 0, "legacy_data": [],
            "sdss_data": None, "polar_plot": None, "polar_big_plot": None,
            "polar_plot_out": None, "polar_big_plot_out": None,
            "classifications": [], "classified_by_users": [],
            "most_confident_classification": None, "raw_alerts": [],
            "general_error_message": f"An unexpected error occurred: {str(e)}"
        }

# ... rest of utils.py ...