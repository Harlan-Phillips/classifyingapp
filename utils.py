# Astropy modules
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from astropy.io import fits
from astropy import units as u
from astropy.time import Time

# Data handling modules
import pandas as pd
import numpy as np

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
    df_alerts = pd.DataFrame([val['candidate'] for val in get_dets(s, name)])
    df_alerts['isalert'] = [True] * len(df_alerts)
    lc = df_alerts

    # Get 30-day history from forced photometry 
    det_prv_forced = get_prv_dets_forced(s, name) 
    if det_prv_forced is not None:  # if source is recent enough
        df_forced = pd.DataFrame(det_prv_forced)
        if 'limmag5sig' in df_forced.keys():  # otherwise no point
            df_forced['isalert'] = [False] * len(df_forced)
            
            # Merge the two dataframes
            lc = df_alerts.merge(
                df_forced, on='jd', how='outer', 
                suffixes=('_alerts', '_forced30d')).sort_values('jd').reset_index()
            
            cols_to_drop = ['index', 'rcid', 'rfid', 'sciinpseeing', 'scibckgnd',
                'scisigpix', 'magzpsci', 'magzpsciunc', 'magzpscirms', 'clrcoeff',
                'clrcounc', 'exptime', 'adpctdif1', 'adpctdif2', 'procstatus',
                'distnr', 'ranr', 'decnr', 'magnr', 'sigmagnr', 'chinr',
                'sharpnr', 'alert_mag', 'alert_ra', 'alert_dec', 'ra', 'dec',
                'forcediffimflux', 'forcediffimfluxunc', 'limmag3sig']
            cols_to_drop_existing = [col for col in cols_to_drop if col in lc.columns]
            lc = lc.drop(cols_to_drop_existing, axis=1)
            lc['fid'] = lc['fid_alerts'].combine_first(lc['fid_forced30d'])
            lc['programid'] = lc['programid_alerts'].combine_first(lc['programid_forced30d'])
            lc['field'] = lc['field_alerts'].combine_first(lc['field_forced30d'])
            lc['isalert'] = lc['isalert_alerts'].combine_first(lc['isalert_forced30d'])
            lc = lc.drop(['fid_alerts', 'fid_forced30d', 'field_alerts', 'field_forced30d',
                          'programid_alerts', 'programid_forced30d', 'isalert_alerts',
                          'isalert_forced30d'], axis=1)

            # Select magnitudes. Options: magpsf/sigmapsf (alert), mag/magerr (30d)
            lc['mag_final'] = lc['magpsf']  # alert value
            lc['emag_final'] = lc['sigmapsf']  # alert value
            if 'mag' in lc.keys():  # sometimes not there...
                lc.loc[lc['snr'] > 3, 'mag_final'] = lc.loc[lc['snr'] > 3, 'mag']  # 30d hist
                lc['emag_final'] = lc['sigmapsf']  # alert value
                lc.loc[lc['snr'] > 3, 'emag_final'] = lc.loc[lc['snr'] > 3, 'magerr']  # 30d hist
                lc = lc.drop(['magpsf', 'sigmapsf', 'magerr', 'mag'], axis=1)

            # Select limits. Sometimes limmag5sig is NaN, but if that's a nondet too, then...
            lc['maglim'] = lc['limmag5sig'] if 'limmag5sig' in lc.columns else None

            # Define whether detection or not
            lc['isdet'] = np.logical_or(lc['isalert'] == True, lc['snr'] > 3 if 'snr' in lc.columns else False)

            # Drop final things
            cols_to_drop_final = ['pid', 'diffmaglim', 'snr', 'limmag5sig', 'programid', 'field']
            cols_to_drop_final_existing = [col for col in cols_to_drop_final if col in lc.columns]
            lc = lc.drop(cols_to_drop_final_existing, axis=1)

            # Drop rows where both mag_final and maglim is NaN
            drop = np.logical_and(np.isnan(lc['mag_final']), np.isnan(lc['maglim']))
            lc = lc[~drop]
        else:
            # Still make some of the same changes
            if 'magpsf' in lc.keys():
                lc['mag_final'] = lc['magpsf']
            else:
                lc['mag_final'] = None
            
            if 'sigmapsf' in lc.keys():
                lc['emag_final'] = lc['sigmapsf']
            else:
                lc['emag_final'] = None
            lc = lc.drop(['magpsf', 'sigmapsf', 'programid'], axis=1, errors='ignore')
    else:
        # Still make some of the same changes
        if 'magpsf' in lc.keys():
            lc['mag_final'] = lc['magpsf']
        else:
            lc['mag_final'] = None
        
        if 'sigmapsf' in lc.keys():
            lc['emag_final'] = lc['sigmapsf']
        else:
            lc['emag_final'] = None
        lc = lc.drop(['magpsf', 'sigmapsf', 'programid'], axis=1, errors='ignore')

    df_prv = pd.DataFrame(get_prv_dets(s, name))
    if df_prv is not None:
        if len(df_prv) > 0:  # not always the case
            df_prv['isalert'] = [False] * len(df_prv)
            # Merge the two dataframes
            lc = lc.merge(
                df_prv, on='jd', how='outer', 
                suffixes=('_alerts', '_30d')).sort_values('jd').reset_index()
            
            cols_to_drop = ['index', 'rcid', 'rfid', 'sciinpseeing', 'scibckgnd',
                'scisigpix', 'magzpsci', 'magzpsciunc', 'magzpscirms', 'clrcoeff',
                'clrcounc', 'exptime', 'adpctdif1', 'adpctdif2', 'procstatus', 
                'distnr', 'ranr', 'decnr', 'magnr', 'sigmagnr', 'chinr',
                'sharpnr', 'alert_mag', 'alert_ra', 'alert_dec', 'ra', 'dec',
                'programpi', 'nid', 'rbversion', 'pdiffimfilename',
                'forcediffimflux', 'forcediffimfluxunc', 'limmag3sig',
                'pid', 'programid', 'candid', 'tblid', 'xpos', 'ypos', 'chipsf',
                'magap', 'sigmagap', 'sky', 'magdiff', 'fwhm', 'classtar', 'mindtoedge',
                'magfromlim', 'seeratio', 'aimage', 'bimage', 'aimagerat', 'bimagerat',
                'elong', 'nneg', 'rb', 'sumrat', 'magapbig', 'sigmagapbig', 'scorr', 'nbad']
            cols_to_drop_existing = [col for col in cols_to_drop if col in lc.columns]
            lc = lc.drop(cols_to_drop_existing, axis=1)
            lc['fid'] = lc['fid_alerts'].combine_first(lc['fid_30d'])
            lc['isalert'] = lc['isalert_alerts'].combine_first(lc['isalert_30d'])
            lc = lc.drop(['fid_alerts', 'fid_30d', 'field_alerts', 'field_30d', 'field',
                          'programid', 'isalert_alerts', 'isalert_30d', 'pid'], axis=1,
                         errors='ignore')

            # Put magpsf into mag_final
            if 'magpsf' in lc:
                lc['mag_final'] = lc['mag_final'].combine_first(lc['magpsf'])
            if 'sigmapsf' in lc:
                lc['emag_final'] = lc['emag_final'].combine_first(lc['sigmapsf'])

            # Select limits. Options: diffmaglim, limmag5sig
            if 'maglim' in lc:  # if it had a det_prv_forced
                lc['maglim'] = lc['maglim'].combine_first(lc['diffmaglim'] if 'diffmaglim' in lc.columns else None)
            else:
                lc['maglim'] = lc['diffmaglim'] if 'diffmaglim' in lc.columns else None
                if 'diffmaglim' in lc.columns:
                    lc = lc.drop(['diffmaglim'], axis=1)

    # Define whether detection or not
    try:
        lc['isdet'] = np.logical_or(lc['isalert'] == True, ~np.isnan(lc['mag_final']))
    except TypeError:
        raise ValueError("Data type issue detected while processing the light curve.")
    # If there were no prv dets at all, add a maglim column
    if 'maglim' not in lc.keys():
        lc['maglim'] = [None] * len(lc)

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
            vmax=median+u_scale_factor[ii]*sig)
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
        plt.imshow(np.asarray(img))
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
    """
    Plots the light curve of a given source and saves the plot as a PNG file.

    Parameters:
    lc (DataFrame): A DataFrame containing light curve data with columns 'fid', 'jd', 'mag_final', and 'emag_final'.
    source_id (str): The identifier of the source whose light curve is being plotted.

    Returns:
    str: The filename of the saved plot.
    """

    non_dets = lc[(lc['isdet'] == False) & (lc['maglim'] > 1)]
    
    if lc['mag_final'].isna().sum() > 0:
        print("NaN values found in 'mag_final'. Dropping NaN values.")
        lc = lc.dropna(subset=['mag_final'])
    
    if non_dets['maglim'].isna().sum() > 0:
        print("NaN values found in 'maglim'. Dropping NaN values.")
        non_dets = non_dets.dropna(subset=['maglim'])

    # Convert JD to MJD
    lc['mjd'] = Time(lc['jd'], format='jd').mjd
    # Reference to MJD 58000
    lc['mjd'] = lc['mjd'] - 58000

    # Convert JD to MJD
    non_dets['mjd'] = Time(non_dets['jd'], format='jd').mjd
    # Reference to MJD 58000
    non_dets['mjd'] = non_dets['mjd'] - 58000

    fig, ax = plt.subplots(figsize=(10/1.5 + .5, 6/1.2))
    
    # Define colors and symbols
    color_map = {'g': 'seagreen', 'r': 'crimson', 'i': 'goldenrod'}
    marker_map = {'g': 's', 'r': 'o', 'i': 's'}
    
    # Creating a dictionary to store scatter plot references
    scatter_dict = {}

    non_det_scatter_dict = {}
    non_det_elements = []
    for band in non_dets['fid'].unique():
        if band == 1:
            filter_name = 'g'
        elif band == 2:
            filter_name = 'r'
        elif band == 3:
            filter_name = 'i'
        band_data = non_dets[non_dets['fid'] == band]
        scatter = ax.scatter(band_data['mjd'], band_data['maglim'], edgecolor=color_map[filter_name], facecolor=color_map[filter_name], marker='^', label=f'Upper Limit {filter_name}-band')
        
        # Adjusting the transparency separately
        facecolors = scatter.get_facecolors()
        edgecolors = scatter.get_edgecolors()
        
        for facecolor in facecolors:
            facecolor[3] = 0.05  # Make facecolor fully transparent

        for edgecolor in edgecolors:
            edgecolor[3] = 1  # Set edgecolor to be semi-transparent

        scatter.set_facecolors(facecolors)
        scatter.set_edgecolors(edgecolors)

        # Storing scatter plot references and labels for non-detections
        labels = [f'MJD: {mjd + 58000:.5f}<br>Maglim: {maglim:.5f}<br>Filter: {filter_name}-band' for mjd, maglim in zip(band_data['mjd'], band_data['maglim'])]
        non_det_scatter_dict[scatter] = labels
        non_det_elements.append(scatter)

    # Associating ID to color
    for band in lc['fid'].unique():
        if band == 1:
            filter_name = 'g'
        elif band == 2:
            filter_name = 'r'
        elif band == 3:
            filter_name = 'i'
        
        band_data = lc[lc['fid'] == band]
        
        scatter = ax.scatter(band_data['mjd'], band_data['mag_final'],
                             color=color_map[filter_name], label=f'{filter_name}-band', marker=marker_map[filter_name])
        
        ax.errorbar(band_data['mjd'], band_data['mag_final'], yerr=band_data['emag_final'],
                             fmt='none', color=color_map[filter_name], alpha=0.5)
        
        # Storing scatter plot references and labels
        labels = [f'MJD: {mjd + 58000:.5f}<br>Mag: {mag:.5f}<br>Filter: {filter_name}-band' for mjd, mag in zip(band_data['mjd'], band_data['mag_final'])]
        scatter_dict[scatter] = labels
    
   
    # Adding lavels for non-detections when hovered over
    for scatter, labels in non_det_scatter_dict.items():
        tooltip = plugins.PointHTMLTooltip(scatter, labels=labels, css="background-color: white; color: black; font-size: 16px;")
        plugins.connect(fig, tooltip)
    
    # Adding labels for detections when hovered over
    for scatter, labels in scatter_dict.items():
        tooltip = plugins.PointHTMLTooltip(scatter, labels=labels, css="background-color: white; color: black; font-size: 16px;")
        plugins.connect(fig, tooltip)

    elements = [list(scatter_dict.keys()), non_det_elements]
    labels = ['Alerts', 'Limits']
    plugins.connect(fig, plugins.InteractiveLegendPlugin(elements, labels))

    # Safeguard against invalid values for axis limits
    detection_mags = lc['mag_final']
    if not detection_mags.empty:
        ax.set_ylim(detection_mags.max() + 0.5, detection_mags.min() - 0.5)
    
    # Set x-axis limits based on the span parameter
    if span == 'detections':
        detection_dates = lc['mjd']
        if not detection_dates.empty:
            diff = detection_dates.max() - detection_dates.min()
            if diff < 1:
                ax.set_xlim(detection_dates.min() - (diff * 1.2), detection_dates.max() + (diff * 1.2))
            else:
                ax.set_xlim(detection_dates.min() - 0.5, detection_dates.max() + 0.5)
    

    # Finalize the plot
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('MJD - 58000', fontsize=16)
    ax.set_ylabel('Magnitude', fontsize=16)
    ax.set_title(f'Light Curve for {source_id}', fontsize=18)
    #ax.legend(framealpha=1, facecolor='white')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), fancybox=True, shadow=True, ncol=3)
    ax.grid(alpha =.1)
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)

    # Convert the plot to HTML with mpld3
    html_str = mpld3.fig_to_html(fig)

    custom_css = """
    <style>
    body {font-size: 16px;}
    .mpld3-legend text { 
        font-size: 14px; 
        fill: #000000; 
    }
    .mpld3-legend rect { 
        fill: #FFFFE0; 
        stroke: #000000; 
        opacity: 0.8; 
    }
    .mpld3-tooltip {
        background-color: white;
        color: black;
        font-size: 16px;
    }
    </style>
    """
    html_str = custom_css + html_str

    # Save the HTML to a file
    plot_filename = f'static/light_curves/{source_id}_light_curve.html'
    if span == "detections":
        plot_filename = f'static/light_curves/{source_id}_light_curve_zoomed.html'

    with open(plot_filename, 'w') as f:
        f.write(html_str)

    plt.close(fig)

    return plot_filename

def plot_big_light_curve(lc, source_id, span=None):
    """
    Plots the light curve of a given source and saves the plot as a PNG file.

    Parameters:
    lc (DataFrame): A DataFrame containing light curve data with columns 'fid', 'jd', 'mag_final', and 'emag_final'.
    source_id (str): The identifier of the source whose light curve is being plotted.

    Returns:
    str: The filename of the saved plot.
    """
    
    non_dets = lc[(lc['isdet'] == False) & (lc['maglim'] > 1)]

    if lc['mag_final'].isna().sum() > 0:
        print("NaN values found in 'mag_final'. Dropping NaN values.")
        lc = lc.dropna(subset=['mag_final'])

    if non_dets['maglim'].isna().sum() > 0:
        print("NaN values found in 'maglim'. Dropping NaN values.")
        non_dets = non_dets.dropna(subset=['maglim'])
    # Convert JD to MJD
    lc['mjd'] = Time(lc['jd'], format='jd').mjd
    # Reference to MJD 58000
    lc['mjd'] = lc['mjd'] - 58000

    # Convert JD to MJD for non-detections
    non_dets['mjd'] = Time(non_dets['jd'], format='jd').mjd
    # Reference to MJD 58000
    non_dets['mjd'] = non_dets['mjd'] - 58000

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors and symbols
    color_map = {'g': 'seagreen', 'r': 'crimson', 'i': 'goldenrod'}
    marker_map = {'g': 's', 'r': 'o', 'i': 's'}
    
    # Creating a dictionary to store scatter plot references
    scatter_dict = {}

    non_det_scatter_dict = {}
    non_det_elements = []
    for band in non_dets['fid'].unique():
        if band == 1:
            filter_name = 'g'
        elif band == 2:
            filter_name = 'r'
        elif band == 3:
            filter_name = 'i'
        band_data = non_dets[non_dets['fid'] == band]
        scatter = ax.scatter(band_data['mjd'], band_data['maglim'], edgecolor=color_map[filter_name], facecolor=color_map[filter_name], marker='^', label=f'Upper Limit {filter_name}-band')
        
        # Adjusting the transparency separately
        facecolors = scatter.get_facecolors()
        edgecolors = scatter.get_edgecolors()
        
        for facecolor in facecolors:
            facecolor[3] = 0.05  # Make facecolor fully transparent

        for edgecolor in edgecolors:
            edgecolor[3] = 1  # Set edgecolor to be semi-transparent

        scatter.set_facecolors(facecolors)
        scatter.set_edgecolors(edgecolors)
        
        # Storing scatter plot references and labels for non-detections
        labels = [f'MJD: {mjd + 58000:.5f}<br>Mag: {maglim:.5f}<br>Filter: {filter_name}-band' for mjd, maglim in zip(band_data['mjd'], band_data['maglim'])]
        non_det_scatter_dict[scatter] = labels
        non_det_elements.append(scatter)

    # Associating ID to color
    for band in lc['fid'].unique():
        if band == 1:
            filter_name = 'g'
        elif band == 2:
            filter_name = 'r'
        elif band == 3:
            filter_name = 'i'
        
        band_data = lc[lc['fid'] == band]
        
        scatter = ax.scatter(band_data['mjd'], band_data['mag_final'],
                             color=color_map[filter_name], label=f'{filter_name}-band', marker=marker_map[filter_name])
        
        ax.errorbar(band_data['mjd'], band_data['mag_final'], yerr=band_data['emag_final'],
                             fmt='none', color=color_map[filter_name], alpha=0.5)
        
        # Storing scatter plot references and labels
        labels = [f'MJD: {mjd + 58000:.5f}<br>Mag: {mag:.5f}<br>Filter: {filter_name}-band' for mjd, mag in zip(band_data['mjd'], band_data['mag_final'])]
        scatter_dict[scatter] = labels

    # Adding tooltips for non-detections
    for scatter, labels in non_det_scatter_dict.items():
        tooltip = plugins.PointHTMLTooltip(scatter, labels=labels, css="background-color: white; color: black; font-size: 16px;")
        plugins.connect(fig, tooltip)

    # Adding tooltip
    for scatter, labels in scatter_dict.items():
        tooltip = plugins.PointHTMLTooltip(scatter, labels=labels, css="background-color: white; color: black; font-size: 12px;")
        plugins.connect(fig, tooltip)

    elements = [list(scatter_dict.keys()), non_det_elements]
    labels = ['Alerts', 'Limits']
    plugins.connect(fig, plugins.InteractiveLegendPlugin(elements, labels))

    # Safeguard against invalid values for axis limits
    detection_mags = lc['mag_final']
    if not detection_mags.empty:
        ax.set_ylim(detection_mags.max() + 0.5, detection_mags.min() - 0.5)

    # Set x-axis limits based on the span parameter
    if span == 'detections':
        detection_dates = lc['mjd']
        if not detection_dates.empty:
            diff = detection_dates.max() - detection_dates.min()
            if diff < 1:
                ax.set_xlim(detection_dates.min() - (diff * 1.2), detection_dates.max() + (diff * 1.2))
            else:    
                ax.set_xlim(detection_dates.min() - 0.5, detection_dates.max() + 0.5)

    # Finalize the plot
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('MJD - 58000', fontsize=20)
    ax.set_ylabel('Magnitude', fontsize=20)
    ax.set_title(f'Light Curve for {source_id}', fontsize=22)
    ax.legend(framealpha=1, facecolor='white')
    ax.grid(alpha=.1)
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)

    # Convert the plot to HTML with mpld3
    html_str = mpld3.fig_to_html(fig)

    custom_css = """
    <style>
    body {font-size: 16px;}
    .mpld3-legend text { 
        font-size: 18px; 
        fill: #000000; 
    }
    .mpld3-legend rect { 
        fill: #FFFFE0; 
        stroke: #000000; 
        opacity: 0.8; 
    }
    .mpld3-tooltip {
        background-color: white;
        color: black;
        font-size: 20px;
    }
    </style>
    """
    html_str = custom_css + html_str

    # Save the HTML to a file
    plot_filename = f'static/light_curves/{source_id}_big_light_curve.html'
    
    if span == "detections":
        plot_filename = f'static/light_curves/{source_id}_big_light_curve_zoomed.html'

    with open(plot_filename, 'w') as f:
        f.write(html_str)

    plt.close(fig)

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
        logging.debug(f"RA: {ra}, Dec: {dec}, Scatter Separation: {scat_sep}")

        # Fetch galactic coordinates
        galactic_l, galactic_b = get_galactic(ra, dec)
        logging.debug(f"Galactic Coordinates - l: {galactic_l}, b: {galactic_b}")

        # Fetch ecliptic coordinates
        ecliptic_lon, ecliptic_lat = get_ecliptic(ra, dec)
        logging.debug(f"Ecliptic Coordinates - lon: {ecliptic_lon}, lat: {ecliptic_lat}")

        # Fetch detections
        dets = get_dets(kowalski_session, source_id)
        logging.debug(f"Detections: {dets}")

        # Transform detections into alerts and count them
        alerts_raw = alert_table(dets)
        alerts_raw = alerts_raw.to_dict(orient='records') if alerts_raw is not None else []
        alert_count = len(alerts_raw)
        logging.debug(f"Alerts: {alerts_raw}, Alert Count: {alert_count}")

        # Fetch DRB (Detection Real-Bogus) values
        med_drb, min_drb, max_drb, avg_drb = get_drb(kowalski_session, source_id, dets)
        logging.debug(f"DRB - Med: {med_drb}, Min: {min_drb}, Max: {max_drb}, Avg: {avg_drb}")

        # Directories for cutouts and plots
        cutout_dir = os.path.join(basedir, 'static', 'cutouts')
        light_cur = os.path.join(basedir, 'static', 'light_curves')
        wise_dir = os.path.join(basedir, 'static', 'wise_plots')

        # WISE plot
        wise_plot_path = os.path.join(wise_dir, f"{source_id}_wise_plot.html")
        if os.path.exists(wise_plot_path):
            wise_filename = f"static/wise_plots/{source_id}_wise_plot.html"
        else:
            wise_filename = plot_wise(kowalski_session, source_id, ra, dec, wise_plot_path)

        # Light curves
        light_curve_path = os.path.join(light_cur, f"{source_id}_light_curve.html")
        big_light_curve_path = os.path.join(light_cur, f"{source_id}_big_light_curve.html")

        if os.path.exists(light_curve_path):
            plot_filename = f"static/light_curves/{source_id}_light_curve.html"
            plot_filename_zoomed = f"static/light_curves/{source_id}_light_curve_zoomed.html"
        else:
            light_curve = get_lc(kowalski_session, source_id)
            plot_filename = plot_light_curve(light_curve, source_id)
            plot_filename_zoomed = plot_light_curve(light_curve, source_id, "detections")

        if os.path.exists(big_light_curve_path):
            plot_big_filename = f"static/light_curves/{source_id}_big_light_curve.html"
            plot_big_filename_zoomed = f"static/light_curves/{source_id}_big_light_curve_zoomed.html"
        else:
            light_curve = get_lc(kowalski_session, source_id)
            plot_big_filename = plot_big_light_curve(light_curve, source_id)
            plot_big_filename_zoomed = plot_big_light_curve(light_curve, source_id, "detections")

        # ZTF cutouts
        ztf_cutout_paths = [
            os.path.join(cutout_dir, f"{source_id}_first.png"),
            os.path.join(cutout_dir, f"{source_id}_last.png"),
            os.path.join(cutout_dir, f"{source_id}_highest_snr.png"),
            os.path.join(cutout_dir, f"{source_id}_median.png"),
            os.path.join(cutout_dir, f"{source_id}_highest_drb.png"),
            os.path.join(cutout_dir, f"{source_id}_brightest_g.png"),
            os.path.join(cutout_dir, f"{source_id}_brightest_r.png"),
            os.path.join(cutout_dir, f"{source_id}_lowest_drb.png")
        ]
        ztf_cutout_basenames = [os.path.basename(path) for path in ztf_cutout_paths if os.path.exists(path)]
        logging.debug(f"ZTF Cutout Basenames: {ztf_cutout_basenames}")

        if len(ztf_cutout_basenames) != 8:
            ztf_cutout = filter_and_plot_alerts(kowalski_session, cutout_dir, source_id)
            ztf_cutout_basenames = [os.path.basename(path) for path in ztf_cutout]

        # Pan-STARRS (PS1) cutouts
        ps1_cutout_path = os.path.join(cutout_dir, f"{source_id}_ps1.png")
        if os.path.exists(ps1_cutout_path):
            ps1_cutout_basename = f"{source_id}_ps1.png"
        else:
            ps1_cutout = plot_ps1_cutout(kowalski_session, cutout_dir, source_id, ra, dec)
            ps1_cutout_basename = os.path.basename(ps1_cutout) if ps1_cutout else None

        # Legacy Survey (LS) cutouts
        ls_cutout_path = os.path.join(cutout_dir, f"{source_id}_ls.png")
        ls_cutout_basename = ''
        for attempt in range(5):  # Retry up to 5 times
            if os.path.exists(ls_cutout_path):
                ls_cutout_basename = f"{source_id}_ls.png"
                break
            else:
                ls_cutout = plot_ls_cutout(kowalski_session, cutout_dir, source_id, ra, dec)
                ls_cutout_basename = os.path.basename(ls_cutout) if ls_cutout else ''
            time.sleep(2)  # Wait for 2 seconds before retrying
        logging.debug(f"LS Cutout Basename: {ls_cutout_basename}")

        legacy_survey_data = xmatch_ls(ra, dec) # Fetch Legacy Survey data
        
        legacy_amount = legacy_survey_data.shape[0]
        if legacy_amount > 0:
            legacy_closest = legacy_survey_data.iloc[0]
            legacy_data = [
                legacy_closest.sep_arcsec.round(2),   # Separation in arcseconds
                legacy_closest.pa_degree.round(1),    # Position angle in degrees
                legacy_closest.z_phot_median.round(2),# Photometric redshift median
                legacy_closest.z_phot_l68.round(2),   # 68% lower bound on photometric redshift
                legacy_closest.z_phot_u68.round(2),   # 68% upper bound on photometric redshift
                legacy_closest.type 
            ]
        else:
            legacy_data = []
        
        logging.debug(f"Legacy Survey Data: {legacy_survey_data}")

        sdss_data = None
        if dets and dets[0]['candidate']['ssdistnr'] != -999.0 and dets[0]['candidate']['ssmagnr'] != -999.0:
            sdss_data = {
                'ssdistnr': dets[0]['candidate']['ssdistnr'],
                'ssmagnr': dets[0]['candidate']['ssmagnr']
            }
        logging.debug(f"SDSS Data: {sdss_data}")

        # Aggregate Pan-STARRS data and remove duplicates
        pan_starrs_data = []
        seen_sgscore1 = set()
        for det in dets:
            candidate = det['candidate']
            # Filter based on your conditions
            if 'distpsnr1' in candidate and candidate['distpsnr1'] != -999.0 and \
            'sgscore1' in candidate and candidate['sgscore1'] != -999.0 and \
            candidate['distpsnr1'] <= 3:
                if candidate['sgscore1'] not in seen_sgscore1:
                    pan_starrs_data.append({
                        'distpsnr1': candidate['distpsnr1'],
                        'sgscore1': candidate['sgscore1']
                    })
                    seen_sgscore1.add(candidate['sgscore1'])

        # Create DataFrame only from filtered data
        pan_starrs_df = pd.DataFrame(pan_starrs_data)

        if not pan_starrs_df.empty:
            # Find the closest match
            closest_ps1 = pan_starrs_df.loc[pan_starrs_df['distpsnr1'].idxmin()]
            ps1_dist = closest_ps1['distpsnr1']
            ps1_sgs = closest_ps1['sgscore1']
        else:
            ps1_dist = None
            ps1_sgs = None

        # Create the polar plot
        ztf_alerts = pd.DataFrame([det['candidate'] for det in dets])
        polar_plot_path = os.path.join('static', 'light_curves', f'{source_id}_polar_plot.html')
        polar_big_plot_path = os.path.join('static', 'light_curves', f'{source_id}_big_polar_plot.html')
        polar_plot_path_out = os.path.join('static', 'light_curves', f'{source_id}_polar_plot_out.html')
        polar_big_plot_path_out = os.path.join('static', 'light_curves', f'{source_id}_big_polar_plot_out.html')
        
        if not os.path.exists(polar_plot_path) or not os.path.exists(polar_plot_path_out):
            ra_ps1, dec_ps1 = analyze_ps1_photoz(kowalski_session, source_id, ra, dec, 3)
            plot_polar_coordinates(ztf_alerts, ra_ps1, dec_ps1, legacy_survey_data, ra, dec, polar_plot_path, xlim=(-2, 2), ylim=(-2, 2), point_size=15)
            plot_polar_coordinates(ztf_alerts, ra_ps1, dec_ps1, legacy_survey_data, ra, dec, polar_plot_path_out, xlim=(-10, 10), ylim=(-10, 10), point_size=15)
            plot_big_polar_coordinates(ztf_alerts, ra_ps1, dec_ps1, legacy_survey_data, ra, dec, polar_big_plot_path, xlim=(-2, 2), ylim=(-2, 2), point_size=17)
            plot_big_polar_coordinates(ztf_alerts, ra_ps1, dec_ps1, legacy_survey_data, ra, dec, polar_big_plot_path_out, xlim=(-10, 10), ylim=(-10, 10), point_size=17)
        # Retrieve classifications and determine the most confident classification
        classifications = Classification.query.filter_by(source_id=source_id).all()
        classification_counts = defaultdict(lambda: {'count': 0, 'confidence': 0})
        classified_by_users = []

        for classification in classifications:
            classification_counts[classification.classification]['count'] += 1
            classified_by_users.append(User.query.get(classification.user_id).username)
            if classification.confidence == 'Not confident':
                classification_counts[classification.classification]['confidence'] += 1
            elif classification.confidence == 'Confident':
                classification_counts[classification.classification]['confidence'] += 2
            elif classification.confidence == 'Certain':
                classification_counts[classification.classification]['confidence'] += 3

        if classification_counts:
            most_confident_classification = max(
                classification_counts.items(),
                key=lambda x: (x[1]['confidence'], x[1]['count'])
            )[0]
        else:
            most_confident_classification = None

        logging.debug(f"Most Confident Classification: {most_confident_classification}")

        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        ra_str = coord.ra.to_string(unit=u.hour, sep=':', precision=4)
        dec_str = coord.dec.to_string(unit=u.degree, sep=':', precision=4)

        logging.debug(f"RA (string): {ra_str}, Dec (string): {dec_str}")
        # Return the core data needed
        data = {
            "source_id": source_id,
            "ra": ra_str,
            "dec": dec_str,
            "scat_sep": scat_sep,
            "galactic_l": galactic_l,
            "galactic_b": galactic_b,
            "span": get_span(kowalski_session, source_id, dets),
            "ecliptic_lon": ecliptic_lon,
            "ecliptic_lat": ecliptic_lat,
            "dets": dets,
            "alert_count": alert_count,  # Match key 'alert_count'
            "med_drb": med_drb,  # Match key 'med_drb'
            "min_drb": min_drb,
            "max_drb": max_drb,
            "avg_drb": avg_drb,
            "ps1_dist": ps1_dist,
            "ps1_sgs": ps1_sgs,
            "wise_plot": wise_filename,  # Match key 'wise_plot'
            "plot_filename": plot_filename,  # Match key 'plot_filename'
            "plot_filename_zoomed": plot_filename_zoomed,  # Match key 'plot_filename_zoomed'
            "plot_big_filename": plot_big_filename,  # Match key 'plot_big_filename'
            "plot_big_filename_zoomed": plot_big_filename_zoomed,  # Match key 'plot_big_filename_zoomed'
            "ztf_cutout": ztf_cutout_basenames,  # Match key 'ztf_cutout'
            "ps1_cutout": ps1_cutout_basename,  # Match key 'ps1_cutout'
            "ls_cutout": ls_cutout_basename,  # Match key 'ls_cutout'
            "legacy_amount": legacy_amount,
            "legacy_data": legacy_data,
            "sdss_data": sdss_data,  # Match key 'sdss_data'  # Match key 'pan_starrs_df'
            "polar_plot": polar_plot_path,  # Match key 'polar_plot'
            "polar_big_plot": polar_big_plot_path,  # Match key 'polar_big_plot'
            "polar_plot_out": polar_plot_path_out,  # Match key 'polar_plot_out'
            "polar_big_plot_out": polar_big_plot_path_out,  # Match key 'polar_big_plot_out'
            "classifications": classifications,  # Match key 'classifications'
            "classified_by_users": classified_by_users,  # Match key 'classified_by_users'
            "most_confident_classification": most_confident_classification,  # Match key 'most_confident_classification'
            "ra_str": ra_str,  # Match key 'ra_str'
            "dec_str": dec_str  # Match key 'dec_str'
        }

        return data

    except Exception as e:
        logging.error(f"Error while fetching transient data: {str(e)}")
        return None