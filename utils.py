# Astropy modules
from astropy.coordinates import SkyCoord
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

def logon():
    """ Log onto Kowalski """
    username = "user"
    password = "pass"
    s = Kowalski(
        protocol='https', host='kowalski.caltech.edu', port=443,
            verbose=False, username=username, password=password)
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
                "candidate.sgscore1": 1
            }
        }
    }
    query_result = s.query(query=q)
    try:
        out = query_result['default']['data']
        return out
    except:
        return []
   
def get_pos(s,name):
    """ Calculate the median position from alerts, and the scatter """
    det_alerts = get_dets(s, name)
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

def plot_triplet(triplet):
    """
    Plot the triplet images (science, template, difference) with enhanced settings.
    """
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1))
    titles = ['Science', 'Reference', 'Difference']

    # Normalize the images for better contrast
    for ax, img, title in zip(axes, triplet.transpose((2, 0, 1)), titles):
        # Normalize the image for better display quality
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        ax.imshow(img, cmap='gray', origin='lower')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def get_brightest_dets(s, name):
    q0 = {
        "query_type": "find_one",
        "query": {
            "catalog": "ZTF_alerts",
            "filter": {"objectId": name}
        }
    }
    out = s.query(q0)
    
    # Check if the query result contains the expected structure
    if "default" in out and "data" in out["default"]:
        alert = out["default"]["data"]
        return alert
    else:
        # Handle the case where the expected keys are not found
        print(f"Error: Expected data not found in the query result for object {name}")
        return None

def plot_ztf_cutout(s,ddir,name):
    """ Plot the ZTF cutouts: science, reference, difference """
    fname = "%s/%s_triplet.png" %(ddir,name)
    print(fname)
    if os.path.isfile(fname)==False:
        q0 = {
                "query_type": "find_one",
                "query": {
                            "catalog": "ZTF_alerts",
                            "filter": {"objectId": name}
                        }
            }
        out = s.query(q0)
        alert = out["default"]["data"]
        tr = make_triplet(alert)
        plot_triplet(tr)
        plt.tight_layout()
        plt.savefig(fname, bbox_inches = "tight")
        plt.close()
    return fname


def get_brightest_triplet(ddir, detections, name):
    fname = "%s/%s_triplet_brightest.png" %(ddir,name)
    """ Get the alert with the brightest magnitude (lowest magpsf) """
    
    
    
    # Select a random detection from the filtered list
    brightest_alert = detections
    
    tr = make_triplet(brightest_alert)
    plot_triplet(tr)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches = "tight")
    plt.close()
    return fname


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

def plot_light_curve(lc, source_id):
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
    
    if lc['maglim'].isna().sum() > 0:
        print("NaN values found in 'maglim'. Dropping NaN values.")
        lc = lc.dropna(subset=['maglim'])

    # Convert JD to MJD
    lc['mjd'] = Time(lc['jd'], format='jd').mjd
    # Reference to MJD 58000
    lc['mjd'] = lc['mjd'] - 58000

    # Convert JD to MJD
    non_dets['mjd'] = Time(non_dets['jd'], format='jd').mjd
    # Reference to MJD 58000
    non_dets['mjd'] = non_dets['mjd'] - 58000

    fig, ax = plt.subplots(figsize=(10/1.5 + .5, 6/1.5))
    
    # Define colors and symbols
    color_map = {'g': 'seagreen', 'r': 'crimson', 'i': 'goldenrod'}
    marker_map = {'g': 's', 'r': 'o', 'i': 's'}
    
    # Creating a dictionary to store scatter plot references
    scatter_dict = {}

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
        labels = [f'MJD: {mjd + 58000}<br>Mag: {mag}' for mjd, mag in zip(band_data['mjd'], band_data['mag_final'])]
        scatter_dict[scatter] = labels
    
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
        scatter = ax.scatter(band_data['mjd'], band_data['maglim'], color=color_map[filter_name], marker='^', label=f'Upper Limit {filter_name}-band')
        
        
        # Storing scatter plot references and labels for non-detections
        labels = [f'MJD: {mjd + 58000}<br>Maglim: {maglim}' for mjd, maglim in zip(band_data['mjd'], band_data['maglim'])]
        non_det_scatter_dict[scatter] = labels
        non_det_elements.append(scatter)
  
            

   
    # Adding tooltips for non-detections
    for scatter, labels in non_det_scatter_dict.items():
        tooltip = plugins.PointHTMLTooltip(scatter, labels=labels, css="background-color: white; color: black; font-size: 16px;")
        plugins.connect(fig, tooltip)
    # Adding tooltip
    for scatter, labels in scatter_dict.items():
        tooltip = plugins.PointHTMLTooltip(scatter, labels=labels, css="background-color: white; color: black; font-size: 16px;")
        plugins.connect(fig, tooltip)

    elements = [list(scatter_dict.keys()), non_det_elements]
    labels = ['Alerts', 'Limits']
    plugins.connect(fig, plugins.InteractiveLegendPlugin(elements, labels))

    # Finalize the plot
    ax.invert_yaxis()
    ax.set_xlabel('MJD - 58000', fontsize=16)
    ax.set_ylabel('Magnitude', fontsize=16)
    ax.set_title(f'Light Curve for {source_id}', fontsize=18)
    ax.legend(framealpha=1, facecolor='white')
    ax.grid(True)
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
    with open(plot_filename, 'w') as f:
        f.write(html_str)

    plt.close(fig)

    return plot_filename

def plot_big_light_curve(lc, source_id):
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

    if lc['maglim'].isna().sum() > 0:
        print("NaN values found in 'maglim'. Dropping NaN values.")
        lc = lc.dropna(subset=['maglim'])
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
        labels = [f'MJD: {mjd + 58000}<br>Mag: {mag}' for mjd, mag in zip(band_data['mjd'], band_data['mag_final'])]
        scatter_dict[scatter] = labels
    
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
        scatter = ax.scatter(band_data['mjd'], band_data['maglim'], color=color_map[filter_name], marker='^', label=f'Upper Limit {filter_name}-band')
        
        
        # Storing scatter plot references and labels for non-detections
        labels = [f'MJD: {mjd + 58000}<br>Maglim: {maglim}' for mjd, maglim in zip(band_data['mjd'], band_data['maglim'])]
        non_det_scatter_dict[scatter] = labels
        non_det_elements.append(scatter)
    
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

    # Finalize the plot
    ax.invert_yaxis()
    ax.set_xlabel('MJD - 58000', fontsize=20)
    ax.set_ylabel('Magnitude', fontsize=20)
    ax.set_title(f'Light Curve for {source_id}', fontsize=22)
    ax.legend(framealpha=1, facecolor='white')
    ax.grid(True)
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)

    # Convert the plot to HTML with mpld3
    html_str = mpld3.fig_to_html(fig)

    custom_css = """
    <style>
    body {font-size: 20px;}
    .mpld3-tooltip {font-size: 18px;}
    </style>
    """
    html_str = custom_css + html_str

    # Save the HTML to a file
    plot_filename = f'static/light_curves/{source_id}_big_light_curve.html'
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
    fig, ax = plt.subplots(figsize=(3.7, 3.7))
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
        labels = [f"<div>RA: {ra:.6f}, Dec: {dec:.6f}</div>" for ra, dec in zip(legacy_survey_data['ra'], legacy_survey_data['dec'])]
        plugins.connect(fig, plugins.PointHTMLTooltip(legacy_scatter, labels=labels))

    # Plot Legacy Survey data if available
    if ra_ps1 != -1 and dec_ps1 != -1:
        # Create a SkyCoord object for the PS1 source
        ps1_coord = SkyCoord(ra=ra_ps1, dec=dec_ps1, unit='deg')
        ps1_ra_offset = (ps1_coord.ra - central_coord.ra).arcsec
        ps1_dec_offset = (ps1_coord.dec - central_coord.dec).arcsec
        # Plot PS1 source
        ps1_scatter = ax.scatter(ps1_ra_offset, ps1_dec_offset, color='purple', marker='*', s=point_size*10, label='PS1')
        plugins.connect(fig, plugins.PointHTMLTooltip(ps1_scatter, labels=[f'PS1 Source<br>RA: {ra_ps1:.6f}<br>Dec: {dec_ps1:.6f}']))
            
        
    # Central source
    central_scatter = ax.scatter(0, 0, color='black', marker='o', s=point_size, label='Transient')  # Reduce marker size to 100
    plugins.connect(fig, plugins.PointHTMLTooltip(central_scatter, labels=['Transient']))

    # Add concentric circles
    circle1 = plt.Circle((0, 0), 3, color='blue', fill=False, alpha=0.1)  # Increase radius to 3 arcseconds
    circle2 = plt.Circle((0, 0), 1.5, color='blue', fill=False, alpha=0.3)  # Increase radius to 1.5 arcseconds
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    ax.set_xlabel('RA (arcsec)', fontsize=16)
    ax.set_ylabel(r'Dec (arcsec)', fontsize=16)
    ax.legend(title='Filter/Catalog')
    ax.set_title('Coordinates of Nearby Sources', fontsize=18)
    ax.legend(loc='upper right')
    ax.grid(True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()

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
    fig, ax = plt.subplots(figsize=(5, 5))
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
        labels = [f"<div>RA: {ra:.6f}, Dec: {dec:.6f}</div>" for ra, dec in zip(legacy_survey_data['ra'], legacy_survey_data['dec'])]
        plugins.connect(fig, plugins.PointHTMLTooltip(legacy_scatter, labels=labels))

    # Plot Legacy Survey data if available
    if ra_ps1 != -1 and dec_ps1 != -1:
        # Create a SkyCoord object for the PS1 source
        ps1_coord = SkyCoord(ra=ra_ps1, dec=dec_ps1, unit='deg')
        ps1_ra_offset = (ps1_coord.ra - central_coord.ra).arcsec
        ps1_dec_offset = (ps1_coord.dec - central_coord.dec).arcsec
        # Plot PS1 source
        ps1_scatter = ax.scatter(ps1_ra_offset, ps1_dec_offset, color='purple', marker='*', s=point_size*10, label='PS1')
        plugins.connect(fig, plugins.PointHTMLTooltip(ps1_scatter, labels=[f'PS1 Source<br>RA: {ra_ps1:.6f}<br>Dec: {dec_ps1:.6f}']))
            
        
    # Central source
    central_scatter = ax.scatter(0, 0, color='black', marker='o', s=point_size, label='Transient')  # Reduce marker size to 100
    plugins.connect(fig, plugins.PointHTMLTooltip(central_scatter, labels=['Transient']))

    # Add concentric circles
    circle1 = plt.Circle((0, 0), 3, color='blue', fill=False, alpha=0.1)  # Increase radius to 3 arcseconds
    circle2 = plt.Circle((0, 0), 1.5, color='blue', fill=False, alpha=0.3)  # Increase radius to 1.5 arcseconds
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    ax.set_xlabel('RA (arcsec)', fontsize=18)
    ax.set_ylabel(r'Dec (arcsec)', fontsize=18)
    ax.legend(title='Filter/Catalog')
    ax.set_title('Coordinates of Nearby Sources', fontsize=20)
    ax.legend(loc='upper right')
    ax.grid(True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()

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
    wsid = "user"
    pwd = "pass"
    jobs = mastcasjobs.MastCasJobs(
            userid=wsid, password=pwd, context="HLSP_PS1_STRM")
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
    
    return -1, -1