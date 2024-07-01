import os
import sys
import glob
import ssl
import csv
import io
import subprocess
import argparse
import logging
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image

from astropy.io import fits as pyfits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.time import Time

from ztfquery.utils import stamps

# Naming the VLASS data file
fname = "VLASS_dyn_summary.php"
# Creating the directory for VLASS Image Plots
basedir = os.path.abspath(os.path.dirname(__file__))
save_directory = os.path.join(basedir, 'static', 'vlass_images')

def get_vlass_data():
    """
    Downloading updated VLASS Data
    """
    # Set SSL context to unverified (bypassing SSL verification)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    import urllib.request
    url = 'https://archive-new.nrao.edu/vlass/VLASS_dyn_summary.php'
    output_file = 'VLASS_dyn_summary.php'
    
    # Download the file from the URL and save it locally
    urllib.request.urlretrieve(url, output_file)
    print(f'File downloaded to: {output_file}')

def ensure_vlass_data():
    """
    Check if the VLASS data file exists locally
    """
    if not os.path.exists(os.path.join(basedir, fname)):
        get_vlass_data()

def get_tiles():
    fname = "VLASS_dyn_summary.php"
    ensure_vlass_data()
    """ Get tiles
    I ran wget https://archive-new.nrao.edu/vlass/VLASS_dyn_summary.php
    """

    # Read all lines in the file
    inputf = open(fname, "r")
    lines = inputf.readlines()
    inputf.close()

    # Extract header form the first line and clean it
    header = list(filter(None, lines[0].split("  ")))
    header = np.array([val.strip() for val in header])

    # Initialize lists to store the tile info
    names = []
    dec_min = []
    dec_max = []
    ra_min = []
    ra_max = []
    obsdate = []
    epoch = []

    # Process each line in the file 
    for line in lines[3:]:
        dat = list(filter(None, line.split("  ")))
        dat = np.array([val.strip() for val in dat])
        names.append(dat[0])  # Tile name
        dec_min.append(float(dat[1]))  # Minimum declination
        dec_max.append(float(dat[2]))  # Maximum declination
        ra_min.append(float(dat[3]))  # Minimum right ascension
        ra_max.append(float(dat[4]))  # Maximum right ascension
        obsdate.append(dat[6])  # Observation date
        epoch.append(dat[5])  # Epoch

    # Convert lists to numpy arrays
    names = np.array(names)
    dec_min = np.array(dec_min)
    dec_max = np.array(dec_max)
    ra_min = np.array(ra_min)
    ra_max = np.array(ra_max)
    obsdate = np.array(obsdate)
    epoch = np.array(epoch)

    # Return as tuple of arrays
    return (names, dec_min, dec_max, ra_min, ra_max, epoch, obsdate)


def search_tiles(tiles, c):
    """ Now that you've processed the file, search for the given RA and Dec

    Parameters
    ----------
    c: SkyCoord object
    """
    ra_h = c.ra.hour
    dec_d = c.dec.deg
    names, dec_min, dec_max, ra_min, ra_max, epochs, obsdate = tiles
    has_dec = np.logical_and(dec_d > dec_min, dec_d < dec_max)
    has_ra = np.logical_and(ra_h > ra_min, ra_h < ra_max)
    in_tile = np.logical_and(has_ra, has_dec)
    name = names[in_tile]
    epoch = epochs[in_tile]
    date = obsdate[in_tile]
    if len(name) == 0:
        print("Sorry, no tile found.")
        return None, None, None
    else:
        return name, epoch, date


def get_subtiles(tilename, epoch):
    global fname
    """ For a given tile name, get the filenames in the VLASS directory.
    Parse those filenames and return a list of subtile RA and Dec.
    RA and Dec returned as a SkyCoord object
    """
    # Adjust epoch for different versions
    if epoch == 'VLASS1.2':
        epoch = 'VLASS1.2v2'
    elif epoch == 'VLASS1.1':
        epoch = 'VLASS1.1v2'

    # URL for VLASS Directory
    url_full = 'https://archive-new.nrao.edu/vlass/quicklook/%s/%s/' % (epoch, tilename)
    print(url_full)
    urlpath = urlopen(url_full)

    string = (urlpath.read().decode('utf-8')).split("\n")   # Read and decode the content of the URL
    vals = np.array([val.strip() for val in string])    # Clean up the string values

    # Identify links and names within the html
    keep_link = np.array(["href" in val.strip() for val in string])
    keep_name = np.array([tilename in val.strip() for val in string])  

    string_keep = vals[np.logical_and(keep_link, keep_name)]    # Filter the revelant strings
    fname = np.array([val.split("\"")[7] for val in string_keep])   # Extract filenames

    # Extract RA and Dec values from filenames
    pos_raw = np.array([val.split(".")[4] for val in fname])
    if '-' in pos_raw[0]:
        # dec < 0
        ra_raw = np.array([val.split("-")[0] for val in pos_raw])
        dec_raw = np.array([val.split("-")[1] for val in pos_raw])
    else:
        # dec > 0
        ra_raw = np.array([val.split("+")[0] for val in pos_raw])
        dec_raw = np.array([val.split("+")[1] for val in pos_raw])
    ra = []
    dec = []
    # Convert RA and Dec to appropriate formats
    for ii, val in enumerate(ra_raw):
        # 24 hours is the same as hour 0
        if val[1:3] == '24':
            rah = '00'
        else:
            rah = val[1:3]
        # calculate RA in hours mins and seconds
        hms = "%sh%sm%ss" % (rah, val[3:5], val[5:])
        ra.append(hms)
        # calculate Dec in degrees arcminutes and arcseconds
        dms = "%sd%sm%ss" % (
            dec_raw[ii][0:2], dec_raw[ii][2:4], dec_raw[ii][4:])
        dec.append(dms)
    ra = np.array(ra)
    dec = np.array(dec)
    c_tiles = SkyCoord(ra, dec, frame='icrs')
    return fname, c_tiles

def get_cutout(imname, name, c, epoch):
    global save_directory  
    print("Generating cutout")
    # Position of source
    ra_deg = c.ra.deg
    dec_deg = c.dec.deg

    print("Cutout centered at position %s, %s" % (ra_deg, dec_deg))

    # Open image and establish coordinate system
    im = pyfits.open(imname, ignore_missing_simple=True)[0].data[0, 0]
    w = WCS(imname)

    # Find the source position in pixels.
    # This will be the center of our image.
    src_pix = w.wcs_world2pix([[ra_deg, dec_deg, 0, 0]], 0)
    x = src_pix[0, 0]
    y = src_pix[0, 1]

    # Check if the source is actually in the image
    pix1 = pyfits.open(imname)[0].header['CRPIX1']
    pix2 = pyfits.open(imname)[0].header['CRPIX2']
    badx = np.logical_or(x < 0, x > 2 * pix1)
    bady = np.logical_or(y < 0, y > 2 * pix2)
    if np.logical_and(badx, bady):
        print("Tile has not been imaged at the position of the source")
        return None
    else:
        # Set the dimensions of the image
        # Say we want it to be 12 arcseconds on a side,
        # to match the DES images
        image_dim_arcsec = 12
        delt1 = pyfits.open(imname)[0].header['CDELT1']
        delt2 = pyfits.open(imname)[0].header['CDELT2']
        cutout_size = image_dim_arcsec / 3600  # in degrees
        dside1 = -cutout_size / 2. / delt1
        dside2 = cutout_size / 2. / delt2

        vmin = -1e-4
        vmax = 1e-3

        im_plot_raw = im[int(y - dside1):int(y + dside1), int(x - dside2):int(x + dside2)]
        # Check if the sliced array is empty
        if im_plot_raw.size == 0:
            print("The cutout image is empty.")
            return None
        im_plot = np.ma.masked_invalid(im_plot_raw)

        # 3-sigma clipping (find root mean square of values that are not above 3 standard deviations)
        rms_temp = np.ma.std(im_plot)
        keep = np.ma.abs(im_plot) <= 3 * rms_temp
        rms = np.ma.std(im_plot[keep])

        # Find peak flux in entire image
        peak_flux = np.ma.max(im_plot.flatten())

        save_path = os.path.join(save_directory, f"{name}_{epoch}.png")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(
            np.flipud(im_plot),
            extent=[-0.5 * cutout_size * 3600., 0.5 * cutout_size * 3600.,
                    -0.5 * cutout_size * 3600., 0.5 * cutout_size * 3600],
            vmin=vmin, vmax=vmax, cmap='YlOrRd')

        peakstr = "Peak Flux %s mJy" % (np.round(peak_flux * 1e3, 3))
        rmsstr = "RMS Flux %s mJy" % (np.round(rms * 1e3, 3))

        plt.title(name + ": %s; \n %s" % (peakstr, rmsstr))
        plt.xlabel("Offset in RA (arcsec)")
        plt.ylabel("Offset in Dec (arcsec)")

        plt.savefig(save_path)
        plt.close()
        print("PNG Downloaded Successfully")

        return peak_flux, rms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_search(name, c, date=None):
    global fname
    """
    Searches the VLASS catalog for a source

    Parameters
    ----------
    names: name of the sources
    c: coordinates as SkyCoord object
    date: date in astropy Time format
    """
    print("Running for %s" % name)
    print("Coordinates %s" % c)
    print("Date: %s" % date)

    # Find the VLASS tile(s)
    tiles = get_tiles()
    tilenames, epochs, obsdates = search_tiles(tiles, c)

    past_epochs = ["VLASS1.1v2", "VLASS1.2v2", "VLASS2.1", "VLASS2.2", "VLASS3.1"]
    current_epoch = "VLASS3.2"

    if tilenames[0] is None:
        print("There is no VLASS tile at this location")

    else:
        for ii, tilename in enumerate(tilenames):
            print()
            print("Looking for tile observation for %s" % tilename)
            epoch = epochs[ii]
            obsdate = obsdates[ii]
            # Adjust name so it works with the version 2 ones for 1.1 and 1.2
            if epoch == 'VLASS1.2':
                epoch = 'VLASS1.2v2'
            elif epoch == 'VLASS1.1':
                epoch = 'VLASS1.1v2'

            if epoch not in past_epochs:
                if epoch == current_epoch:
                    # Make list of observed tiles
                    url_full = 'https://archive-new.nrao.edu/vlass/quicklook/%s/' % (epoch)
                    urlpath = urlopen(url_full)
                    # Get site HTML coding
                    string = (urlpath.read().decode('utf-8')).split("\n")
                    # clean the HTML elements of trailing and leading whitespace
                    vals = np.array([val.strip() for val in string])
                    # Make list of useful html elements
                    files = np.array(['alt="[DIR]"' in val.strip() for val in string])
                    useful = vals[files]
                    # Splice out the name from the link
                    obsname = np.array([val.split("\"")[7] for val in useful])
                    observed_current_epoch = np.char.replace(obsname, '/', '')

                    # Check if tile has been observed yet for the current epoch
                    if epoch not in observed_current_epoch:
                        print("Sorry, tile will be observed later in this epoch")
                else:
                    print("Sorry, tile will be observed in a later epoch")
            else:
                print("Tile Found:")
                print(tilename, epoch)
                subtiles, c_tiles = get_subtiles(tilename, epoch)
                # Find angular separation from the tiles to the location
                dist = c.separation(c_tiles)
                # Find tile with the smallest separation
                subtile = subtiles[np.argmin(dist)]
                url_get = "https://archive-new.nrao.edu/vlass/quicklook/%s/%s/%s" % (
                    epoch, tilename, subtile)
                imname = "%s.I.iter1.image.pbcor.tt0.subim.fits" % subtile[0:-1]
                fname = url_get + imname
                print(fname)
                if len(glob.glob(imname)) == 0:
                    cmd = "curl -O %s" % fname
                    print(cmd)
                    os.system(cmd)
                # Get image cutout and save FITS data as png
                out = get_cutout(imname, name, c, epoch)
                if out is not None:
                    peak, rms = out
                    print("Peak flux is %s uJy" % (peak * 1e6))
                    print("RMS is %s uJy" % (rms * 1e6))
                    limit = rms * 1e6
                    obsdate = Time(obsdate, format='iso').mjd
                    print("Tile observed on %s" % obsdate)
                    print(limit, obsdate)
                # Remove FITS file
                os.remove(imname)

