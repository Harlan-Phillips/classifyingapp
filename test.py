import gzip
import io
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Astropy modules
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from astropy.time import Time

# Data handling modules
import pandas as pd
import numpy as np

# Compression and file handling modules
from collections import Counter, defaultdict
import random
import time

# Image handling modules
from PIL import Image
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

def logon():
    """ Log onto Kowalski """
    username = "htp24"
    password = "98]2(c,xLd53"
    s = Kowalski(
        protocol='https', host='kowalski.caltech.edu', port=443,
        verbose=False, username=username, password=password)
    return s

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
        print(f"Plotting {title} with min={np.min(img)}, max={np.max(img)}")  # Debugging print
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        ax.imshow(img, cmap='gray', origin='lower')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    return fig  # Return the figure to allow saving

def plot_ztf_cutout(s, ddir, name):
    """ Plot the ZTF cutouts: science, reference, difference """
    fnames = []
    need_query = False

    for i in range(3):
        fname = "%s/%s_triplet%d.png" % (ddir, name, i + 1)
        if not os.path.isfile(fname):
            need_query = True
        fnames.append(fname)

    if need_query:
        q0 = {
            "query_type": "find",
            "query": {
                "catalog": "ZTF_alerts",
                "filter": {"objectId": name}
            },
            "kwargs": {
                "limit": 3,
            }
        }
        out = s.query(q0)
        
        for i, alert in enumerate(out["default"]["data"]):
            fname = "%s/%s_triplet%d.png" % (ddir, name, i + 1)
            if not os.path.isfile(fname):
                print(f"Processing {name} - Creating {fname}")
                tr = make_triplet(alert)
                fig = plot_triplet(tr)
                fig.savefig(fname, bbox_inches="tight")
                plt.close(fig)

    return fnames

def main():
    kowalski_session = logon()
    cutout_dir = 'testpic'
    source_id = 'ZTF20aajihcg'
    ztf_cutout = plot_ztf_cutout(kowalski_session, cutout_dir, source_id)
    print(f"Saved cutouts: {ztf_cutout}")

if __name__ == '__main__':
    main()