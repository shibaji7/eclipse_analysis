"""
    This python module is used to create all the plots
"""

import sys
sys.path.append("py/")
import argparse
import datetime as dt
from dateutil import parser as dparser
import eutils as utils
import numpy as np

from read_fitacf import Radar

from generate_plots import (
    setup,
    generate_fov_overview,
    generate_conjugate_fov_overview,
    create_rti_plots
)

methods = ["plot_rti"]
setup()

if "plot_fov" in methods:
    ## Create 2021 Eclipse Geometry on southerin hemisphere
    generate_fov_overview(
        ["fir"], dt.datetime(2021,12,4,7,30), cb=True,
    )
    generate_conjugate_fov_overview(
        ["fir"], ["sas", "bks", "pgr"], dt.datetime(2021,12,4,7,30), cb=True,
        hemi="south"
    )
    generate_conjugate_fov_overview(
        ["sas", "bks", "pgr"], ["fir"], dt.datetime(2021,12,4,7,30), cb=False,
        central_longitude=-120.0, central_latitude=50.0,
        extent=[-150, -60, 30, 90], plt_lats = np.arange(30, 90, 10),
        overlay_eclipse_other_hemi=True, hemi="north", 
        other_instruments=[
            ("ott", "mag", 45.4030, -75.5520), 
            ("brd", "mag", 49.8700, -99.9739),
            ("fcc", "mag", 58.7590, -94.0880),
            ("mhisr", "isr", 42.6, -(360-288.5)),
        ]
    )

if "download" in methods:
    # Load and save all the dates for FIR, SAS, BKS, WAL, KAP, GBR radars
    start_date, rads = (
        dt.datetime(2021,12,2), 
        ["fir", "sas", "bks", "pgr"]
    )
    for d in range(5):
        for rad in rads:
            Radar(
                rad, 
                [
                    start_date + dt.timedelta(d), 
                    start_date + dt.timedelta(d) + dt.timedelta(1)
                ], 
                type="fitacf"
            )

    from supermag import SuperMAG
    sm = SuperMAG.FetchSM(
        "database/", 
        [start_date, start_date+dt.timedelta(5)], 
        uid="shibaji7", 
        stations=["fcc", "brd", "ott"]
    )

if "plot_rti" in methods:
    rad_beams = [("sas", 4)]
    yscale = "srange" 
    range = [0,4500]
    channel = None
    tfreq = None
    dates = [dt.datetime(2021,12,4,6), dt.datetime(2021,12,4,10)]
    create_rti_plots(rad_beams, dates, range=range, tfreq=tfreq, channel=channel)
