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
    # Create 2021 Eclipse Geometry on southerin hemisphere
    generate_fov_overview(
        ["fir"], dt.datetime(2021,12,4,7,30), cb=True,
    )
    generate_conjugate_fov_overview(
        ["fir"], ["sas", "bks", "pgr"], dt.datetime(2021,12,4,7,30), cb=True,
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

if "plot_rti" in methods:
    rad = "fir"
    dates = [dt.datetime(2021,12,4,6), dt.datetime(2021,12,4,10)]
    create_rti_plots(rad, dates)
