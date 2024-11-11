#!/usr/bin/env python3

"""simulate.py: simulate python program for RT"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import argparse
import datetime as dt
from loguru import logger
from dateutil import parser as dparser
from pathlib import Path
import glob
import os
from rt2D import (
    execute_2DRT_WAM_simulations, 
    execute_2DRT_IRI_simulations,
    execute_2DRT_WACCMX_simulations
)
from plots import gerenate_fov_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="waccm", help="Model name [wam/gitm/waccm/iri]"
    )
    parser.add_argument(
        "-md", "--method", default="rt", help="Method rt/rti/movie"
    )
    parser.add_argument("-r", "--rad", default="fhw", help="Radar code (default cvw)")
    parser.add_argument("-rs", "--rads", default="cvw,cve,fhe,fhw,bks,gbr", help="Radar code (default cvw)")
    parser.add_argument(
        "-bm", "--beam", default=7, type=int, help="Radar beam (default 7)"
    )
    parser.add_argument(
        "-ev",
        "--event",
        default=dt.datetime(2017, 8, 21, 16),
        help="Event date for simulation [YYYY-mm-ddTHH:MM]",
        type=dparser.isoparse,
    )
    args = parser.parse_args()
    args.rads = args.rads.split(",")
    logger.info("\n Parameter list for simulation ")
    for k in vars(args).keys():
        print("     ", k, "->", str(vars(args)[k]))
    #gerenate_fov_plot(args.rads, args.beam, args.event)
    if args.method == "rt":
        if args.model == "wam":
            execute_2DRT_WAM_simulations(args)
        if args.model == "waccm":
            execute_2DRT_WACCMX_simulations(args)
        if args.model == "iri":
            execute_2DRT_IRI_simulations(args)
    if args.method == "movie":
        import sys
        sys.path.append("py/")
        import eutils
        eutils.create_movie(
            f"figures/rt/{args.event.strftime('%Y-%m-%d')}/{args.rad}/{'%02d'%args.beam}/", 
            "movie.avi", "*.png", fps=3
        )