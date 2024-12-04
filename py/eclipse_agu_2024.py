
import sys
sys.path.append("py/")
import argparse
import datetime as dt
from dateutil import parser as dparser
import eutils as utils
import numpy as np

from analysis_plot import (
    genererate_RTI, genererate_Fan, 
    generate_fovt, generate_time_series_plots
)


# Generate all plots for 2017 Eclipses
date_eclipse_peak = dt.datetime(2017,8,21,18)
date_window = [dt.datetime(2017,8,21,16), dt.datetime(2021,12,4,14)]
# Create 2021 Eclipse Geometry
generate_fovt(
    ["cvw", "fhe", "bks"], date_eclipse_peak, beams=[], cb=False,
    central_longitude=-100.0, central_latitude=20.0,
    extent=[-150, -70, 30, 90], plt_lats = np.arange(30, 90, 10)
)

# # Generate all plots for 2021 Eclipses
# date_eclipse_peak = dt.datetime(2021,12,4,7,35)
# date_window = [dt.datetime(2021,12,4,2), dt.datetime(2021,12,4,14)]
# # Create 2021 Eclipse Geometry
# generate_fovt(["fir"], date_eclipse_peak, cb=True)