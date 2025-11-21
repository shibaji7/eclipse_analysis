

import sys
sys.path.extend([
    "py/", 
    "Projects/2021_Eclipse_Special/", 
    "Projects/2021_Eclipse_Special/ASHLEY_v1/ASHLEY-A/",
    "Projects/2021_Eclipse_Special/ASHLEY_v1/ASHLEY-E/",
])
import argparse
import datetime as dt
from dateutil import parser as dparser
import eutils as utils
import numpy as np

from read_fitacf import Radar

from plotutils import (
    setup,
    generate_fov_overview,
    generate_conjugate_fov_overview,
    create_fan_plots,
    create_mix_ts, create_rays,
    create_fan_plots_stack,
    create_map_plots,
    plot_hall_conductivity,
    create_Digisonde_plots,
    create_map_plot_cpcp_sd,
    create_MCM_plots,
)


methods = [
    # "plot_fov",
    # "plot_maps", 
    "plot_hall_conductivity",
    # "fan_plot_fir", 
    # "others"
]
setup()

if "others" in methods:
    create_MCM_plots()
    create_mix_ts()
    create_Digisonde_plots()
    create_rays()
    create_rays(run_name="Dec2021_gitm_base_Modeled")

if "plot_fov" in methods:
    ## Create 2021 Eclipse Geometry on southerin hemisphere
    generate_fov_overview(
        ["fir", "mcm"], dt.datetime(2021,12,4,7,30), cb=True,
    )
    generate_conjugate_fov_overview(
        ["fir", "mcm"], ["sas"], dt.datetime(2021,12,4,7,30), cb=True,
        hemi="south"
    )

if "fan_plot_fir" in methods:
    rads = ["fir", "mcm"]
    channel = [None, None]
    tfreq = [[12.0, 12.2], None]
    dates = [

        dt.datetime(2021,12,4,6),
             dt.datetime(2021,12,4,6,30),
             dt.datetime(2021,12,4,7), 
             dt.datetime(2021,12,4,7,30),
             #dt.datetime(2021,12,4,7,35),
             dt.datetime(2021,12,4,7,40),
             #dt.datetime(2021,12,4,7,45),
             dt.datetime(2021,12,4,7,50),
             dt.datetime(2021,12,4,7,55),
            #  dt.datetime(2021,12,4,8),
              dt.datetime(2021,12,4,8, 5), #dt.datetime(2021,12,4,7,40),
            #  dt.datetime(2021,12,4,8,30),
            dt.datetime(2021,12,4,8, 10),
            dt.datetime(2021,12,4,8, 15),
            dt.datetime(2021,12,4,8, 30),
            dt.datetime(2021,12,4,9,30),
             ]
    create_fan_plots_stack(
        rads, dates, tfreq=tfreq, channel=channel,
        central_longitude=100, central_latitude=-60.0,
        #extent=[-40, -120, -90, -50], 
        extent=[-180, 180, -90, -50], 
        plt_lats = np.arange(-90, -49, 10), 
        p_min=[-50, 100], p_max=[50, 600], mark_lon=-50,
    )

if "fan_plot_mcm" in methods:
    rads = ["mcm"]
    channel = None
    tfreq = None
    dates = [
        dt.datetime(2021,12,4,7), 
             dt.datetime(2021,12,4,7,30),
             dt.datetime(2021,12,4,7,35),
             dt.datetime(2021,12,4,7,40),
            #  dt.datetime(2021,12,4,7,45),
             dt.datetime(2021,12,4,7,50),
             dt.datetime(2021,12,4,7,55),
            #  dt.datetime(2021,12,4,8,2), #dt.datetime(2021,12,4,7,40),
             dt.datetime(2021,12,4,8,30),
             dt.datetime(2021,12,4,9), dt.datetime(2021,12,4,9,30),
    ]
    create_fan_plots(rads, dates, tfreq=tfreq, channel=channel)

if "plot_maps" in methods:
    create_map_plots()

if "plot_hall_conductivity" in methods:
    plot_hall_conductivity(cond="Hall")
    plot_hall_conductivity()
    create_map_plot_cpcp_sd()

# if "plot_rti_analysis" in methods:
#     calculate_decay_rate


