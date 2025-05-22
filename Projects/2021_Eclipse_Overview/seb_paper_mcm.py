import sys
sys.path.append("py/")
import argparse
import datetime as dt
from dateutil import parser as dparser
import eutils as utils
import numpy as np

from loguru import logger
from plot import RangeTimePlot

from read_fitacf import Radar


from generate_plots import (
    setup,
    generate_fov_overview,
    generate_conjugate_fov_overview,
    create_rti_plots,
    create_fan_plots,
    create_ISR_plots,
    create_RTP_pydarn_plots,
)

methods = ["fan_plot", "plot_fov", "plot_rti"]

if "plot_fov" in methods:
    ## Create 2021 Eclipse Geometry on southerin hemisphere
    generate_fov_overview(
        ["fir", "mcm"], dt.datetime(2021,12,4,7,30), cb=True
    )

if "plot_rti" in methods:
    rad_beams = [
        dict(rad="mcm", beam=0, channel=None, tfreq=None),
        dict(rad="mcm", beam=7, channel=None, tfreq=None),
        dict(rad="mcm", beam=15, channel=None, tfreq=None)
    ]
    yscale = "srange" 
    range = [0,3000]
    dates = [dt.datetime(2021,12,4,6), dt.datetime(2021,12,4,10)]
    
    
    ftitle =  fr"Rad: mcm / "+\
            dates[1].strftime("%d %b, %Y") if dates[0].day == dates[1].day else dates[0].strftime("%d-") + dates[1].strftime("%d %b, %Y")
    rti = RangeTimePlot(
        range, 
        dates, 
        ftitle, 
        3,
        font="sans-sarif",
    )
    tags = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]
    for j, rad_beam in enumerate(rad_beams):
        rad, beam = rad_beam["rad"], rad_beam["beam"]
        title = fr"{tags[j]} Beam: {beam}"
        radar = Radar(rad, dates, type="fitacf")
        radar.calculate_ground_range()
        df = radar.df.copy()
        logger.info(f"Reading radar: {rad} / Beam: {beam} / Unique: {df.tfreq.unique()}, {df.channel.unique()}, {df.bmnum.unique()}")
        if rad_beam["channel"]:
            df = df[df.channel==rad_beam["channel"]]
        df["unique_tfreq"] = df.tfreq.apply(lambda x: int(x/0.5)*0.5)
        if rad_beam["tfreq"]: 
            df = df[df.unique_tfreq==rad_beam["tfreq"]]
        ax = rti.addParamPlot(
            rad, df, 
            beam, title=title,
            p_max=500, p_min=0,
            xlabel="Time, UT" if j==2 else "", ylabel="Slant Range, km", 
            zparam="v", label=r"Velocity, $ms^{-1}$",
            cmap="Spectral", cbar=j==0, add_gflg=False,
            yparam=yscale, kind="scatter"
        )
        rti.overlay_eclipse_shadow(rad, beam, dates, ax, j==2, dx=0.05)
        # rti.add_conjugate_eclipse(rad, beam, dates, ax)
        ax.set_ylim(range)
        ax.set_xlim(dates)
    rti.save(f"figures_2021/rti.{rad}-{beam}.png")
    rti.close()


if "fan_plot" in methods:
    rads = ["mcm"]
    channel = None
    tfreq = None
    dates = [dt.datetime(2021,12,4,6),dt.datetime(2021,12,4,6,30),dt.datetime(2021,12,4,7), dt.datetime(2021,12,4,7,30), dt.datetime(2021,12,4,8),
             dt.datetime(2021,12,4,8, 30), dt.datetime(2021,12,4,9), dt.datetime(2021,12,4,9, 30),
             dt.datetime(2021,12,4,10)]
    create_fan_plots(rads, dates, tfreq=tfreq, channel=channel)