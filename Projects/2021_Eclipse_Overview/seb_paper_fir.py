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

methods = [
#    "plot_rti", 
    "fan_plot",
]


def plotr(rad_beams, fname, dates, range, yscale):
    ftitle =  fr"Rad: fir / "+\
            dates[1].strftime("%d %b, %Y") if dates[0].day == dates[1].day else dates[0].strftime("%d-") + dates[1].strftime("%d %b, %Y")
    rti = RangeTimePlot(
        range, 
        dates, 
        ftitle, 
        4,
        font="sans-sarif",
    )
    tags = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]
    for j, rad_beam in enumerate(rad_beams):
        rad, beam = rad_beam["rad"], rad_beam["beam"]
        title = fr"{tags[j]} Beam: {beam} / Ch: {rad_beam['channel']} / $f_0$= {rad_beam['tfreq'] if rad_beam['tfreq'] else 'all'}"
        radar = Radar(rad, dates, type="fitacf")
        radar.calculate_ground_range()
        df = radar.df.copy()
        df.srange /= 2
        logger.info(f"Reading radar: {rad} / Beam: {beam} / Unique: {df.tfreq.unique()}, {df.channel.unique()}, {df.bmnum.unique()}")
        if rad_beam["channel"]:
            df = df[df.channel==rad_beam["channel"]]
        df["unique_tfreq"] = df.tfreq.apply(lambda x: int(x/0.5)*0.5)
        v, tf = np.array(df.v), np.array(df.unique_tfreq)
        v[tf==10.5] *= -1
        df.v = v
        if rad_beam["tfreq"]: 
            df = df[df.unique_tfreq==rad_beam["tfreq"]]
        ax = rti.addParamPlot(
            rad, df, 
            beam, title=title,
            p_max=30, p_min=-30,
            xlabel="Time, UT" if j==3 else "", ylabel=r"$\frac{1}{2}$-Slant Range, km", 
            zparam="v", label=r"Velocity, $ms^{-1}$",
            cmap="Spectral", cbar=j==0, add_gflg=False,
            yparam=yscale, kind="scatter"
        )
        rti.overlay_eclipse_shadow(rad, beam, dates, ax, j==2, dx=0.05)
        # rti.add_conjugate_eclipse(rad, beam, dates, ax)
        ax.set_ylim(range)
        ax.set_xlim(dates)
    rti.save(fname)
    rti.close()

if "plot_rti" in methods:
    rad_beams = [
        dict(rad="fir", beam=3, channel=2, tfreq=None),
        dict(rad="fir", beam=7, channel=2, tfreq=None),
        dict(rad="fir", beam=11, channel=2, tfreq=None),
        dict(rad="fir", beam=15, channel=2, tfreq=None)
    ]
    yscale = "srange" 
    range = [0,2500]
    dates = [dt.datetime(2021,12,4,6), dt.datetime(2021,12,4,10)]
    plotr(rad_beams, f"figures_2021/rti.fir-15-2-all.png", dates, range, yscale)

    rad_beams = [
        dict(rad="fir", beam=7, channel=2, tfreq=10.5),
        dict(rad="fir", beam=7, channel=2, tfreq=12),
        dict(rad="fir", beam=7, channel=2, tfreq=13.5),
        dict(rad="fir", beam=7, channel=2, tfreq=15.5)
    ]
    plotr(rad_beams, f"figures_2021/rti.fir-7-2-individual.png", dates, range, yscale)
    


if "fan_plot" in methods:
    rads = ["fir"]
    channel = 1
    tfreq = None
    dates = [dt.datetime(2021,12,4,6),dt.datetime(2021,12,4,6,30),dt.datetime(2021,12,4,7), dt.datetime(2021,12,4,7,30), dt.datetime(2021,12,4,8),
             dt.datetime(2021,12,4,8, 30), dt.datetime(2021,12,4,9), dt.datetime(2021,12,4,9, 30),
             dt.datetime(2021,12,4,10)]
    create_fan_plots(
        rads, dates, tfreq=tfreq, channel=channel,
        central_longitude=-90.0, central_latitude=-60.0,
    extent=[40, 120, -90, -45], plt_lats = np.arange(-90, -45, 10), 
    p_min=-30, p_max=30, mark_lon=60
    )