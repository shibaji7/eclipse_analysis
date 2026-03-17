import sys
sys.path.extend([
    "py/", 
])

from loguru import logger
import datetime as dt
from read_fitacf import Radar
from plot import RangeTimePlot
import numpy as np

def create_rti_plots(
    rad_beams_ch_freq, dates, yscale="srange", 
    range=[0, 4500]
):
    date = dates[1].strftime("%d %b, %Y") if dates[0].day == dates[1].day else dates[0].strftime("%d-") + dates[1].strftime("%d %b, %Y")
    rti = RangeTimePlot(
        range, 
        dates, 
        date, 
        len(rad_beams_ch_freq),
        font="sans-sarif",
    )
    for j, rad_beam in enumerate(rad_beams_ch_freq):
        rad, beam, channel, tfreq, params = (
            rad_beam[0], rad_beam[1], rad_beam[2], rad_beam[3], rad_beam[4]
        )
        title = params["title"]
        p_max, p_min = params["p_max"], params["p_min"]
        cmap, cbar = params["cmap"], params["cbar"]
        radar = Radar(rad, dates, type="fitacf")
        radar.calculate_ground_range()
        df = radar.df.copy()
        logger.info(f"Reading radar: {rad} / Beam: {beam} / Unique: {df.tfreq.unique()}, {df.channel.unique()}, {df.bmnum.unique()}")
        if channel:
            df = df[df.channel==channel]
        df["unique_tfreq"] = df.tfreq.apply(lambda x: int(x/0.5)*0.5)
        v, tf = np.array(df.v), np.array(df.unique_tfreq)
        # v[tf==10.5] *= -1
        # df.v = v
        if tfreq: 
            df = df[df.unique_tfreq==tfreq]
        ax = rti.addParamPlot(
            rad, df, 
            beam, title=title,
            p_max=p_max, p_min=p_min,
            xlabel="Time, UT" if j==len(rad_beams_ch_freq)-1 else "", 
            ylabel="Slant Range, km", 
            zparam="v", label=r"Velocity, $ms^{-1}$",
            cmap=cmap, cbar=cbar, add_gflg=False,
            yparam="srange", kind="scatter"
        )
        rti.overlay_eclipse_shadow(
            rad, beam, dates, ax, j==-1, 0.0
        )
        ax.set_ylim(range)
        ax.set_xlim(dates)
    rti.save(f"figures_2021/rti.Seb.png")
    rti.close()
    return


def create_RTI_plots():
    rad_beams = [
        (
            "fir", 7, 1, 12.,
            dict(
                title = fr"(A) Rad: fir / Beam: 7 / ch: 1 / $f_0$: 12.0 MHz",
                cmap="jet_r", p_max=50, p_min=-100, cbar=True
            )
        ),
        (
            "mcm", 7, None, None,
            dict(
                title = fr"(B) Rad: mcm / Beam: 7 / ch: all / $f_0$: 10.5 MHz",
                cmap="GnBu", p_max=600, p_min=100, cbar=True
            )
        ), 
    ]
    range = [0,4500]
    dates = [dt.datetime(2021,12,4,6), dt.datetime(2021,12,4,10)]
    create_rti_plots(rad_beams, dates, range=range)
    return

create_RTI_plots()