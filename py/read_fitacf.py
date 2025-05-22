"""
    This python module is used to read the dataset from fitacf/fitacf3 
    level dataset.
"""

#import mplstyle

import os
import pandas as pd
import pydarn 
import glob
import bz2
from loguru import logger
import datetime as dt
import numpy as np
from scipy import constants as C
import eutils as utils
from tqdm import tqdm
tqdm.pandas()

def gsMapSlantRange(slant_range, altitude=None, elevation=None):
    """Calculate the ground scatter mapped slant range.
    See Bristow et al. [1994] for more details. (Needs full reference)
    Parameters
    ----------
    slant_range
        normal slant range [km]
    altitude : Optional[float]
        altitude [km] (defaults to 300 km)
    elevation : Optional[float]
        elevation angle [degree]
    Returns
    -------
    gsSlantRange
        ground scatter mapped slant range [km] (typically slightly less than
        0.5 * slant_range.  Will return -1 if
        (slant_range**2 / 4. - altitude**2) >= 0. This occurs when the scatter
        is too close and this model breaks down.
    """
    Re = 6731.1

    # Make sure you have altitude, because these 2 projection models rely on it
    if not elevation and not altitude:
        # Set default altitude to 300 km
        altitude = 300.0
    elif elevation and not altitude:
        # If you have elevation but not altitude, then you calculate altitude,
        # and elevation will be adjusted anyway
        altitude = np.sqrt(Re ** 2 + slant_range ** 2 + 2. * slant_range * Re *
                           np.sin(np.radians(elevation))) - Re
    if (slant_range**2) / 4. - altitude ** 2 >= 0:
        gsSlantRange = Re * \
            np.arcsin(np.sqrt(slant_range ** 2 / 4. - altitude ** 2) / Re)
        # From Bristow et al. [1994]
    else:
        gsSlantRange = -1
    return gsSlantRange

class Radar(object):

    def __init__(self, rad, dates=None, clean=False, type="fitacf",):
        logger.info(f"Initialize radar: {rad}/{dates[0]}")
        self.rad = rad
        self.dates = dates
        self.clean = clean
        self.type = type
        self.__setup__()
        self.__fetch_data__()
        d = dates[0].replace(hour=0,minute=0,second=0)
        if d in [
            dt.datetime(2024,4,8), 
            dt.datetime(2017,8,21),
            dt.datetime(2021,12,4),
        ]:
            logger.info("Creating shadows!")
            # self.create_eclipse_shadow()
        self.calculate_decay_rate()
        return

    def compute_recalculations(self, vheight_model="standard_model", gsmap_model=""):
        logger.info("Calculate v-Height...")
        
        return

    def calculate_vheight(self, n_hop):
        if self.rad == "fir": self.df.frang = 0.
        self.df["srange"] = self.df.frang + (self.df.rsep * self.df.slist)
        logger.info("Calculate v-Height...")
        from calc_vheight import Thomas, chisham
        # self.df["Thomas_vheight"] = self.df.apply(
        #     lambda row: Thomas(row["srange"], row["gflg"]), axis=1
        # )
        self.df["Thomas_vheight"] = self.df.srange.apply(
            lambda x: Thomas(x, 1)
        )
        self.df["Chisham_vheight"] = self.df.srange.apply(
            lambda x: chisham(x)
        )
        self.df["Chisham_gsmap"] = self.df.apply(
            lambda row: gsMapSlantRange(row["srange"], row["Chisham_vheight"]), 
            axis = 1
        )
        self.df["Thomas_gsmap"] = self.df.apply(
            lambda row: gsMapSlantRange(row["srange"], row["Thomas_vheight"]), 
            axis = 1
        )
        return

    def calculate_ground_range(self, n_hop=1):
        self.calculate_vheight(n_hop)
        logger.info("Calculate 1-hop ground range to ionospheric point...")
        Re = 6731.0 # Radius of the Earth
        a, b = (
            Re**2+(Re+self.df.Thomas_vheight)**2-(self.df.srange/(2*n_hop))**2,
            2*Re*(Re+self.df.Thomas_vheight)
        )
        self.df["Gn"] = n_hop*Re*np.arccos(a/b)
        return
    
    def __setup__(self):
        logger.info(f"Setup radar: {self.rad}")
        date = self.dates[0].strftime("%Y%m%d")
        self.files = glob.glob(f"/sd-data/{self.dates[0].year}/{self.type}/{self.rad}/{date}.*")
        self.files = glob.glob(f"/home/chakras4/Research/Backup_CodeBase/2021_Datasets/{date}.*{self.rad}.*")
        # self.files = glob.glob(f"database/{self.type}/*{self.rad}.*")
        self.files.sort()
        self.hdw = pydarn.read_hdw_file(self.rad)
        self.fov = pydarn.Coords.GEOGRAPHIC(self.hdw.stid)
        logger.info(f"Files: {len(self.files)}")
        return

    def get_lat_lon_along_beam(self, beam):
        lats, lons = self.fov[0], self.fov[1]
        return lats[:,beam], lons[:,beam]

    def __fetch_data__(self):
        self.fname = f"database/{self.rad}.{self.type}.{self.dates[0].strftime('%Y%m%d')}.csv"
        # self.fname = f"database/{self.rad}.{self.type}.csv"
        logger.info(f"load files {self.fname}")
        if self.clean: os.remove(self.fname)
        if os.path.exists(self.fname):
            self.df = pd.read_csv(self.fname, parse_dates=["time"])
        else:
            records = []
            for f in self.files:
                logger.info(f"Reading file: {f}")
                with bz2.open(f) as fp:
                    reader = pydarn.SuperDARNRead(fp.read(), True)
                    records += reader.read_fitacf()
            if len(records)>0:
                self.__tocsv__(records)
        print(f"Dataframe: {self.df.shape}")
        if "lat" not in self.df.columns:
            self.df.tfreq = np.round(np.array(self.df.tfreq)/1e3, 1)
            # self.update_location_details()
        self.check_the_sounding_mode()
        return

    def __tocsv__(self, records):
        time, v, slist, p_l, frang, scan, beam,\
            w_l, gflg, elv, phi0, tfreq, rsep,\
            skynoise, nrang, cha = (
            [], [], [],
            [], [], [],
            [], [], [],
            [], [], [],
            [], [], [], []
        )
        for r in records:
            if "v" in r.keys():
                t = dt.datetime(
                    r["time.yr"], 
                    r["time.mo"],
                    r["time.dy"],
                    r["time.hr"],
                    r["time.mt"],
                    r["time.sc"],
                    r["time.us"],
                )
                time.extend([t]*len(r["v"]))
                cha.extend([r["channel"]]*len(r["v"]))
                tfreq.extend([r["tfreq"]]*len(r["v"]))
                rsep.extend([r["rsep"]]*len(r["v"]))
                frang.extend([r["frang"]]*len(r["v"]))
                nrang.extend([r["nrang"]]*len(r["v"]))
                skynoise.extend([r["noise.sky"]]*len(r["v"]))
                scan.extend([r["scan"]]*len(r["v"]))
                beam.extend([r["bmnum"]]*len(r["v"]))
                v.extend(r["v"])
                gflg.extend(r["gflg"])
                slist.extend(r["slist"])
                p_l.extend(r["p_l"])
                w_l.extend(r["w_l"])
                if "elv" in r.keys(): elv.extend(r["elv"])
                if "phi0" in r.keys(): phi0.extend(r["phi0"])                
            
        self.df = pd.DataFrame()
        self.df["v"] = v
        self.df["gflg"] = gflg
        self.df["slist"] = slist
        self.df["bmnum"] = beam
        self.df["p_l"] = p_l
        self.df["w_l"] = w_l
        if len(elv) > 0: self.df["elv"] = elv
        if len(phi0) > 0: self.df["phi0"] = phi0
        self.df["time"] = time
        self.df["tfreq"] = tfreq
        self.df["scan"] = scan
        self.df["rsep"] = rsep
        self.df["frang"] = frang
        self.df["nrang"] = nrang
        self.df["noise.sky"] = skynoise
        self.df["channel"] = cha

        if self.dates:
            self.df = self.df[
                (self.df.time>=self.dates[0]) & 
                (self.df.time<=self.dates[1])
            ]
        self.calculate_ground_range()
        self.df.to_csv(self.fname, index=False, header=True)
        return

    def to_csv(
        self, 
        names=[
            "time", "bmnum", "gflg", "noise.sky",
            "nrang", "p_l", "scan", "slist", "tfreq",
            "v", "w_l",
        ],
        params={
            "time": "Datetime of the observation (U: datetime)",
            "bmnum": "Beam number (U: 1)",
            "gflg": "Ground scatter flag (U: 1)",
            "noise.sky": "Sky noise (U: 1)",
            "nrang": "Number of range gates (U: 1)",
            "p_l": "Power (U: dB)",
            "scan": "Scan flag (U: 1)",
            "slist": "Range gate (U: 1)",
            "tfreq": "Operating frequency (U: kHz)",
            "v": "LoS Doppler Velocity (U: m/s)",
            "w_l": "Spectral width (U: m/s)",
        }
    ):
        o = self.df[names].copy()
        txt = "=======================================\n Parameter Description:"
        for n in names:
            txt += f" {n} - {params[n]}\n"
        txt += "=======================================\n"
        datetext = self.dates[0].strftime("%Y%m%d")
        fname = f"database/{self.rad}.{self.type}.{datetext}.csv"
        o.to_csv(fname, index=False, header=True)
        with open(fname, "r+") as fp:
            content = fp.read()
            fp.seek(0, 0)
            fp.write(txt.rstrip("\r\n") + "\n" + content)
        return

    def update_location_details(self):
        logger.info(f"Inside Location details!")
        self.df = self.df.progress_apply(self.__latlon__, axis=1)
        self.df.to_csv(self.fname, index=False, header=True)
        return

    def create_eclipse_shadow(self):
        logger.info(f"Create shadow!")
        lats, lons = (
            self.fov[0].T, self.fov[1].T
        )
        folder = f"database/eclipse_path/{self.dates[0].strftime('%Y-%m-%d')}/"
        os.makedirs(folder, exist_ok=True)
        for bm in self.df.bmnum.unique():
            file = f"{folder}/oc.{self.rad}.{bm}.csv"
            if not os.path.exists(file):
                o = self.df[self.df.bmnum==bm]
                dates = o.time
                p = utils.get_rti_eclipse(dates, lats[bm,:], lons[bm,:])
                frame = pd.DataFrame()
                frame["dates"] = dates
                for g in range(p.shape[1]):
                    frame[f"gate_{g}"] = p[:, g]
                frame.to_csv(file, header=True, index=False)
        return

    def __latlon__(self, row):
        lat, lon = self.fov[0].T, self.fov[1].T
        row["lat"], row["lon"] = (
            lat[row.bmnum, row.slist],
            lon[row.bmnum, row.slist]
        )
        p = utils.get_eclipse(row.time, [300], [row.lat], [row.lon])[0]
        row["occul"] = p[0,0,0,0]
        return row

    def recalculate_elv_angle(self, XOff=0, YOff=100, ZOff=0):
        return

    def check_the_sounding_mode(self):
        frequency, scan_time, beams = set(), set(), set()
        txt = ""
        for bm in self.df.bmnum.unique():
            beams.add(bm)
            o = self.df[(self.df.bmnum==bm)].groupby(by="time").mean().reset_index()
            x = np.rint(
                    (o.time.iloc[1] - o.time.iloc[0]).total_seconds()
                )
            scan_time.add(x)
            tf = set()
            for t in o.tfreq:
                frequency.add(np.rint(t))
                tf.add(str(np.rint(t)))
            txt += f"Beam: {bm}, t={x}, f={','.join(list(tf))}\n"
        return

    def calculate_decay_rate(self):
        logger.info(f"Calculate Decay")
        f, w_l = np.array(self.df.tfreq)*1e6, np.array(self.df.w_l)
        k = 2*np.pi*f/C.c
        self.df["tau_l"] = 1.e3/(k*w_l)
        return

if __name__ == "__main__":
    # dates = [dt.datetime(2024,4,8), dt.datetime(2024,4,9)]
    # Radar("bks", dates, type="fitacf")
    # Radar("fhe", dates, type="fitacf")
    # Radar("fhw", dates, type="fitacf")
    # Radar("kap", dates, type="fitacf")
    # Radar("gbr", dates, type="fitacf")
    # dates = [dt.datetime(2017,8,21), dt.datetime(2017,8,22)]
    # Radar("bks", dates, type="fitacf")
    # Radar("fhe", dates, type="fitacf")
    # Radar("fhw", dates, type="fitacf")
    # Radar("cve", dates, type="fitacf")
    # Radar("cvw", dates, type="fitacf")
    # dates = [dt.datetime(2023,10,14), dt.datetime(2023,10,15)]
    # Radar("fhe", dates, type="fitacf")
    # Radar("fhw", dates, type="fitacf")
    # Radar("cve", dates, type="fitacf")
    # Radar("cvw", dates, type="fitacf")
    dates = [dt.datetime(2021,12,4), dt.datetime(2021,12,5)]
    Radar("fir", dates, type="fitacf")
    Radar("mcm", dates, type="fitacf")
    # dates = [dt.datetime(2017,5,27), dt.datetime(2017,5,28)]
    # Radar("bks", dates, type="fitacf")
    # Radar("fhe", dates, type="fitacf")
    # Radar("fhw", dates, type="fitacf")
    pass