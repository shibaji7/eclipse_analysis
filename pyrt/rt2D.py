#!/usr/bin/env python3

"""rt2D.py: simulate python program for RT"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import copy
import datetime as dt
import glob
import os

import numpy as np
import pandas as pd
import pydarn
from dateutil import parser as dparser
from geopy.distance import great_circle as GC
from loguru import logger
from scipy.io import savemat
from density_models import WAMIPE, IRI2D, WACCMX
from rays import Rays2D
from scipy.io import loadmat
import plots
import utils

class RayTrace2D(object):
    """
    Ray trace class to trace all the points
    """

    def __init__(
        self,
        event,
        rad,
        beam,
        cfg,
    ):
        self.beam = beam
        self.event = event
        self.rad = rad
        self.folder = utils.get_folder(rad, beam, event)
        os.makedirs(self.folder, exist_ok=True)
        self.cfg = cfg
        self.hdw = pydarn.read_hdw_file(self.rad)
        self.fig_name = self.folder  + "/{time}.png".format(
            time=self.event.strftime("%H%M")
        )
        self.edensity_file = self.folder + "/{dn}.mat".format(
            dn=self.event.strftime("%H.%M")
        )
        self.sim_fname = self.folder + "/{date}_rt.mat".format(
            date=self.event.strftime("%H%M")
        )
        self._estimate_bearing_()
        self._eclipse()
        return

    def _eclipse(self):
        self.p = np.zeros((
            len(self.bearing_object["lat"]), 
            len(self.bearing_object["ht"])
        ))*np.nan
        h = np.mean(self.bearing_object["ht"])
        e = utils.Eclipse()
        for i, lat, lon in zip(
            range(len(self.bearing_object["lat"])),
            self.bearing_object["lat"],
            self.bearing_object["lon"]
        ):
            self.p[i, :] = e.create_eclipse_shadow(self.event, lat, lon, h)
        print(np.nanmin(self.p), np.nanmax(self.p))
        return

    def _estimate_bearing_(self):
        """Estimate laitude and logitude bearings"""
        fname = self.folder + f"/bearing.mat"
        bearing = self.hdw.boresight.physical - ((self.beam - self.hdw.beams/2) * self.hdw.beam_separation)
        logger.info(f"Bearing angle of beam {self.beam} is {bearing} deg")
        lat, lon = (self.hdw.geographic.lat, self.hdw.geographic.lon)
        p = (lat, lon)
        gc = GC(p, p)
        dist = np.linspace(
            0, self.cfg.max_ground_range_km, self.cfg.number_of_ground_step_km
        )

        m = {}
        lats, lons = [], []
        for d in dist:
            x = gc.destination(p, bearing, distance=d)
            lats.append(x[0])
            lons.append(x[1])
        m["dist"], m["lat"], m["lon"] = dist, np.array(lats), np.array(lons)
        (
            m["olat"],
            m["olon"],
            m["rb"],
            m["num_range"],
            m["max_range"],
            m["range_inc"],
        ) = (
            lat,
            lon,
            bearing,
            float(len(dist)),
            float(self.cfg.max_ground_range_km),
            float(dist[1] - dist[0]),
        )
        m["ht"] = np.arange(
            self.cfg.start_height_km,
            self.cfg.end_height_km,
            self.cfg.height_incriment_km,
        ).astype(float)
        m["start_height"], m["height_inc"], m["num_heights"], m["heights"] = (
            float(self.cfg.start_height_km),
            float(self.cfg.height_incriment_km),
            float(
                len(
                    np.arange(
                        self.cfg.start_height_km,
                        self.cfg.end_height_km,
                        self.cfg.height_incriment_km,
                    )
                )
            ),
            np.arange(
                self.cfg.start_height_km,
                self.cfg.end_height_km,
                self.cfg.height_incriment_km,
            ),
        )

        m["freq"], m["tol"], m["nhops"] = (
            float(self.cfg.frequency),
            float(1e-7),
            float(self.cfg.nhops),
        )
        m["elev_s"], m["elev_i"], m["elev_e"] = (
            float(self.cfg.start_elevation),
            float(self.cfg.elevation_inctiment),
            float(self.cfg.end_elevation),
        )
        m["radius_earth"] = self.cfg.radius_earth
        savemat(fname, m)
        self.bearing_object = copy.copy(m)
        return

    def read_density_rays(self, fname):
        self.density = loadmat(fname)["ne"]
        self.sim_fname = self.folder + "{date}.{bm}_rt.mat".format(
            bm="%02d" % self.beam, date=self.event.strftime("%H%M")
        )
        logger.info("Data-Model comparison: reading rays....")
        self.rays = Rays2D.read_rays(self.event, self.rad, self.beam, self.cfg, self.folder, self.sim_fname)
        return        

    def compile(self, density):
        """Compute RT using Pharlap"""
        self.density = density
        pwd = os.getcwd() + "/pyrt/pharlap/pharlap_4.5.3/dat"
        cmd = "export DIR_MODELS_REF_DAT={pwd};\
                cd pyrt/pharlap/;\
                matlab -softwareopengl -nodisplay -nodesktop -nosplash -nojvm -r \"UT=[{ut}];rad='{rad}';dic='{dic}';fname='{fname}';bm={bm};\
                rt_2D;exit;\"".format(
            pwd=pwd,
            ut=self.event.strftime("%Y %m %d %H %M"),
            rad=self.rad,
            dic=self.folder,
            bm=self.beam,
            fname=self.sim_fname,
        )
        logger.info(f"Running command: {cmd}")
        os.system(cmd)
        logger.info("Data-Model comparison: reading rays....")
        self.rays = Rays2D.read_rays(
            self.event, self.rad, self.beam, self.cfg, self.folder, self.sim_fname
        )
        return
    

def execute_2DRT_WAM_simulations(args):
    cfg = utils.read_params_2D()
    rtobj = RayTrace2D(args.event, args.rad, args.beam, cfg)
    wam = WAMIPE(cfg)
    #if not os.path.exists(rtobj.edensity_file):
    wam.fetch_dataset_by_locations(
        args.event, 
        rtobj.bearing_object["lat"],
        rtobj.bearing_object["lon"],
        rtobj.bearing_object["ht"], 
        dlat=0.2, dlon=0.2,
        to_file=rtobj.edensity_file
    )
    rtobj.compile(wam.param_val)
    utils.clean()
    plots.plot_rays(
        rtobj.folder,
        rtobj.fig_name,
        rtobj,
        fr"WAM-IPE + {args.rad.upper()}/{str(args.beam)}, $f_0$={str(cfg.frequency)} MHz",
        maxground = cfg.max_ground_range_km+10,
    )
    return


def execute_2DRT_IRI_simulations(args):
    cfg = utils.read_params_2D()
    start_mask_time = args.event + dt.timedelta(
        seconds=cfg.iri_param.start_mask_time*cfg.iri_param.step_sec
    )
    logger.info(f"start_mask_time>> {start_mask_time}")
    for i in np.arange(cfg.iri_param.run_it):
        d = args.event + dt.timedelta(seconds=int(i)*cfg.iri_param.step_sec)
        rtobj = RayTrace2D(d, args.rad, args.beam, cfg)
        if not os.path.exists(rtobj.sim_fname):
            iri = IRI2D(
                d, 
                rtobj.bearing_object["lat"],
                rtobj.bearing_object["lon"],
                rtobj.bearing_object["ht"],  
                to_file=rtobj.edensity_file, 
                cfg=cfg, 
                eclipse_prop=dict(
                    start_mask_time = start_mask_time
                ),
                iri_version=20,
            )
            rtobj.compile(iri.param_val)
            utils.clean()
            plots.plot_rays(
                rtobj.folder,
                rtobj.fig_name,
                rtobj,
                fr"IRI20 + {args.rad.upper()}/{str(args.beam)}, $f_0$={str(cfg.frequency)} MHz",
                maxground = cfg.max_ground_range_km+10,
            )
    return


def execute_2DRT_WACCMX_simulations(args):
    cfg = utils.read_params_2D()
    for d in range(30):
        event = args.event + dt.timedelta(minutes=d*5)
        rtobj = RayTrace2D(event, args.rad, args.beam, cfg)
        if not os.path.exists(rtobj.fig_name):
            waccm = WACCMX(cfg)
            waccm.fetch_dataset_by_locations(
                event, 
                rtobj.bearing_object["lat"],
                rtobj.bearing_object["lon"],
                rtobj.bearing_object["ht"], 
                dlat=0.2, dlon=0.2,
                to_file=rtobj.edensity_file
            )
            rtobj.compile(waccm.param_val)
            utils.clean()
            plots.plot_rays(
                rtobj.folder,
                rtobj.fig_name,
                rtobj,
                fr"WACCM-X + {args.rad.upper()}/{str(args.beam)}, $f_0$={str(cfg.frequency)} MHz",
                maxground = cfg.max_ground_range_km+10,
            )
    plots.create_movie(
        rtobj.folder+"/", 
        "{rad}_{bm}".format(rad=args.rad, bm="%02d" % args.beam),
        ".png"
    )
    return