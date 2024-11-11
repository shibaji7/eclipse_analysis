#!/usr/bin/env python3

"""gitm.py: fetch GITM data"""

__author__ = "Mrak, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import glob
import os
import pandas as pd

import numpy as np
import utils
from loguru import logger
from scipy import io
from scipy.io import savemat
import xarray as xr
import netCDF4 as nc

class WAMIPE(object):
    def __init__(
        self, cfg, 
        folder="database/simulation/WAM-IPE/", 
        param="electron_density"
    ):
        self.cfg = cfg
        self.folder = folder
        self.param = param
        self.load_dataset()
        return

    def load_dataset(self):
        """
        Load all dataset available
        """
        files = np.array(sorted(glob.glob(f"{self.folder}*.nc")))
        self.file_dates = np.array(
            [
                dt.datetime.strptime(
                    os.path.split(i)[-1].replace(".nc", "").replace("ipe.", ""), 
                    "%Y%m%d_%H%M%S"
                )
                for i in files
            ]
        )
        self.dataset = {}
        for i, f in enumerate(files):
            ds = xr.open_dataset(f)
            logger.info(f"Loading file: {f}")
            o = dict(
                lon = ((ds["lon"].values + 180) % 360) - 180, 
                lat = ds["lat"].values,
                alt = ds["alt"].values,
            )
            o[self.param] = ds[self.param].values
            self.dataset[self.file_dates[i]] = o
        return

    def fetch_dataset_by_locations(
        self, time, lats, lons, alts, 
        dlat=0.2, dlon=0.2, to_file=None
    ):
        """
        Fetch data by lat/lon limits
        """
        logger.info(f"Time: {time}")
        ds = self.dataset[time]
        out, ix = np.zeros((len(alts), len(lons))) * np.nan, 0
        for lat, lon in zip(lats, lons):
            i_lon, i_lat = (
                np.argmin(np.abs(lon-ds["lon"])),
                np.argmin(np.abs(lat-ds["lat"]))
            )
            alt, d = (
                ds["alt"],
                ds[self.param][:, i_lat, i_lat]
            )
            method = "intp" if np.max(alt) > max(alts) else "extp"
            out[:, ix] = utils.interpolate_by_altitude(
                np.array(alt), alts, np.array(d),
                self.cfg.scale, self.cfg.kind, method=method
            ) * 1e-6
            ix += 1
        self.param_val = out
        if to_file:
            mobj = {}
            mobj["ne"] = out
            savemat(to_file, mobj)
        return out


class GITM(object):
    def __init__(self, cfg, year=2017, folder="dataset/GITM/20170821/", param="eden"):
        self.cfg = cfg
        self.year = year
        self.folder = folder
        self.param = param
        self.load_dataset()
        return

    def load_dataset(self):
        """
        Load all dataset available
        """
        files = np.array(sorted(glob.glob(f"{self.folder}*.sav")))
        self.file_dates = np.array(
            [
                dt.datetime.strptime(os.path.split(i)[1][7:-4], "%y%m%d_%H%M%S")
                for i in files
            ]
        )
        self.dataset = {}
        for i, f in enumerate(files):
            self.file_dates[i] = self.file_dates[i].replace(year=self.year)
            D = io.readsav(f)
            self.dataset[self.file_dates[i]] = D
            logger.info(f"Loading file: {f}")
        return

    def fetch_dataset(self, time, latlim, lonlim):
        """
        Fetch data by lat/lon limits
        """
        i = np.argmin([np.abs((t - time).total_seconds()) for t in self.file_dates])
        D = self.dataset[self.file_dates[i]]
        glat = (D["lat1"] * 180 / np.pi)[:, 0]
        glon = (D["lon1"][0, :] * 180) / np.pi
        idx = np.logical_and(glon >= lonlim[0], glon <= lonlim[1])
        idy = np.logical_and(glat >= latlim[0], glat <= latlim[1])
        galt = np.squeeze(D[self.param]) / 1e3
        out = D[self.param][:, idx, :][:, :, idy]
        return out

    def fetch_dataset_by_locations(self, time, lats, lons, alts):
        """
        Fetch data by lat/lon limits
        """
        n = len(lats)
        i = np.argmin([np.abs((t - time).total_seconds()) for t in self.file_dates])
        logger.info(f"Time: {self.file_dates[i]}")
        D = self.dataset[self.file_dates[i]]
        glat = (D["lat1"] * 180 / np.pi)[:, 0]
        glon = (D["lon1"][0, :] * 180) / np.pi
        galt = D["alt1"][:, 0, 0] / 1e3
        out, ix = np.zeros((len(alts), n)) * np.nan, 0
        for lat, lon in zip(lats, lons):
            lon = np.mod(360 + lon, 360)
            idx = np.argmin(np.abs(glon - lon))
            idy = np.argmin(np.abs(glat - lat))
            o = D[self.param][:, idx, idy]
            out[:, ix] = (
                utils.interpolate_by_altitude(
                    galt, alts, o, self.cfg.scale, self.cfg.kind, method="extp"
                )
                * 1e-6
            )
            ix += 1
        self.param, self.alts = out, galt
        return out, galt

    @staticmethod
    def create_object(
        cfg,
        year=2017,
        folder="dataset/GITM/20170821/",
        param="eden",
        time=dt.datetime(2017, 8, 21, 17, 30),
        lats=[],
        lons=[],
        alts=[],
        to_file=None,
        prev=False,
    ):
        mobj = {}
        gitm = GITM(cfg, year, folder, param)
        mobj["ne"], _ = gitm.fetch_dataset_by_locations(time, lats, lons, alts)
        if to_file:
            savemat(to_file, mobj)
        return gitm

    @staticmethod
    def create_density_files(
        cfg,
        year=2017,
        folder="dataset/GITM/20170821/",
        param="eden",
        time=dt.datetime(2017, 8, 21, 17, 30),
        lats=[],
        lons=[],
        alts=[],
        to_file=None,
        prev=None
    ):
        gitm = GITM.create_object(
            cfg, year, folder,
            param, time, lats,
            lons, alts, to_file,
        )
        if prev:
            GITM.create_object(
                cfg, year, folder,
                param, time - dt.timedelta(minutes=30), lats,
                lons, alts, prev,
            )
        return gitm


class IRI2D(object):
    
    def __init__(
        self, 
        time, 
        lats, 
        lons, 
        alts, 
        to_file=None, 
        cfg=None, 
        eclipse_prop = dict(),
        iri_version=20,
    ):
        """
        Setup all parameters
        """
        self.time = time
        self.lats = lats
        self.lons = lons
        self.alts = alts
        self.to_file = to_file
        self.eclipse_prop = eclipse_prop
        self.cfg = cfg
        self.alt_range = [alts[0], alts[-1], alts[1]-alts[0]]
        self.iri_version = iri_version
        self.load_data()
        if "start_mask_time" in eclipse_prop and\
            eclipse_prop["start_mask_time"]<=time:
            self.load_eclipse()
        self.save_to_file()
        return
    
    def load_data(self, iri_version=None):
        """
        Load all IRI dataset
        """
        import iricore
        iri_version = iri_version if iri_version else self.iri_version
        self.param_val = np.zeros((len(self.alts), len(self.lats)))
        for i in range(len(self.lats)):
            iriout = iricore.iri(
                self.time, 
                self.alt_range,
                self.lats[i],
                self.lons[i],
                iri_version,
            )
            self.param_val[:,i] = iriout.edens * 1e-6
        return
    
    def load_eclipse(self):
        """
        Create Eclipse in the system
        """
        import sys
        sys.path.append("py/")
        from eutils import Eclipse
        e = Eclipse()
        logger.info(f"Inside eclipse mask: {self.time}")
        oclt = np.zeros([len(self.alts), len(self.lons)])
        for i in range(len(self.alts)):
            for j in range(len(self.lons)):
                oclt[i,j] = e.create_eclipse_shadow(
                    self.time, self.lats[j], 
                    self.lons[j], self.alts[i]
                )
        oclt[oclt>=0.96] = 1.
        logger.info(f"Min/max value O: {np.nanmax(oclt)}/{np.nanmin(oclt)}")
        oclt = np.nan_to_num(oclt)
        p = 1.0 - oclt
        logger.info(f"Min/max value P: {np.nanmax(p)}/{np.nanmin(p)}")
        self.param_val = self.param_val * p
        return
    
    def save_to_file(self, to_file=None):
        """
        Save to file
        """
        to_file = to_file if to_file else self.to_file
        logger.info(f"IRI save to file: {to_file}")
        if to_file:
            mobj = {}
            mobj["ne"] = self.param_val
            savemat(to_file, mobj)
        return

pconst = {
    "boltz": 1.38066e-23,  # Boltzmann constant  in Jule K^-1
    "h": 6.626e-34,  # Planks constant  in ergs s
    "c": 2.9979e08,  # in m s^-1
    "avo": 6.023e23,  # avogadro's number
    "Re": 6371.0e3,
    "amu": 1.6605e-27,
    "q_e": 1.602e-19,  # Electron charge in C
    "m_e": 9.109e-31,  # Electron mass in kg
    "g": 9.81,  # Gravitational acceleration on the surface of the Earth
    "eps0": 1e-9 / (36 * np.pi),
    "R": 8.31,  # J mol^-1 K^-1
}

class WACCMX(object):

    def __init__(
        self, cfg,
        fname="../SolarEclipse/dataset/29jan2022/ECL_1100_0100_all.nc", 
        params = [
            {
                "name": "e",
                "unit_multiplier": 1.,
                "unit": "cc",
                "mol_to_cc": "T",
                "interpolate": {
                    "scale": "log",
                    "type": "linear"
                }
            }
        ],
        lat_res=1,
        lon_res=1,
        event=dt.datetime(2017,8,21)
    ):
        self.cfg = cfg
        self.lat_res = lat_res
        self.lon_res = lon_res
        self.event = event
        self.fname = fname
        self.params = params
        self.dataframe = pd.DataFrame()
        self.load_file()
        self.extract_dimension()
        self.run_extractions()
        return

    def transform_density(self, T, var, unit="cc"):
        """
        Transform electron density from mol/mol to /cc or /cm
        Formula
        -------
        P = nKT, K = Boltzmann constant in SI
        lev: pressure in hPa (1hPa=100Pa)
        T: neutral temperature
        e/O2/...: density in mol/mol

        T in K, P in Pa, n is electron density in /cubic meter or /cc
        """
        logger.info(f"Transform density scale/unit {self.fname}")
        P = self.lev * 1e2
        u = 1 if unit == "cm" else 1e-6
        den = np.zeros_like(var)
        for i, p in enumerate(P):
            na = p * pconst["avo"] / (pconst["R"] * T[:, i, :, :])
            n = u * na * var[:, i, :, :]
            den[:, i, :, :] = n
        return den

    def load_file(self):
        """
        Load files from local dataset.
        """
        logger.info(f"Load {self.fname}")
        self.ds = nc.Dataset(self.fname)
        return

    def extract_dimension(self):
        """
        Extract data dimensions
        """
        # Change these values
        x = np.arange(0, 91, self.lat_res)
        y = np.arange(-180, 0, self.lon_res)
        X, Y = np.meshgrid(x, y)
        self.dataframe["lat"], self.dataframe["lon"] = X.ravel(), Y.ravel()
        logger.info(f"Run extraction dimension: {self.fname}")
        self.time = [
            self.event + dt.timedelta(seconds=float(d))
            for d in self.ds.variables["time"][:]
        ]
        self.latx = self.ds.variables["lat"][:]
        self.lonx = self.ds.variables["lon"][:]
        self.lonx = np.where(self.lonx > 180, self.lonx - 360, self.lonx)
        self.lev = self.ds.variables["lev"][:]
        self.Z3 = self.ds.variables["Z3GM"][:] / 1e3
        return

    def run_extractions(self):
        """
        Extract informations from
        the file.
        """
        self.dataset = {}
        for i, p in enumerate(self.params):
            logger.info(f"Load {self.fname}, {p['name']}")
            var = self.ds.variables[p["name"]][:]
            if "mol_to_cc" in p.keys():
                var = (
                    self.transform_density(
                        self.ds.variables[p["mol_to_cc"]][:], var, p["unit"]
                    )
                    * p["unit_multiplier"]
                )
            self.dataset[i] = {}
            self.dataset[i]["value"] = var
        return

    def fetch_dataset_by_locations(
        self, time, lats, lons, alts, 
        dlat=0.2, dlon=0.2, to_file=None
    ):
        """
        Fetch data by lat/lon limits
        """
        time_ind = self.time.index(time)
        ds = self.dataset[0]
        logger.info(f"Time: {time}, {time_ind}")
        out, ix = np.zeros((len(alts), len(lons))) * np.nan, 0
        for lat, lon in zip(lats, lons):
            i_lon, i_lat = (
                np.argmin(np.abs(lon-self.lonx)),
                np.argmin(np.abs(lat-self.latx))
            )
            alt, d = (
                self.Z3[time_ind, :, i_lat, i_lon],
                ds["value"][time_ind, :, i_lat, i_lon]
            )
            method = "intp" if np.max(alt) > max(alts) else "extp"
            out[:, ix] = utils.interpolate_by_altitude(
                np.array(alt), alts, np.array(d),
                self.cfg.scale, self.cfg.kind, method=method
            )
            ix += 1
        self.param_val = out
        if to_file:
            mobj = {}
            mobj["ne"] = out
            savemat(to_file, mobj)
        return out

if __name__ == "__main__":
    w = WACCMX()