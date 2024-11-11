
import h5py    
import numpy as np
import datetime as dt

class GPS(object):

    def __init__(self, fname="database/conv_20240408T0000-20240409T0000.h5"):
        self.fname = fname
        with h5py.File(self.fname, "r") as f:
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            print(f[a_group_key].keys())
            self.lats = list(f[a_group_key]["lat"])
            self.lons = list(f[a_group_key]["lon"])
            self.tec = np.array(f[a_group_key]["im"])
            self.time = list(f[a_group_key]["time"])
            self.times = [dt.datetime(2024, 4, 8) + dt.timedelta(minutes=5*t) for t in range(288)]
            print(self.tec.shape)
        return

    def get_tec_along_beam(self, lats, lons):
        tec = np.zeros((len(self.times), len(lats)))*np.nan
        for i, t in enumerate(self.times):
            for j, lat, lon in zip(range(len(lats)), lats, lons):
                ix = np.argmin(np.abs([(tx-t).total_seconds() for tx in self.times]))
                iy = np.argmin(np.abs(lon-self.lons))
                iz = np.argmin(np.abs(lat-self.lats))
                print(lat, lon, self.tec[ix, iy, iz])
                tec[i, j] = self.tec[ix, iy, iz]
        print(np.nanmin(tec), np.nanmax(tec))
        tecv = dict(
            X=self.times,
            Y=180 + 45*np.arange(101),
            Z=tec
        )
        return tecv

g = GPS()