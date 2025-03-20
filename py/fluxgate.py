import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy.signal import detrend

def get_fluxgate(
    date=dt.datetime(2021,12,4), 
    stns=["pg0", "pg1", "pg2", "pg3", "pg4", "pg5"], 
    folder="database/magdataset/"
):
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame()
    for stn in stns:
        doy = f"{date.year}{'%04d'%date.timetuple().tm_yday}"
        url = f"http://mist.ece.vt.edu/data/fluxgate/ascii?site={stn}&YMD={doy}"
        file = os.path.join(folder, f"{stn}_{doy}.dat")
        if not os.path.exists(file):
            os.system(f"wget -O {file} {url}")
        o = []
        with open(file, "r") as f:
            lines = f.readlines()
        for i, l in enumerate(lines[1:]):
            l = [float(r) for r in list(filter(None, l.split(" ")))]
            o.append(dict(
                bx=l[0],
                by=l[1],
                bz=l[2],
                stn=stn,
                bh=np.sqrt(l[1]**2+l[2]**2),
                date=date+dt.timedelta(seconds=i)
            ))
        o = pd.DataFrame.from_records(o)
        o.dropna(inplace=True)
        o.bh = detrend(o.bh)
        df = pd.concat([df, o])
    return df

if __name__ == "__main__":
    get_fluxgate()