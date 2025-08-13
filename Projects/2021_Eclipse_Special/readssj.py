import copy
from loguru import logger
import numpy as np
import pandas as pd
import struct

class SSJ(object):

    def __init__(self, filename):
        self.filename = filename
        self.read()
        return

    def extract_big_end(self, f):
        raw_bytes = f.read(2)
        value = struct.unpack('>H', raw_bytes)[0]
        return value

    def read(self):
        with open(self.filename, "rb") as file:
            doy, hrs, mins, secs, year = (
                self.extract_big_end(file),
                self.extract_big_end(file),
                self.extract_big_end(file),
                self.extract_big_end(file),
                self.extract_big_end(file)-50,
            )
            gd_glat, gd_glon, alt_km = (
                self.extract_big_end(file),
                self.extract_big_end(file)/10,
                self.extract_big_end(file)*1.852,
            )
            gd_glat = (gd_glat-900)/10 if gd_glat<=1800 else (gd_glat-4995)/10
            geo_glat, geo_glon = (
                self.extract_big_end(file),
                self.extract_big_end(file)/10,
            )
            geo_glat = (geo_glat-900)/10 if geo_glat<=1800 else (geo_glat-4995)/10
            gm_glat, gm_glon, mlt_hr, mlt_min, mlt_sec = (
                self.extract_big_end(file),
                self.extract_big_end(file)/10,
                self.extract_big_end(file),
                self.extract_big_end(file),
                self.extract_big_end(file),
            )
            gm_glat = (gm_glat-900)/10 if gm_glat<=1800 else (gm_glat-4995)/10
            print(gm_glat, gm_glon, mlt_hr, mlt_min, mlt_sec)

            data_hr_day, data_min_day, data_sec_day = (
                self.extract_big_end(file),
                self.extract_big_end(file),
                self.extract_big_end(file),
            )
            print(data_hr_day, data_min_day, data_sec_day)     
        return

if __name__ == "__main__":
    ssj = SSJ("j5f1821335")