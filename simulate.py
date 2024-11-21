"""
    This python module is used to simulate all events
"""

import sys
sys.path.append("py/")
import argparse
import datetime as dt
from dateutil import parser as dparser
import eutils as utils
import numpy as np

from analysis_plot import genererate_RTI, genererate_Fan, generate_fovt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", default=dt.datetime(2021,12,4,6), help="Start date", type=dparser.parse)
    parser.add_argument("-e", "--end", default=dt.datetime(2021,12,4,10), help="End date", type=dparser.parse)
    parser.add_argument("-r", "--rad", default="fir", help="Radar code", type=str)
    parser.add_argument("-f", "--file_type", default="fitacf", help="File type other than fitacf", type=str)
    parser.add_argument("-b", "--beam", default=15, help="Radar beam", type=int)
    parser.add_argument("-m", "--method", default="rti", help="Method name", type=str)
    parser.add_argument("-p", "--params", default="v", help="Parameters name", type=str)
    parser.add_argument("-d", "--fan_date", default=dt.datetime(2021,12,4,6), help="Start date", type=dparser.parse)
    parser.add_argument("-fw", "--fan_time_window", default=90, help="Fan run time if invoke Fan", type=int)
    args = parser.parse_args()
    args.params = args.params.split(",")
    for k in vars(args).keys():
        print("     " + k + "->" + str(vars(args)[k]))
    
    if args.method == "rti":
        beams = [args.beam]
        generate_fovt(
            args.rad,
            [args.start, args.end],
        )
        # if args.beam == -1: beams = np.arange(24)
        # for bm in beams:
        #     genererate_RTI(
        #         args.rad, bm,
        #         [args.start, args.end],
        #         args.file_type,
        #         param_list=args.params,
        #         channel=2, tfreq=10.5
        #     )
        #     pass
    elif args.method == "fan":
        genererate_Fan(
            args.rad, [args.start, args.end],
            args.fan_date, args.file_type,
            param=args.params[0]
        )
    elif args.method == "fan-movie":
        for i in range(args.fan_time_window):
            date = args.fan_date + dt.timedelta(minutes=i)
            genererate_Fan(
                args.rad, [args.start, args.end],
                date, args.file_type,
                param=args.params[0]
            )
        utils.create_movie("figures", "movie.mp4", "fan*.png")