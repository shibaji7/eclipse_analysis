import pydarn
import numpy as np

def save_fov(rad):
    hdw = pydarn.read_hdw_file(rad)
    latFull, lonFull = pydarn.Coords.GEOGRAPHIC(hdw.stid)
    header = ",".join([f"beam_{b}" for b in range(np.min(latFull.shape))])
    np.savetxt(f"figures/{rad}_lat.csv", latFull, delimiter=',', fmt='%s', header=header, comments='')
    np.savetxt(f"figures/{rad}_lon.csv", lonFull, delimiter=',', fmt='%s', header=header, comments='')
    return

if __name__ == "__main__":
    save_fov("fir")