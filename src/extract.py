"""Extract per-cycle discharge records from NASA PCoE Li-ion aging .mat files.

Writes data/capacity.csv: one row per discharge cycle with capacity, start time,
rest time since the previous discharge ended, peak temperature and end voltage.
"""
import argparse, os, datetime as dt
import numpy as np, pandas as pd, scipy.io as sio

CELLS = ["B0005", "B0006", "B0007", "B0018"]  # 24 C, 2 A CC discharge, same protocol


def to_dt(v):
    v = np.atleast_1d(v).astype(float)
    sec = v[5]
    return dt.datetime(int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4])) + dt.timedelta(seconds=sec)


def main(raw_dir, out_dir):
    rows = []
    for cell in CELLS:
        cyc = sio.loadmat(os.path.join(raw_dir, cell + ".mat"), simplify_cells=True)[cell]["cycle"]
        last_end = None
        k = 0
        for c in cyc:
            start = to_dt(c["time"])
            d = c["data"]
            if c["type"] != "discharge":
                continue
            t = np.atleast_1d(d["Time"]).astype(float)
            V = np.atleast_1d(d["Voltage_measured"]).astype(float)
            Tc = np.atleast_1d(d["Temperature_measured"]).astype(float)
            rest_h = np.nan if last_end is None else (start - last_end).total_seconds() / 3600.0
            rows.append(dict(cell=cell, cycle=k + 1, capacity_Ah=float(np.atleast_1d(d["Capacity"])[0]),
                             t_start=start.isoformat(), rest_before_h=rest_h,
                             T_max_C=float(Tc.max()), V_end=float(V[-1])))
            last_end = start + dt.timedelta(seconds=float(t[-1]))
            k += 1
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "capacity.csv"), index=False)
    print(pd.DataFrame(rows).groupby("cell").capacity_Ah.agg(["count", "first", "last", "min"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    main(a.raw, a.out)
