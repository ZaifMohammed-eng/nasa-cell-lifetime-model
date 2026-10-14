"""Fetch the four NASA PCoE .mat files used here into data/raw/.

Tries a public GitHub mirror of the NASA set (raw .mat files). If that fails,
download "Battery Data Set" from the NASA Prognostics Data Repository
(https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
and copy B0005.mat, B0006.mat, B0007.mat and B0018.mat into data/raw/.
"""
import os, urllib.request

MIRROR = "https://raw.githubusercontent.com/natskiu/Nasa-Battery/main/Data/{}.mat"
CELLS = ["B0005", "B0006", "B0007", "B0018"]
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(out, exist_ok=True)
for c in CELLS:
    dst = os.path.join(out, c + ".mat")
    if os.path.exists(dst):
        continue
    for branch in ("main", "master"):
        try:
            urllib.request.urlretrieve(MIRROR.format(c).replace("/main/", f"/{branch}/"), dst)
            print("got", c)
            break
        except Exception as e:
            err = e
    else:
        print(f"could not download {c} ({err}); see the docstring for the NASA source")
