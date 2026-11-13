"""Fetch NASA PCoE cells run under other protocols (4 C, 43 C, 4 A discharge) and
extract them to data/shift_capacity.csv. Used only to test the serving gates;
never used for training."""
import os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import extract  # noqa: E402

CELLS = ["B0029", "B0033", "B0036", "B0045", "B0053"]
MIRROR = "https://raw.githubusercontent.com/natskiu/Nasa-Battery/main/Data/{}.mat"
raw = os.path.join(ROOT, "data", "raw_shift")
os.makedirs(raw, exist_ok=True)
for c in CELLS:
    dst = os.path.join(raw, c + ".mat")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(MIRROR.format(c), dst)
extract.CELLS = CELLS
extract.main(raw, raw)
os.replace(os.path.join(raw, "capacity.csv"), os.path.join(ROOT, "data", "shift_capacity.csv"))
