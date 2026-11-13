import numpy as np
import pandas as pd
import pytest

from mlops.core import CellEOLForecaster, HParams, analytic_eol, loco_eval
from src.lifetime import eol_cycle, power_law

DF = pd.read_csv("data/capacity.csv")
SHIFT = pd.read_csv("data/shift_capacity.csv")


@pytest.fixture(scope="module")
def model():
    return CellEOLForecaster.fit(DF[DF.cell != "B0005"])


@pytest.mark.parametrize("log_a,z", [(-8.0, 1.2), (-5.0, 0.8), (-11.0, 2.0)])
def test_analytic_eol_matches_research_grid_search(log_a, z):
    assert analytic_eol(log_a, z) == eol_cycle(lambda n: power_law([log_a, z], n, 1.0), 1.0)


def test_predict_is_deterministic_and_ordered(model):
    h = DF[DF.cell == "B0005"].sort_values("cycle").iloc[:40]
    a, b = model.predict(h), model.predict(h)
    assert a == b
    assert a["eol_p05"] <= a["eol_p50"] <= a["eol_p95"]


def test_json_roundtrip(model):
    h = DF[DF.cell == "B0005"].sort_values("cycle").iloc[:40]
    assert CellEOLForecaster.from_json(model.to_json()).predict(h) == model.predict(h)


def test_short_history_is_routed(model):
    h = DF[DF.cell == "B0005"].sort_values("cycle").iloc[:15]
    p = model.predict(h)
    assert p["decision"] == "route_to_engineer"
    assert any(r.startswith("too_few_cycles") for r in p["reasons"])


def test_zero_capacity_is_rejected(model):
    h = DF[DF.cell == "B0005"].sort_values("cycle").iloc[:40].copy()
    h.loc[h.index[10], "capacity_Ah"] = 0.0
    assert model.predict(h)["decision"] == "reject"


def test_hot_cell_is_routed(model):
    h = SHIFT[SHIFT.cell == "B0029"].sort_values("cycle")
    p = model.predict(h)
    assert p["decision"] == "route_to_engineer"
    assert any(r.startswith("ood:temperature") for r in p["reasons"])


def test_tuned_beats_fleet_mean_baseline():
    r = loco_eval(DF, HParams(), ks=(30, 40, 50))
    from mlops.core import fleet_mean_baseline
    assert r["mae"] < fleet_mean_baseline(DF)


def test_glitch_crossing_80pct_is_not_reported_as_eol(model):
    h = DF[DF.cell == "B0006"].sort_values("cycle").iloc[:40].copy().reset_index(drop=True)
    h.loc[19:, "capacity_Ah"] *= 0.75
    p = model.predict(h)
    assert p["decision"] == "route_to_engineer"
    assert "eol_crossing_with_anomaly" in p["reasons"]
