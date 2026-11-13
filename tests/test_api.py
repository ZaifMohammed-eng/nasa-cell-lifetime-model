import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops.core import CellEOLForecaster

DF = pd.read_csv("data/capacity.csv")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    path = d / "model.json"
    path.write_text(CellEOLForecaster.fit(DF[DF.cell != "B0006"]).to_json())
    os.environ["MODEL_URI"] = str(path)
    os.environ["REQUEST_LOG"] = str(d / "req.jsonl")
    import serve.app as app_mod
    with TestClient(app_mod.app) as c:
        yield c, d


def payload(cell, k, df=DF):
    g = df[df.cell == cell].sort_values("cycle").iloc[:k][["cycle", "capacity_Ah", "rest_before_h", "T_max_C"]]
    g = g.astype(object).where(g.notna(), None)
    return {"cell_id": cell, "cycles": g.to_dict("records")}


def test_health(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"


def test_forecast_in_distribution(client):
    c, d = client
    r = c.post("/v1/forecast", json=payload("B0006", 40))
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"forecast", "route_to_engineer"}
    assert body["eol_p05"] <= body["eol_p50"] <= body["eol_p95"]
    assert (d / "req.jsonl").read_text().strip()


def test_schema_violation_returns_422(client):
    c, _ = client
    assert c.post("/v1/forecast", json={"cell_id": "x", "cycles": [{"cycle": 0, "capacity_Ah": 1.0}]}).status_code == 422


def test_shifted_cell_routed(client):
    c, _ = client
    shift = pd.read_csv("data/shift_capacity.csv")
    assert c.post("/v1/forecast", json=payload("B0029", 40, shift)).json()["decision"] == "route_to_engineer"


def test_metrics_exposed(client):
    c, _ = client
    assert "eol_decisions_total" in c.get("/metrics").text


def test_demo_page_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "live demo" in r.text


def test_every_builtin_case_behaves_as_expected(client, monkeypatch):
    # Expectations are for the tuned champion; the fixture model uses untuned defaults,
    # whose wide intervals (correctly) route several held-out cases to an engineer.
    c, _ = client
    import serve.app as app_mod
    champ = CellEOLForecaster.from_json(open("serve/model/champion.json").read())
    monkeypatch.setitem(app_mod.STATE, "model", champ)
    ids = [x["id"] for x in c.get("/v1/cases").json()]
    assert len(ids) == 22
    failed = [i for i in ids if not c.post(f"/v1/cases/{i}/run").json()["pass"]]
    assert failed == []


def test_holdout_case_never_uses_the_scored_cell(client):
    c, _ = client
    r = c.post("/v1/cases/holdout-B0005-k40/run").json()
    assert "refit without B0005" in r["scored_by"] and "B0005" not in r["scored_by"].split("(cells")[1]


def test_unknown_case_404(client):
    c, _ = client
    assert c.post("/v1/cases/nope/run").status_code == 404


def test_rate_limit_per_ip(client, monkeypatch):
    c, _ = client
    import serve.app as app_mod
    monkeypatch.setattr(app_mod, "RATE_LIMIT_PER_MIN", 3)
    app_mod.RATE.clear()
    h = {"X-Forwarded-For": "203.0.113.5"}
    codes = [c.post("/v1/cases/holdout-B0006-k40/run", headers=h).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert c.post("/v1/cases/holdout-B0006-k40/run", headers={"X-Forwarded-For": "203.0.113.6"}).status_code == 200
    app_mod.RATE.clear()
