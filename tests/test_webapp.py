"""The live lab's API: a run streams its progress, then predictions and rule edits work."""
import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from webapp import jobs  # noqa: E402
from webapp.server import app  # noqa: E402

FAST = {"slow_motion": False, "epochs": 40, "oracle_samples": 500, "n_rules": 4}


@pytest.fixture(scope="module")
def run():
    client = TestClient(app)
    run_id = client.post("/api/runs", json=FAST).json()["id"]
    events = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: {}":
                events.append(json.loads(line[len("data: "):]))
    return client, run_id, events


def test_config_is_validated_and_clamped():
    cfg = jobs.clean_config({"n_rules": 99, "epochs": 1, "order": "1", "unknown": 5})
    assert cfg["n_rules"] == 16 and cfg["epochs"] == 20 and cfg["order"] == 1 and "unknown" not in cfg
    with pytest.raises(ValueError):
        jobs.clean_config({"hidden_layers": "999"})


def test_run_streams_every_stage_live(run):
    _, _, events = run
    kinds = [e["type"] for e in events]
    assert "error" not in kinds, [e for e in events if e["type"] == "error"]
    for kind in ("data", "nn_epoch", "nn_done", "importance", "rules_proposed", "anfis_epoch", "done"):
        assert kind in kinds
    assert kinds.count("nn_epoch") > 10                       # one event per network epoch
    done_stages = {e["stage"] for e in events if e["type"] == "stage" and e["status"] == "done"}
    assert {"data", "network", "importance", "rules", "anfis", "prune", "evaluate"} <= done_stages


def test_prediction_matches_the_explanation(run):
    client, run_id, events = run
    done = next(e for e in events if e["type"] == "done")
    out = client.post(f"/api/runs/{run_id}/predict", json={"x": done["patients"]["x"][0]}).json()
    assert 0 <= out["nn_prob"] <= 1
    assert abs(sum(r["contribution"] for r in out["rules"]) - out["fuzzy_output"]) < 1e-6
    assert client.post(f"/api/runs/{run_id}/predict", json={"x": [1, 2]}).status_code == 422


def test_rule_edits_refit_and_are_validated(run):
    client, run_id, events = run
    done = next(e for e in events if e["type"] == "done")
    rules = [r["antecedent"] for r in done["model"]["rules"]]
    edited = [[-1] * len(rules[0])] + rules[1:]                 # first rule becomes "always"
    res = client.post(f"/api/runs/{run_id}/rules", json={"antecedents": edited}).json()
    assert res["metrics"]["conditions"] <= done["summary"]["final"]["conditions"]
    assert client.get(f"/api/runs/{run_id}/model").json()["metrics"] == res["metrics"]
    assert client.post(f"/api/runs/{run_id}/rules", json={"antecedents": [[7] * len(rules[0])]}).status_code == 422
    client.post(f"/api/runs/{run_id}/rules", json={"antecedents": rules})   # reset
    rows = client.get("/api/runs").json()
    assert any(r["id"] == run_id and r["status"] == "done" for r in rows)
