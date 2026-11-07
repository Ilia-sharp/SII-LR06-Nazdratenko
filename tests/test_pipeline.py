from __future__ import annotations

import json
from dataclasses import replace

from fastapi.testclient import TestClient

from src.artifacts import load_bundle
from src.api import create_app
from src.config import CONFIG
from src.pipeline import train_and_evaluate


def _write_csv(events, path):
    frame = events.copy()
    frame["timestamp"] = (
        frame["timestamp"].astype("int64") // 1_000_000
    )
    frame.to_csv(path, index=False)


def test_full_pipeline_and_api(tmp_path, small_events):
    events_path = tmp_path / "events.csv"
    _write_csv(small_events, events_path)
    artifacts = tmp_path / "artifacts"
    config = replace(
        CONFIG,
        target_mod=30,
        target_remainder=15,
        context_size=5,
        top_n=10,
        test_fraction=0.2,
    )

    metadata = train_and_evaluate(
        events_path, artifacts, config=config, progress=lambda _: None
    )
    assert (artifacts / "model.npz").exists()
    assert (artifacts / "metadata.json").exists()
    assert metadata["evaluation"]["evaluated_visitors"] == 1
    assert metadata["example"]["visitor_id"] == 15

    bundle = load_bundle(artifacts)
    assert bundle.test_visitors.tolist() == [15]
    assert bundle.test_items.tolist() == [3]

    app = create_app(artifacts)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.get("/recommend", params={"visitor_id": 15, "n": 10})
    assert response.status_code == 200
    payload = response.json()
    assert payload["visitor_id"] == 15
    assert payload["context_items"]
    assert all(item["itemid"] != 1 for item in payload["recommendations"])
    assert 3 in [item["itemid"] for item in payload["recommendations"]]

    post = client.post(
        "/recommend", json={"item_ids": [1, 2], "n": 5}
    )
    assert post.status_code == 200
    assert post.json()["input_items"] == [1, 2]
    assert all(item["itemid"] not in (1, 2) for item in post.json()["recommendations"])

    cold = client.get("/recommend", params={"visitor_id": 999999})
    assert cold.status_code == 200
    assert cold.json()["model"] == "popularity_cold_start"

    assert client.get("/metrics").status_code == 200
    assert "covisitation" in client.get("/metrics").json()

    metrics_file = json.loads((artifacts / "metrics.json").read_text("utf-8"))
    assert "popularity" in metrics_file["evaluation"]
