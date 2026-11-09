from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts import load_bundle, recommend_for_visitor

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "data" / "check_cases.json"
ARTIFACTS_DIR = ROOT / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "model.npz").exists(),
    reason="нет закоммиченных артефактов модели",
)


def test_check_cases_match_committed_model():
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert len(cases) >= 7

    bundle = load_bundle(ARTIFACTS_DIR)
    for case in cases:
        actual = recommend_for_visitor(
            bundle, case["visitor_id"], n=10
        )
        assert actual["model"] == case["expected_model"]
        assert actual["history_size"] == case["history_size"]
        assert actual["context_items"] == case["context_items"]

        actual_items = [item["itemid"] for item in actual["recommendations"]]
        expected_items = [item["itemid"] for item in case["top10"]]
        assert actual_items == expected_items

        actual_scores = [item["score"] for item in actual["recommendations"]]
        expected_scores = [item["score"] for item in case["top10"]]
        assert actual_scores == pytest.approx(expected_scores, abs=1e-6)
