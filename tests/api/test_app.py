"""Tests for the Auto-ART API (v1 + legacy endpoints)."""

import os
import pytest
import json
from unittest.mock import patch

# Disable auth for tests before importing the app
os.environ["AUTO_ART_AUTH_MODE"] = "disabled"

from auto_art.api.app import create_app


@pytest.fixture
def client():
    """Create a Flask test client with auth disabled."""
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# Status / Health
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_status_endpoint(client):
    response = client.get("/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "running"
    assert "version" in data
    assert "features" in data


def test_v1_status_endpoint(client):
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "running"


# ---------------------------------------------------------------------------
# Attack / Defense listing
# ---------------------------------------------------------------------------

def test_attacks_endpoint(client):
    response = client.get("/api/v1/attacks")
    assert response.status_code == 200
    data = response.get_json()
    assert "attacks" in data
    assert len(data["attacks"]) >= 50


def test_defenses_endpoint(client):
    response = client.get("/api/v1/defenses")
    assert response.status_code == 200
    data = response.get_json()
    assert "defenses" in data
    assert len(data["defenses"]) >= 15


# ---------------------------------------------------------------------------
# OpenAPI spec
# ---------------------------------------------------------------------------

def test_openapi_spec(client):
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.3"
    assert "paths" in data


# ---------------------------------------------------------------------------
# Evaluate endpoint (v1)
# ---------------------------------------------------------------------------

def test_evaluate_requires_json(client):
    response = client.post("/api/v1/evaluate", data="not json")
    assert response.status_code in (400, 415)


def test_evaluate_validates_fields(client):
    # Missing model_path should fail validation
    response = client.post("/api/v1/evaluate", json={
        "framework": "pytorch",
        "model_type": "classification",
    })
    assert response.status_code in (400, 422)


def test_evaluate_with_valid_payload(client):
    # Should accept valid payload (actual evaluation would need a real model)
    response = client.post("/api/v1/evaluate", json={
        "model_path": "/tmp/nonexistent_model.pt",
        "framework": "pytorch",
        "model_type": "classification",
        "num_samples": 10,
    })
    # May return 404 (model not found) or 500 (can't load), but not 400
    assert response.status_code in (200, 404, 500)


# ---------------------------------------------------------------------------
# Scan endpoint
# ---------------------------------------------------------------------------

def test_scan_with_valid_payload(client):
    response = client.post("/api/v1/scan", json={
        "model_path": "/tmp/test_model.pt",
        "preset": "quick_scan",
    })
    # Should accept the request (returns submitted or error finding model)
    assert response.status_code in (200, 404, 500)
    if response.status_code == 200:
        data = response.get_json()
        assert "scan_config" in data


# ---------------------------------------------------------------------------
# Legacy evaluate_model endpoint
# ---------------------------------------------------------------------------

def test_legacy_evaluate_model_not_json(client):
    response = client.post("/evaluate_model", data="this is not json")
    assert response.status_code in (400, 415)


def test_legacy_evaluate_model_valid(client):
    response = client.post("/evaluate_model", json={
        "model_path": "/tmp/test.pt",
        "framework": "pytorch",
        "model_type": "classification",
    })
    # May fail due to model not existing, but shouldn't be a validation error
    assert response.status_code in (200, 404, 500)
