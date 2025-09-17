import pytest

from ai_cost_sdk.config import load_config


def test_load_config_requires_ids(monkeypatch):
    monkeypatch.delenv("TENANT_ID", raising=False)
    monkeypatch.delenv("PROJECT_ID", raising=False)
    with pytest.raises(ValueError):
        load_config()


def test_load_config_populates_defaults(monkeypatch):
    monkeypatch.setenv("TENANT_ID", "tenant")
    monkeypatch.setenv("PROJECT_ID", "project")
    monkeypatch.delenv("SERVICE_NAME", raising=False)
    cfg = load_config()
    assert cfg.tenant_id == "tenant"
    assert cfg.project_id == "project"
    assert cfg.service_name == "ai-cost-sdk"
