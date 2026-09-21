from fastapi.testclient import TestClient

from pathology.config import Settings
from pathology.main import create_app


def test_local_dotenv_loads_but_environment_wins(tmp_path, monkeypatch):
    # Monkeypatch records/restores all keys even though dotenv writes os.environ itself.
    keys = ["LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "LLM_API_KEY=test-only-secret\nLLM_MODEL=deepseek-flash\nLLM_BASE_URL=https://api.deepseek.com\n"
    )
    monkeypatch.setenv("LLM_MODEL", "explicit-model")
    settings = Settings.from_env(env)
    assert settings.llm_api_key == "test-only-secret"
    assert settings.llm_model == "explicit-model"
    assert settings.llm_provider == "DeepSeek"


def test_health_names_provider_without_disclosing_key(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, llm_api_key="test-only-secret"))) as client:
        response = client.get("/api/health")
    assert response.json()["assistant_provider"] == "DeepSeek"
    assert response.json()["assistant_model"] == "deepseek-flash"
    assert "test-only-secret" not in response.text


def test_provider_detection_uses_exact_hostname(tmp_path):
    assert (
        Settings(data_dir=tmp_path, llm_base_url="https://api.deepseek.com.fake.example").llm_provider
        != "DeepSeek"
    )
    assert (
        Settings(
            data_dir=tmp_path, llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).llm_provider
        == "通义千问"
    )
