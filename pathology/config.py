import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_backend: str = "demo"
    model_weights: str = "weights/model.pt"
    foundation_base_path: str = ""
    max_upload_mb: int = 256
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-flash"
    llm_api_key: str = ""

    @property
    def llm_provider(self):
        from urllib.parse import urlparse

        hostname = urlparse(self.llm_base_url).hostname or ""
        if hostname == "api.deepseek.com":
            return "DeepSeek"
        if hostname == "dashscope.aliyuncs.com" or hostname.endswith(".maas.aliyuncs.com"):
            return "通义千问"
        return "兼容模型服务"

    @classmethod
    def from_env(cls, env_file=None):
        # Explicit environment variables take precedence; secrets stay on this machine.
        load_dotenv(env_file if env_file is not None else PROJECT_ROOT / ".env", override=False)
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            model_backend=os.getenv("MODEL_BACKEND", "demo"),
            model_weights=os.getenv("MODEL_WEIGHTS", "weights/model.pt"),
            foundation_base_path=os.getenv("FOUNDATION_BASE_PATH", ""),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "256")),
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        )
