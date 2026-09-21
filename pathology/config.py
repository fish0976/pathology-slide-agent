import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_backend: str = "demo"
    model_weights: str = "weights/model.pt"
    max_upload_mb: int = 256
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen-plus"
    llm_api_key: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            model_backend=os.getenv("MODEL_BACKEND", "demo"),
            model_weights=os.getenv("MODEL_WEIGHTS", "weights/model.pt"),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "256")),
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            llm_model=os.getenv("LLM_MODEL", "qwen-plus"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
        )
