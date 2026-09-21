import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import answer_question
from .config import Settings
from .imaging import synthetic_slide
from .pipeline import analyze
from .slides import SUPPORTED, Slide
from .store import Store

logger = logging.getLogger(__name__)


class AnalyzeOptions(BaseModel):
    tile_size: int = Field(256, ge=64, le=1024)
    max_tiles: int = Field(128, ge=1, le=512)
    level: int = Field(0, ge=0, le=20)
    normalize: bool = True


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(settings=None):
    settings = settings or Settings.from_env()
    store = Store(settings.data_dir)
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="slide")
    slots = threading.BoundedSemaphore(4)
    mutation_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_):
        yield
        executor.shutdown(wait=True)

    app = FastAPI(title="Pathology Slide Agent", version="0.1.0", lifespan=lifespan)
    app.state.store = store

    def get_job(job_id):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return job

    def job_dir(job_id):
        get_job(job_id)  # Only server-generated IDs in database may resolve to paths.
        return settings.data_dir / job_id

    def add_job(job_id, filename, suffix):
        job = {
            "id": job_id,
            "filename": filename,
            "suffix": suffix,
            "status": "uploaded",
            "progress": 0,
            "stage": "uploaded",
            "error": None,
            "report": None,
        }
        store.save(job)
        return job

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "model_backend": settings.model_backend,
            "assistant": "llm" if settings.llm_api_key else "local",
            "assistant_provider": settings.llm_provider if settings.llm_api_key else None,
            "assistant_model": settings.llm_model if settings.llm_api_key else None,
        }

    @app.get("/api/jobs")
    def jobs():
        return [{k: v for k, v in j.items() if k != "report"} for j in store.list()]

    @app.post("/api/demo", status_code=201)
    def demo():
        job_id = uuid.uuid4().hex
        folder = settings.data_dir / job_id
        folder.mkdir()
        im = synthetic_slide()
        im.save(folder / "source.png")
        im.thumbnail((1200, 1200))
        im.save(folder / "thumbnail.png")
        return add_job(job_id, "synthetic-he-demo.png", ".png")

    @app.post("/api/slides", status_code=201)
    async def upload(file: Annotated[UploadFile, File()]):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED:
            await file.close()
            raise HTTPException(415, "支持 PNG/JPG/SVS/NDPI/TIFF")
        job_id = uuid.uuid4().hex
        folder = settings.data_dir / job_id
        folder.mkdir()
        path = folder / ("source" + suffix)
        try:
            size = 0
            with path.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, f"文件超过 {settings.max_upload_mb} MB 上限")
                    stream.write(chunk)
            with Slide(path) as slide:
                slide.thumbnail().save(folder / "thumbnail.png")
        except Exception as exc:
            for child in folder.iterdir():
                child.unlink()
            folder.rmdir()
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(422, "切片无法读取；请检查格式、文件完整性及 OpenSlide 安装。") from exc
        finally:
            await file.close()
        return add_job(job_id, (file.filename or "slide").replace("\\", "/").split("/")[-1], suffix)

    def run_job(job, options):
        try:
            job.update(status="running", stage="read_slide")
            store.save(job)

            def progress(value, stage):
                job.update(progress=value, stage=stage)
                store.save(job)

            folder = settings.data_dir / job["id"]
            report = analyze(folder / ("source" + job["suffix"]), folder, settings, options, progress)
            job.update(status="completed", progress=100, stage="completed", report=report)
        except Exception:
            logger.exception("Analysis failed: %s", job["id"])
            job.update(status="failed", error="分析失败，请检查所选层级、模型权重和服务日志。")
        finally:
            store.save(job)
            slots.release()

    @app.post("/api/jobs/{job_id}/analyze", status_code=202)
    def start(job_id: str, options: AnalyzeOptions):
        with mutation_lock:
            job = get_job(job_id)
            if job["status"] in ("queued", "running"):
                raise HTTPException(409, "该切片正在分析")
            if not slots.acquire(blocking=False):
                raise HTTPException(429, "分析队列已满，请稍后重试")
            job.update(status="queued", progress=0, error=None, report=None)
            store.save(job)
            executor.submit(run_job, job.copy(), options.model_dump())
        return job

    @app.get("/api/jobs/{job_id}")
    def detail(job_id: str):
        return get_job(job_id)

    @app.get("/api/jobs/{job_id}/assets/{asset}")
    def asset(job_id: str, asset: str):
        if asset not in {"thumbnail.png", "heatmap.png"}:
            raise HTTPException(404, "资源不存在")
        path = job_dir(job_id) / asset
        if not path.exists():
            raise HTTPException(404, "资源尚未生成")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}/report")
    def download_report(job_id: str, format: str = "json"):
        report = get_job(job_id)["report"]
        if not report:
            raise HTTPException(409, "报告尚未生成")
        if format == "json":
            content = json.dumps(report, ensure_ascii=False, indent=2)
            media = "application/json"
        elif format == "md":
            content = f"# {report['title']}\n\n{report['summary']}\n\n"
            content += (
                "## 分析数据\n\n```json\n"
                + json.dumps(
                    {k: report[k] for k in ("mode", "quality", "sampling", "regions", "trace")},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n```\n\n## 适用边界\n\n"
            )
            content += "\n".join("- " + x for x in report["limitations"])
            media = "text/markdown"
        else:
            raise HTTPException(422, "仅支持 json 或 md")
        return Response(
            content,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="report-{job_id}.{format}"'},
        )

    @app.post("/api/jobs/{job_id}/chat")
    async def chat(job_id: str, question: Question):
        report = get_job(job_id)["report"]
        if not report:
            raise HTTPException(409, "请先完成分析")
        return await answer_question(question.question, report, settings)

    dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
