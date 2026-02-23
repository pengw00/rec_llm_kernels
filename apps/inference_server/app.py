import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
MODEL_VERSION_PATH = Path(os.environ.get("MODEL_VERSION_PATH", str(MODEL_DIR / "version.txt")))
POLL_INTERVAL_S = float(os.environ.get("MODEL_POLL_INTERVAL_S", "0.5"))


@dataclass
class ModelState:
    version: str = "unknown"
    last_loaded_ns: int = 0
    last_seen_mtime_ns: int = -1

    def load(self) -> bool:
        try:
            stat = MODEL_VERSION_PATH.stat()
        except FileNotFoundError:
            self.version = "missing"
            self.last_loaded_ns = time_ns()
            self.last_seen_mtime_ns = -1
            return True

        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
        if mtime_ns == self.last_seen_mtime_ns and self.version != "unknown":
            return False

        version = MODEL_VERSION_PATH.read_text(encoding="utf-8").strip() or "empty"
        self.version = version
        self.last_loaded_ns = time_ns()
        self.last_seen_mtime_ns = mtime_ns
        return True


def time_ns() -> int:
    return time.time_ns()


state = ModelState()
app = FastAPI(title="Inference Server (Sim)", version="0.1")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=0)
    max_tokens: int = Field(32, ge=1, le=4096)


@app.on_event("startup")
async def _startup() -> None:
    state.load()

    async def _watch() -> None:
        while True:
            try:
                state.load()
            except Exception:
                # Keep serving even if reload fails.
                pass
            await asyncio.sleep(POLL_INTERVAL_S)

    asyncio.create_task(_watch())


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "model_version": state.version}


@app.get("/model")
def model_info() -> dict[str, Any]:
    return {"model_version": state.version, "version_path": str(MODEL_VERSION_PATH)}


@app.post("/reload")
def reload_model() -> dict[str, Any]:
    changed = state.load()
    return {"reloaded": True, "changed": changed, "model_version": state.version}


@app.post("/generate")
def generate(req: GenerateRequest, response: Response) -> dict[str, Any]:
    # "Generate" a response deterministically for demo purposes.
    version = state.version
    response.headers["X-Model-Version"] = version
    response.headers["X-Inference-Backend"] = "sim"

    text = f"[model={version}] " + req.prompt[::-1]
    return {"model_version": version, "text": text}
