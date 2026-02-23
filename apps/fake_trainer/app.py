import os
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from kubernetes import client, config
from pydantic import BaseModel, Field


INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://inference-server:8000")
NAMESPACE = os.environ.get("POD_NAMESPACE", "default")
CONFIGMAP_NAME = os.environ.get("MODEL_CONFIGMAP_NAME", "model-weights")
VERSION_KEY = os.environ.get("MODEL_VERSION_KEY", "version.txt")
INFERENCE_POD_LABEL_SELECTOR = os.environ.get("INFERENCE_POD_LABEL_SELECTOR", "app=inference-server")


app = FastAPI(title="Fake Trainer (Sim)", version="0.1")


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=0)


class UpdateRequest(BaseModel):
    version: str = Field(..., min_length=1)


@app.on_event("startup")
def _startup() -> None:
    # In-cluster first; fall back to local kubeconfig for minikube dev.
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.post("/send_prompt")
def send_prompt(req: PromptRequest) -> dict[str, Any]:
    try:
        r = requests.post(f"{INFERENCE_URL}/generate", json={"prompt": req.prompt, "max_tokens": 32}, timeout=10)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to call inference service: {e}") from e
    return {
        "status_code": r.status_code,
        "model_version": r.headers.get("X-Model-Version"),
        "response": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text,
    }


def _reload_inference_replicas() -> list[dict[str, Any]]:
    """
    Best-effort: force all inference pods to reload immediately after an update.
    This avoids waiting for ConfigMap volume propagation delays.
    """
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(NAMESPACE, label_selector=INFERENCE_POD_LABEL_SELECTOR)

    results: list[dict[str, Any]] = []
    for pod in pods.items:
        pod_ip = getattr(pod.status, "pod_ip", None)
        pod_name = pod.metadata.name if pod.metadata else "unknown"
        if not pod_ip:
            results.append({"pod": pod_name, "reloaded": False, "error": "missing pod_ip"})
            continue
        try:
            r = requests.post(f"http://{pod_ip}:8000/reload", timeout=5)
            results.append({"pod": pod_name, "reloaded": r.ok, "status_code": r.status_code})
        except requests.RequestException as e:
            results.append({"pod": pod_name, "reloaded": False, "error": str(e)})
    return results


@app.post("/update_model")
def update_model(req: UpdateRequest) -> dict[str, Any]:
    v1 = client.CoreV1Api()
    body = {"data": {VERSION_KEY: f"{req.version}\n"}}
    try:
        v1.patch_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE, body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to patch ConfigMap {CONFIGMAP_NAME}: {e}") from e

    reload_results = _reload_inference_replicas()
    return {
        "updated": True,
        "configmap": CONFIGMAP_NAME,
        "namespace": NAMESPACE,
        "version": req.version,
        "reload_results": reload_results,
    }
