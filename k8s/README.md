# Kubernetes Model Serving + Training Simulation

This folder contains a minimal Kubernetes setup for:

- **Inference server cluster**: 2 replicas, load-balanced via a Service
  - Prompt processing: `POST /generate`
  - Model version tracking: response header `X-Model-Version`
  - Hot model reload: in-place reload via watching a mounted `ConfigMap` file (no restart)
- **Fake trainer server**: 1 replica
  - Sends prompts to the inference Service
  - Updates model "weights" by patching the `ConfigMap` (which triggers hot reload on all replicas)

## Local dev (minikube)

1) Start minikube
```bash
minikube start
```

2) Build images inside minikube Docker
```bash
eval "$(minikube docker-env)"
docker build -t inference-server:dev apps/inference_server
docker build -t fake-trainer:dev apps/fake_trainer
```

3) Apply manifests
```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/
```

4) Wait for pods
```bash
kubectl -n model-sim get pods -w
```

## Demo

Port-forward the trainer service:
```bash
kubectl -n model-sim port-forward svc/fake-trainer 8001:8000
```

Send a prompt (trainer -> inference Service):
```bash
curl -s localhost:8001/send_prompt -H 'content-type: application/json' -d '{"prompt":"hello"}' | jq
```

Update the model version (trainer patches ConfigMap; inference replicas reload in-place):
```bash
curl -s localhost:8001/update_model -H 'content-type: application/json' -d '{"version":"v2"}' | jq
```

Send another prompt and observe `model_version` (and header `X-Model-Version`):
```bash
curl -i -s localhost:8001/send_prompt -H 'content-type: application/json' -d '{"prompt":"world"}'
```

## Notes

- This is CPU-only "generation" (a simulation). You can swap in vLLM/SGLang/TGI later if needed.
- Replica synchronization:
  - All inference pods watch the same mounted `ConfigMap` key.
  - The trainer also calls `/reload` on each inference pod directly for faster, near-synchronous updates.
