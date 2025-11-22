"""
FastAPI application for serving the Capsule Memory Engine.

To run this server:
    uvicorn capsule_full_project.api.app:app --reload

The `/predict` endpoint accepts a JSON payload of numeric features and
returns the engine's action logits and selected capacity.
"""

from typing import Any, Dict

from fastapi import FastAPI
import torch

from capsule_full_project.engine import CapsuleMemoryEngine, EngineConfig


app = FastAPI(title="Capsule Memory Engine API")

# Initialize engine once at startup
cfg = EngineConfig()
engine = CapsuleMemoryEngine(cfg)
engine.eval()


@app.post("/predict")
async def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Accepts a dictionary of features and returns engine outputs."""
    with torch.no_grad():
        # Convert values to tensors of shape (1, d)
        feats = {}
        for k, v in features.items():
            if isinstance(v, list):
                feats[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            else:
                feats[k] = torch.tensor([[float(v)]], dtype=torch.float32)
        out = engine.forward_step(feats)
        logits = out["action_logits"].squeeze(0).tolist()
        capacity = int(out["capacity"].item())
        return {"action_logits": logits, "capacity": capacity}