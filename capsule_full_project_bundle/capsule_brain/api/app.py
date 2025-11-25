"""REST API exposing the Capsule Brain."""

from typing import Any, Dict

from flask import Flask, jsonify, request
import torch

from ..core.brain import CapsuleBrain
from ..core.config import CapsuleBrainConfig


def create_app(brain: CapsuleBrain) -> Flask:
    """
    Create a Flask application exposing the Capsule Brain through a
    simple REST API.  The API maintains internal recurrent state
    between calls so that successive observations can be processed
    coherently.  Endpoints:

    - ``GET /skills``: list available skills.
    - ``GET /state``: return current memory pressure and emotion.
    - ``POST /step``: execute one step given observations and return
      the chosen action.
    """
    app = Flask(__name__)
    # Store previous hidden state between calls.  This dictionary
    # contains keys 'h' (FRNN hidden), 'h_ws' (workspace GRU hidden),
    # 'workspace' and 'emotion'.
    prev_state: Dict[str, Any] = {}

    @app.route("/skills", methods=["GET"])
    def list_skills() -> Any:
        return jsonify({"skills": list(brain.skills.keys())})

    @app.route("/state", methods=["GET"])
    def get_state() -> Any:
        # Return a snapshot of memory pressure and mean emotion
        pressure = float(brain.pmm.compute_pressure())
        # Use zero batch to query emotion; do not update state
        with torch.no_grad():
            # dummy zero image of correct size (assume 3×64×64)
            device = next(brain.parameters()).device
            dummy = torch.zeros(1, 3, 64, 64, device=device)
            out = brain(dummy, env=None, prev_state=None)
            emotion = out["emotion"].mean(dim=0).cpu().tolist()
        return jsonify({"memory_pressure": pressure, "emotion": emotion})

    @app.route("/step", methods=["POST"])
    def step_endpoint() -> Any:
        nonlocal prev_state
        data = request.get_json(force=True)
        pixels = torch.tensor(data["pixels"], dtype=torch.float32).unsqueeze(0)
        scalars = None
        goals = None
        env_name = data.get("env")
        device = next(brain.parameters()).device
        pixels = pixels.to(device)
        with torch.no_grad():
            out = brain(
                pixels,
                scalars=scalars,
                goals=goals,
                audio=None,
                env=env_name,
                prev_state=prev_state,
            )
        # Update state for next call
        prev_state = {
            "h": out.get("h"),
            "h_ws": out.get("h_ws"),
            "workspace": out.get("workspace"),
            "emotion": out.get("emotion"),
        }
        action_logits = out["action_logits"]
        action = int(torch.argmax(action_logits, dim=-1).item())
        return jsonify({"action": action})

    return app


if __name__ == "__main__":
    # Instantiate a brain on the appropriate device
    cfg = CapsuleBrainConfig()
    brain = CapsuleBrain(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    brain.to(device)
    app = create_app(brain)
    app.run(host="0.0.0.0", port=5000)
