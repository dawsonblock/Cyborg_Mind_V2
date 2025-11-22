"""
Minimal CLI for interacting with the Capsule Memory Engine.

This script prompts the user for a comma‑separated list of numbers,
processes them through the engine and prints the selected cognitive
capacity along with the action logits.  It demonstrates how to use
the engine in a simple interactive setting.
"""

import argparse
from typing import Dict

import torch

from capsule_full_project.engine import CapsuleMemoryEngine, EngineConfig


def parse_input(line: str, d_input: int) -> Dict[str, torch.Tensor]:
    # Parse numbers from the input string, pad or truncate to d_input
    parts = [float(x.strip()) for x in line.split(",") if x.strip()]
    tensor = torch.tensor(parts, dtype=torch.float32).unsqueeze(0)
    if tensor.size(-1) < d_input:
        pad = torch.zeros(1, d_input - tensor.size(-1))
        tensor = torch.cat([tensor, pad], dim=-1)
    elif tensor.size(-1) > d_input:
        tensor = tensor[:, :d_input]
    return {"obs": tensor}


def run_cli():
    cfg = EngineConfig()
    engine = CapsuleMemoryEngine(cfg)
    engine.eval()
    print("Enter comma‑separated numbers to feed into the engine (Ctrl+C to exit):")
    try:
        while True:
            line = input("> ")
            features = parse_input(line, cfg.encoder.d_input)
            with torch.no_grad():
                out = engine.forward_step(features)
                capacity = int(out["capacity"].item())
                logits = out["action_logits"].squeeze(0).tolist()
            print(f"Selected capacity: {capacity}\nAction logits: {logits}")
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    run_cli()