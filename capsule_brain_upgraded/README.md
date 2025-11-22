# Capsule Brain Integration (Fixed)

This repository contains a fully integrated version of the Capsule Brain with a Minecraft skill.  It incorporates a pseudo‑mode memory, a recurrent core, global workspace, continuous emotion model, capsule network, skill subsystem and supporting infrastructure (API, GUI, deployment, tests).  The package has been refactored to be runnable on standard hardware (CPU/GPU) and to support both training and inference.

## Installation

1. Create and activate a Python environment (Python 3.10 or later is recommended).
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

2. Install the required packages.  CUDA‑enabled PyTorch wheels are available from the [official PyTorch site](https://pytorch.org/get-started/locally/).  Adjust the `pip install torch` command below to match your GPU and CUDA version.
   ```bash
   # Example for CUDA 11.7; change to cpuonly if you do not have a GPU
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
   # Install other dependencies
   pip install -r requirements.txt
   ```

3. (Optional) Install the `minerl` package from [MineRL](https://minerl.readthedocs.io/).  This package is required to train the Minecraft skill and is included in `requirements.txt`.  If you plan to use MineDojo tasks, also install `minedojo`.

## Running the API

The REST API exposes endpoints to query available skills, get internal state and step through the brain.

```bash
python -m capsule_brain.api.app
```

By default the API listens on port 5000.  Use the following endpoints:

* `GET /skills` – returns a list of registered skills (e.g. `minecraft_agent`).
* `GET /state` – returns current memory pressure and a summarised emotion vector.
* `POST /step` – accepts a JSON payload with keys `pixels`, `scalars`, `goal` and an optional `env` to override the skill.  Returns the selected action index.

Example request:

```bash
curl -X POST http://localhost:5000/step \
     -H 'Content-Type: application/json' \
     -d '{"pixels": [[[...]]], "env": "minecraft_agent"}'
```

## Running the GUI

The Tkinter GUI allows you to monitor memory pressure, emotion and workspace activations in real time.  Launch it with:

```bash
python -m capsule_brain.gui.app
```

## Training the Minecraft Skill

### Stage 1: Distill the CLIP teacher

Use the `train_distill.py` script to train a lightweight student encoder that mimics CLIP.  This requires the `minerl` package and will sample frames from the specified environment.

```bash
python -m capsule_brain.skills.minecraft_agent.teacher.train_distill \
       --env MineRLTreechop-v0 \
       --frames 5000 \
       --epochs 1 \
       --batch_size 32 \
       --lr 1e-4 \
       --output student_encoder.pth
```

### Stage 2: PPO fine‑tuning

After distillation, run the PPO trainer to optimise the actor–critic head using Stable‑Baselines 3.  This example trains for 100 000 timesteps and saves the trained policy.

```bash
python -m capsule_brain.skills.minecraft_agent.ppo.trainer \
       --env MineRLTreechop-v0 \
       --timesteps 100000 \
       --output minecraft_skill.zip
```

## Tests

Basic smoke tests are provided under `capsule_brain/tests`.  To run the tests using `pytest`:

```bash
pip install pytest
pytest -q capsule_brain/tests
```

Tests will import the package, perform a forward pass and, if MineRL is installed, verify that the environment adapter can reset and step correctly.

## Deployment

The `deployment` package contains a production wrapper (`brain_production.py`) and monitoring configuration for Prometheus and Grafana.  These files can be used to deploy the brain behind an inference server and collect metrics such as inference latency and memory pressure.  See the comments in those files for details.

---

© 2025 Capsule Brain Integration Project