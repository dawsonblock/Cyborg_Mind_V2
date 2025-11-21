# 🧠 Cyborg Mind v2

> **Hierarchical Reinforcement Learning for Minecraft with Emotion-Consciousness Architecture**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A novel AI system that trains agents to play Minecraft using a two-stage hierarchical learning approach with dynamic memory and recurrent thought processing.

---

## 🌟 Key Features

- **🎓 Two-Stage Learning**: Behavioral Cloning (BC) followed by Proximal Policy Optimization (PPO)
- **🧠 Emotion-Consciousness Brain**: Unified architecture with thought, emotion, and workspace memory
- **💾 Dynamic Memory**: Expandable memory system (256→2048 slots) with automatic garbage collection
- **🔄 Recurrent Processing**: LSTM-based temporal coherence with thought anchoring
- **🎮 20 Discrete Actions**: Comprehensive action space including combos and diagonal movements
- **📊 Full Observability**: TensorBoard integration for real-time training monitoring

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                   CYBORG MIND v2                        │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Phase 1: Teacher Learning (Behavioral Cloning)        │
│  ┌─────────────┐      ┌──────────────────────┐        │
│  │   MineRL    │─────▶│   RealTeacher        │        │
│  │   Dataset   │      │   87M parameters     │        │
│  └─────────────┘      └──────────────────────┘        │
│                                ↓                        │
│  Phase 2: Student Learning (PPO)                       │
│  ┌─────────────┐      ┌──────────────────────┐        │
│  │   MineRL    │◀────▶│  BrainCyborgMind     │        │
│  │     Env     │      │   2.3M parameters    │        │
│  └─────────────┘      └──────────────────────┘        │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### Models

#### RealTeacher (87M parameters)
- Frozen CLIP vision encoder
- Trainable action/value heads
- Trained via supervised learning on expert demonstrations

#### BrainCyborgMind (2.3M parameters)
- Vision adapter + Dynamic GPU PMM (memory)
- Recurrent thought loop (thought, emotion, workspace)
- LSTM core with multi-head outputs
- Trained via PPO with environment interaction

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required
- Python 3.9 or 3.10
- CUDA-capable GPU (recommended) or Apple Silicon
- 16GB+ RAM
- 50GB+ free disk space
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cyborg_mind_v2.git
cd cyborg_mind_v2

# Install dependencies
pip install -r requirements.txt

# Verify installation
python quick_verify.py
```

### For MineRL (Optional - See Installation Guide)

MineRL requires additional setup on macOS. See [`INSTALL.md`](INSTALL.md) for:
- Docker setup (recommended)
- Rosetta 2 method
- Cloud GPU alternatives

---

## 📖 Usage

### 1. Train RealTeacher (Phase 1)

```bash
# Download MineRL dataset (one-time, ~30GB)
python -c "import minerl; minerl.data.download('MineRLTreechop-v0', './data/minerl')"

# Train teacher model (~30-60 minutes on GPU)
export PYTHONPATH=$(pwd):$PYTHONPATH
python training/train_real_teacher_bc.py \
    --env-name MineRLTreechop-v0 \
    --data-dir ./data/minerl \
    --epochs 3 \
    --batch-size 64 \
    --lr 3e-4
```

**Expected Results:**
- Initial accuracy: ~5%
- Final accuracy: 70-80%
- Output: `checkpoints/teacher_best.pt`

### 2. Train BrainCyborgMind (Phase 2)

```bash
# Train brain with PPO (~2-4 hours on GPU)
python training/train_cyborg_mind_ppo.py
```

**Expected Results:**
- Episode rewards increase over time
- Agent learns to navigate and chop trees
- Output: `checkpoints/brain_best.pt`

### 3. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir runs

# Open browser: http://localhost:6006
```

---

## 📊 Results

| Metric | RealTeacher (BC) | BrainCyborgMind (PPO) |
|--------|------------------|------------------------|
| Training Time | 30-60 min | 2-4 hours |
| Parameters | 87.6M | 2.3M |
| Final Accuracy | 70-80% | N/A (RL metric) |
| Episode Reward | N/A | ~15+ (trees chopped) |

---

## 📁 Project Structure

```
cyborg_mind_v2/
├── capsule_brain/
│   └── policy/
│       └── brain_cyborg_mind.py    # Main brain model (2.3M params)
├── training/
│   ├── real_teacher.py             # Teacher model (87M params)
│   ├── train_real_teacher_bc.py    # BC training script
│   └── train_cyborg_mind_ppo.py    # PPO training script
├── envs/
│   ├── action_mapping.py           # 20 discrete actions
│   └── minerl_obs_adapter.py       # Observation preprocessing
├── data/
│   └── minerl/                     # MineRL dataset (~30GB)
├── checkpoints/                    # Saved model weights
├── runs/                           # TensorBoard logs
├── tests/                          # Unit and integration tests
├── docs/                           # Documentation
│   ├── COMPLETE_SYSTEM_GUIDE.md    # Full system explanation
│   ├── HOW_TO_TRAIN.md             # Training guide
│   ├── BUILD_STATUS.md             # Build readiness
│   └── ...                         # More guides
├── quick_verify.py                 # Setup verification
└── requirements.txt                # Python dependencies
```

---

## 🔧 Technical Details

### Action Space (20 Discrete Actions)

```python
0:  no-op              11: sprint_forward
1:  forward            12: sneak_forward
2:  back               13: place_block
3:  left               14: attack_forward
4:  right              15: jump_attack
5:  jump               16: sprint_attack
6:  attack             17: camera_up_right
7:  camera_right       18: camera_down_left
8:  camera_left        19: crouch
9:  camera_up
10: camera_down
```

### Memory System

- **Dynamic GPU PMM**: Cosine similarity-based retrieval
- **Expandable**: 256 → 2048 memory slots
- **Self-Modulated**: Workspace controls what to remember
- **Garbage Collection**: Automatic cleanup of stale memories

### Recurrent States

- **Thought** [32-dim]: Current mental state
- **Emotion** [8-dim]: Emotional context
- **Workspace** [64-dim]: Working memory
- **LSTM Hidden States**: Temporal coherence

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`COMPLETE_SYSTEM_GUIDE.md`](COMPLETE_SYSTEM_GUIDE.md) | Full system architecture and workflow (600+ lines) |
| [`HOW_TO_TRAIN.md`](HOW_TO_TRAIN.md) | Comprehensive training guide |
| [`BUILD_STATUS.md`](BUILD_STATUS.md) | Build readiness report |
| [`INSTALL.md`](INSTALL.md) | Installation instructions (Mac workarounds) |
| [`GYM_FIXED.md`](GYM_FIXED.md) | Solutions for Gym/MineRL installation |
| [`OPTIMIZATION_GUIDE.md`](training/OPTIMIZATION_GUIDE.md) | Performance tuning tips |
| [`DEBUG_SUMMARY.md`](training/DEBUG_SUMMARY.md) | All bugs fixed |

**Total Documentation:** 3,757+ lines

---

## 🧪 Testing

```bash
# Run quick verification
python quick_verify.py

# Run all tests
pytest tests/

# Run specific test
python tests/test_memory_expansion.py
```

---

## 🐛 Known Issues & Solutions

### MineRL on macOS (Apple Silicon)

**Issue:** MineRL requires Java 8, which doesn't support ARM architecture.

**Solutions:**
1. **Docker** (Recommended) - See `GYM_FIXED.md`
2. **Cloud GPU** - Google Colab, Lambda Labs
3. **Rosetta 2** - Run x86 Java via emulation

See [`INSTALL.md`](INSTALL.md) for detailed instructions.

---

## 🎯 Performance

### Benchmarks (NVIDIA RTX 3080)

- **BC Training**: ~30 minutes for 3 epochs
- **PPO Training**: ~2 hours for 1M steps
- **Inference**: ~100 FPS

### Memory Usage

- **Peak RAM**: ~8GB during training
- **VRAM**: ~6GB for batch size 64
- **Disk**: ~50GB total (including dataset)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{cyborg_mind_v2,
  title = {Cyborg Mind v2: Hierarchical RL for Minecraft},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/cyborg_mind_v2}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for CLIP vision encoder
- **MineRL** for dataset and environment
- **PyTorch** team for the framework
- **Hugging Face** for Transformers library

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/cyborg_mind_v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cyborg_mind_v2/discussions)

---

## 🔗 Links

- [MineRL Competition](https://minerl.io/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [PyTorch](https://pytorch.org/)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ for the AI and Gaming communities

</div>
