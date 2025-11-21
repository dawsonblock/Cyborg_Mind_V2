# How to Train Cyborg Mind v2 🚀

Complete step-by-step guide to training your Cyborg Mind agent from scratch to deployment.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Initial Setup](#initial-setup)
3. [Pre-Training Verification](#pre-training-verification)
4. [Training Phase 1: Behavioral Cloning](#training-phase-1-behavioral-cloning)
5. [Training Phase 2: PPO Reinforcement Learning](#training-phase-2-ppo-reinforcement-learning)
6. [Monitoring Training](#monitoring-training)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [Advanced Topics](#advanced-topics)

---

## 🖥️ System Requirements

### Minimum Requirements
- **GPU:** NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, RTX 2080)
- **RAM:** 16GB system memory
- **Storage:** 50GB free space (for MineRL dataset)
- **OS:** Linux or macOS (Windows via WSL2)

### Recommended Requirements
- **GPU:** NVIDIA RTX 3080 Ti or better (12GB+ VRAM)
- **RAM:** 32GB system memory
- **Storage:** 100GB+ SSD
- **CUDA:** 11.8 or 12.1

### Software Dependencies
```bash
# Core dependencies
Python 3.9+
PyTorch 2.0+
CUDA 11.8+ or 12.1
Java 8 (for MineRL)

# Install Java if needed (Ubuntu/Debian)
sudo apt-get install openjdk-8-jdk

# Install Java if needed (macOS)
brew install openjdk@8
```

---

## 🔧 Initial Setup

### Step 1: Install Python Dependencies

```bash
cd /Users/dawsonblock/Desktop/cyborg_mind_v2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers minerl gymnasium tensorboard numpy opencv-python

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

### Step 2: Download MineRL Dataset

**This is the most time-consuming step (~2-4 hours, ~30GB download):**

```bash
# Download MineRL TreeChop dataset
python -c "import minerl; minerl.data.download('MineRLTreechop-v0', './data/minerl')"

# Verify download
python -c "import minerl; data = minerl.data.make('MineRLTreechop-v0', './data/minerl'); print('Dataset ready!')"
```

**Note:** You can start with a smaller dataset for testing:
```bash
# Test with just a few trajectories
python -c "import minerl; minerl.data.download('MineRLTreechop-v0', './data/minerl', num_workers=1, resolution='low')"
```

### Step 3: Create Required Directories

```bash
# Create checkpoint and log directories
mkdir -p checkpoints/real_teacher
mkdir -p checkpoints/cyborg_mind
mkdir -p runs/real_teacher_bc
mkdir -p runs/cyborg_mind_ppo
mkdir -p logs
```

---

## ✅ Pre-Training Verification

**ALWAYS run this before starting training:**

```bash
python -m cyborg_mind_v2.training.verify_training_setup
```

**Expected Output:**
```
✓ Java installation found
✓ Python packages installed
✓ CUDA available
✓ MineRL environment works
✓ Action mapping verified
✓ Observation adapter works
✓ RealTeacher model loads
✓ BrainCyborgMind model loads
✓ All files present

🎉 Training setup is READY!
```

**If any checks fail, see [Troubleshooting](#troubleshooting).**

---

## 🎓 Training Phase 1: Behavioral Cloning

### What is Behavioral Cloning?

Behavioral cloning trains the RealTeacher model to imitate expert demonstrations from the MineRL dataset. This gives the agent a strong starting point before reinforcement learning.

### Quick Test Run (5 minutes)

```bash
# Test with minimal data to verify everything works
python -m cyborg_mind_v2.training.train_real_teacher_bc \
    --env-name MineRLTreechop-v0 \
    --data-dir ./data/minerl \
    --output-ckpt ./checkpoints/real_teacher/test_bc.pt \
    --epochs 1 \
    --batch-size 16 \
    --max-seq-len 16 \
    --device cuda \
    --log-dir runs/real_teacher_bc_test
```

**Expected Results:**
- Loss decreases from ~2.5 to ~2.0
- Accuracy increases to ~30-40%
- No crashes or CUDA OOM errors
- Checkpoint saved to `checkpoints/real_teacher/test_bc.pt`

### Full Training Run (~30-60 minutes)

```bash
python -m cyborg_mind_v2.training.train_real_teacher_bc \
    --env-name MineRLTreechop-v0 \
    --data-dir ./data/minerl \
    --output-ckpt ./checkpoints/real_teacher/bc_full.pt \
    --epochs 3 \
    --batch-size 64 \
    --max-seq-len 64 \
    --lr 3e-4 \
    --device cuda \
    --log-dir runs/real_teacher_bc
```

**Hyperparameters Explained:**
- `epochs`: Number of passes through dataset (3 is good start)
- `batch-size`: Samples per update (64 for 12GB GPU, 32 for 8GB)
- `max-seq-len`: Trajectory length (64 balances memory vs context)
- `lr`: Learning rate (3e-4 is stable for CLIP features)

**Expected Results:**
- **Loss:** Decreases to ~1.0-1.5
- **Accuracy:** Reaches 40-50%
- **Time:** ~10-20 min per epoch (without optimizations)

### Performance Tuning

For **3-5x speedup**, add these flags:

```bash
python -m cyborg_mind_v2.training.train_real_teacher_bc \
    --env-name MineRLTreechop-v0 \
    --data-dir ./data/minerl \
    --output-ckpt ./checkpoints/real_teacher/bc_optimized.pt \
    --epochs 3 \
    --batch-size 128 \
    --max-seq-len 64 \
    --lr 3e-4 \
    --device cuda \
    --use-amp \
    --num-workers 4 \
    --log-dir runs/real_teacher_bc_fast
```

**Key Optimizations:**
- `--use-amp`: Mixed precision training (2x speedup)
- `--num-workers 4`: Parallel data loading (CPU dependent)
- `--batch-size 128`: Larger batches (if GPU memory allows)

See `OPTIMIZATION_GUIDE.md` for more details.

---

## 🤖 Training Phase 2: PPO Reinforcement Learning

### What is PPO?

Proximal Policy Optimization (PPO) trains the BrainCyborgMind agent through trial-and-error in the actual Minecraft environment. This improves upon the BC-pretrained model.

### Important: Edit Configuration First

Open `training/train_cyborg_mind_ppo.py` and modify `PPOConfig`:

```python
@dataclass
class PPOConfig:
    # For quick test (10-15 minutes)
    total_steps: int = 10_000
    steps_per_update: int = 512
    
    # For full training (2-4 hours)
    # total_steps: int = 500_000
    # steps_per_update: int = 2048
    
    # ... rest of config
```

### Quick Test Run (10-15 minutes)

```bash
# Make sure PPOConfig.total_steps = 10_000
python -m cyborg_mind_v2.training.train_cyborg_mind_ppo \
    --env-name MineRLTreechop-v0 \
    --checkpoint-path ./checkpoints/real_teacher/bc_full.pt \
    --output-dir ./checkpoints/cyborg_mind \
    --log-dir runs/cyborg_mind_ppo_test
```

**Expected Results:**
- Agent collects experience and performs PPO updates
- Episode rewards are initially negative (sparse rewards)
- Value loss and policy loss decrease over time
- Entropy decreases (agent becomes more confident)

### Full Training Run (2-4 hours)

```bash
# Set PPOConfig.total_steps = 500_000 in the code
python -m cyborg_mind_v2.training.train_cyborg_mind_ppo \
    --env-name MineRLTreechop-v0 \
    --checkpoint-path ./checkpoints/real_teacher/bc_full.pt \
    --output-dir ./checkpoints/cyborg_mind \
    --log-dir runs/cyborg_mind_ppo \
    --save-interval 50000
```

**Training Timeline:**
- **0-50k steps:** Agent explores, learns basic controls
- **50k-200k steps:** Agent starts approaching trees
- **200k-500k steps:** Agent learns to chop trees consistently
- **500k+ steps:** Fine-tuning and optimization

**Key Metrics to Watch:**
- `reward/mean`: Should increase over time
- `reward/episode_length`: Should stabilize
- `loss/policy`: Should decrease
- `loss/value`: Should decrease then stabilize
- `metrics/entropy`: Should decrease gradually

### PPO Hyperparameters

**Edit these in `train_cyborg_mind_ppo.py` for better performance:**

```python
@dataclass
class PPOConfig:
    # Environment
    env_name: str = "MineRLTreechop-v0"
    
    # Training steps
    total_steps: int = 500_000       # Total env steps
    steps_per_update: int = 2048      # Rollout buffer size
    
    # PPO parameters
    gamma: float = 0.99               # Discount factor
    gae_lambda: float = 0.95          # GAE parameter
    clip_ratio: float = 0.2           # PPO clip range
    
    # Optimization
    ppo_epochs: int = 4               # Update epochs per rollout
    minibatch_size: int = 256         # Minibatch size
    lr: float = 3e-4                  # Learning rate
    max_grad_norm: float = 0.5        # Gradient clipping
    
    # Neural network
    num_actions: int = 20             # Action space size
    
    # Logging
    log_interval: int = 10            # Steps between logs
    save_interval: int = 50_000       # Steps between checkpoints
```

**Tuning Tips:**
- **Higher `gamma` (0.99):** For long-term rewards (tree chopping)
- **Higher `clip_ratio` (0.3):** For more exploration
- **Lower `lr` (1e-4):** For stable fine-tuning
- **More `ppo_epochs` (10):** For better value learning (slower)

---

## 📊 Monitoring Training

### TensorBoard

**Start TensorBoard to visualize training:**

```bash
tensorboard --logdir runs --port 6006
```

**Open in browser:** `http://localhost:6006`

### Key Metrics

#### Behavioral Cloning (BC)
- **`loss/train_ce`:** Cross-entropy loss (should decrease to ~1.0)
- **`train/accuracy`:** Classification accuracy (should reach 40-50%)
- **`train/lr`:** Learning rate (decreases with schedule)
- **`action_prob/idx_*`:** Action distribution (should match dataset)

#### PPO Training
- **`reward/mean`:** Average episode reward (should increase)
- **`reward/episode_length`:** Steps per episode (should stabilize)
- **`loss/policy`:** Policy loss (should decrease)
- **`loss/value`:** Value loss (should decrease then stabilize)
- **`loss/total`:** Combined loss
- **`metrics/entropy`:** Policy entropy (should decrease gradually)
- **`metrics/approx_kl`:** KL divergence (should stay < 0.02)
- **`metrics/clip_fraction`:** % clipped updates (should be 10-30%)

### Console Output

**BC Training:**
```
[BC] Epoch 1/3
[BC] Processed 1000 samples, loss=2.134, lr=0.000300
[BC] Processed 2000 samples, loss=1.876, lr=0.000299
...
[BC] Checkpoint saved to checkpoints/real_teacher/bc_full.pt
```

**PPO Training:**
```
[PPO] Step 512/500000 | Reward: -0.5 | Len: 1024 | VLoss: 0.234 | PLoss: 0.045
[PPO] Step 1024/500000 | Reward: -0.3 | Len: 982 | VLoss: 0.198 | PLoss: 0.038
...
[PPO] Checkpoint saved: checkpoints/cyborg_mind/brain_step_50000.pt
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
```bash
# Reduce batch size
--batch-size 32  # or 16

# Reduce sequence length
--max-seq-len 32  # or 16

# Use gradient accumulation (if implemented)
--gradient-accumulation-steps 2
```

#### 2. **Java Not Found**

**Error:**
```
Java is not installed or not in PATH
```

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt-get install openjdk-8-jdk
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# macOS
brew install openjdk@8
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
```

#### 3. **MineRL Dataset Missing**

**Error:**
```
FileNotFoundError: Dataset not found
```

**Solution:**
```bash
# Download dataset
python -c "import minerl; minerl.data.download('MineRLTreechop-v0', './data/minerl')"
```

#### 4. **Import Errors**

**Error:**
```
ModuleNotFoundError: No module named 'cyborg_mind_v2'
```

**Solution:**
```bash
# Install package in development mode
cd /Users/dawsonblock/Desktop/cyborg_mind_v2
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/Users/dawsonblock/Desktop/cyborg_mind_v2:$PYTHONPATH
```

#### 5. **Slow Training**

**If training is very slow (<100 samples/sec for BC or <10 steps/sec for PPO):**

1. **Check GPU utilization:**
```bash
nvidia-smi -l 1  # Monitor GPU usage
```

2. **Apply optimizations from `OPTIMIZATION_GUIDE.md`:**
   - Enable AMP (mixed precision)
   - Increase batch size
   - Use DataLoader with multiple workers
   - Enable cudnn benchmarking

3. **Profile bottlenecks:**
```python
# Add to training script
import torch.profiler as profiler

with profiler.profile() as prof:
    # Training code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

#### 6. **NaN Losses**

**Error:**
```
Loss is NaN!
```

**Solutions:**
- Lower learning rate: `--lr 1e-4`
- Add gradient clipping: `--max-grad-norm 0.5`
- Check for invalid actions or observations
- Reduce PPO clip ratio: `clip_ratio=0.1`

#### 7. **Agent Not Learning (PPO)**

**Symptoms:** Reward stays flat after 100k steps

**Solutions:**
1. **Check reward shaping:**
```python
# Add intrinsic rewards
shaped_reward = reward + 0.01 * small_progress_signal
```

2. **Verify action space:**
```bash
# Test action mapping
python -c "from cyborg_mind_v2.envs.action_mapping import *; \
    for i in range(20): print(i, index_to_minerl_action(i))"
```

3. **Increase exploration:**
```python
# In PPOConfig
entropy_coef: float = 0.01  # Default
# Try: entropy_coef: float = 0.05  # More exploration
```

4. **Use BC pretrained checkpoint:**
```bash
--checkpoint-path ./checkpoints/real_teacher/bc_full.pt
```

---

## ⚡ Performance Optimization

### Quick Wins (Apply These First)

**1. Enable Mixed Precision (AMP)**
```python
# In train_real_teacher_bc.py (if not already enabled)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits, _ = teacher.predict(pixels, scalars)
    loss = criterion(logits, actions)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Expected Speedup:** 2x faster, 40% less memory

**2. Enable cudnn Benchmarking**
```python
# Add at start of training script
import torch
torch.backends.cudnn.benchmark = True
```

**Expected Speedup:** 10-20% faster

**3. Increase Batch Size**
```bash
# If you have GPU memory to spare
--batch-size 128  # BC training (was 64)
```

**Expected Speedup:** 30-50% faster

### Advanced Optimizations

See `OPTIMIZATION_GUIDE.md` for:
- DataLoader optimization
- Gradient accumulation
- Model quantization
- Distributed training
- Dataset caching

**Expected Total Speedup:** 3-5x faster than baseline

---

## 🎯 Advanced Topics

### Curriculum Learning

Train on progressively harder tasks:

```python
# Stage 1: Learn to move (50k steps)
reward = 0.01 * forward_movement

# Stage 2: Learn to approach trees (100k steps)  
reward = 0.1 * distance_to_tree_decreased

# Stage 3: Learn to chop (full training)
reward = actual_game_reward
```

### Hyperparameter Tuning

Use Optuna or Ray Tune:

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # Train and return validation loss
    loss = train_with_params(lr, batch_size)
    return loss

study = optuna.create_study()
study.optimize(objective, n_trials=20)
```

### Multi-Task Training

Train on multiple MineRL environments:

```python
envs = [
    'MineRLTreechop-v0',
    'MineRLNavigate-v0',
    'MineRLObtainDiamond-v0'
]

for env_name in envs:
    train_on_env(env_name)
```

### Evaluation

```python
# Evaluate trained model
from cyborg_mind_v2.training.evaluate import evaluate_agent

results = evaluate_agent(
    checkpoint_path='checkpoints/cyborg_mind/brain_step_500000.pt',
    num_episodes=100,
    render=True
)

print(f"Success rate: {results['success_rate']:.2%}")
print(f"Average reward: {results['avg_reward']:.2f}")
```

---

## 📚 Additional Resources

### Documentation
- **`DEBUG_SUMMARY.md`** - All bugs fixed and testing checklist
- **`OPTIMIZATION_GUIDE.md`** - Detailed performance tuning
- **`FIXES_SUMMARY.md`** - Complete error reference
- **`QUICK_START.md`** - 5-minute quick start
- **`TRAINING_README.md`** - Architecture overview

### External Links
- [MineRL Documentation](https://minerl.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [PyTorch Docs](https://pytorch.org/docs/)

---

## ✅ Training Checklist

Use this checklist for each training run:

### Pre-Training
- [ ] Java 8 installed and in PATH
- [ ] Python dependencies installed
- [ ] MineRL dataset downloaded
- [ ] GPU drivers and CUDA working
- [ ] `verify_training_setup.py` passes all checks
- [ ] Directories created (`checkpoints/`, `runs/`)

### BC Training
- [ ] Quick test run completed (5 min)
- [ ] Full BC training started
- [ ] TensorBoard monitoring running
- [ ] Loss decreasing steadily
- [ ] Checkpoint saved successfully
- [ ] Accuracy reaches 40%+

### PPO Training  
- [ ] BC checkpoint exists
- [ ] PPO config reviewed and adjusted
- [ ] Quick test run completed (10 min)
- [ ] Full PPO training started
- [ ] Agent interacting with environment
- [ ] Rewards increasing over time
- [ ] Checkpoints saving every 50k steps

### Post-Training
- [ ] Final checkpoint saved
- [ ] TensorBoard logs archived
- [ ] Performance metrics recorded
- [ ] Model evaluated on test episodes
- [ ] Results documented

---

## 🎉 Expected Timeline

### Full Training from Scratch

| Phase | Duration | GPU | Output |
|-------|----------|-----|--------|
| Setup | 2-4 hours | N/A | Dataset downloaded |
| BC Test | 5 min | Yes | Verified working |
| BC Full | 30-60 min | Yes | `bc_full.pt` |
| PPO Test | 10-15 min | Yes | Verified working |
| PPO Full | 2-4 hours | Yes | `brain_step_500000.pt` |
| **Total** | **~3-6 hours** | | **Ready agent** |

### With Optimizations Applied

| Phase | Duration | Speedup |
|-------|----------|---------|
| BC Training | 10-15 min | 3-4x |
| PPO Training | 45-90 min | 2-3x |
| **Total** | **~1-2 hours** | **3x** |

---

## 🚀 Quick Start Commands

**Copy-paste these for a full training run:**

```bash
# 1. Verify setup
python -m cyborg_mind_v2.training.verify_training_setup

# 2. Train BC (30-60 min)
python -m cyborg_mind_v2.training.train_real_teacher_bc \
    --env-name MineRLTreechop-v0 \
    --data-dir ./data/minerl \
    --output-ckpt ./checkpoints/real_teacher/bc_full.pt \
    --epochs 3 \
    --batch-size 64 \
    --device cuda

# 3. Train PPO (2-4 hours, edit config first)
python -m cyborg_mind_v2.training.train_cyborg_mind_ppo \
    --env-name MineRLTreechop-v0 \
    --checkpoint-path ./checkpoints/real_teacher/bc_full.pt \
    --output-dir ./checkpoints/cyborg_mind

# 4. Monitor with TensorBoard
tensorboard --logdir runs
```

**That's it! Happy training! 🎮🤖**

---

## 📞 Support

If you encounter issues not covered in this guide:

1. Check `DEBUG_SUMMARY.md` for known issues
2. Check `FIXES_SUMMARY.md` for error solutions
3. Review TensorBoard logs for anomalies
4. Create detailed bug report using template in `DEBUG_SUMMARY.md`

**Last Updated:** November 2024  
**Version:** 2.0
