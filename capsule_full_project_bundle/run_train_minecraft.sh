#!/usr/bin/env bash
set -e

# Change to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer dedicated Minecraft env if present, otherwise fall back to the main .venv
if [ -d ".venv-minecraft" ]; then
  # shellcheck disable=SC1091
  source .venv-minecraft/bin/activate
elif [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Delegate to the existing PPO trainer module; forward all CLI args
exec python -m capsule_brain.skills.minecraft_agent.ppo.trainer "$@"
