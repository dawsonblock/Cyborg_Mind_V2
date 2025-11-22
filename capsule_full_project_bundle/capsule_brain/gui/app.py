"""Simple Tkinter GUI for monitoring the Capsule Brain.

The GUI displays the current memory pressure, emotion vector and
workspace activations.  It updates these values periodically by
querying the underlying brain.
"""

import threading
import tkinter as tk
from tkinter import ttk

import torch

from ..core.brain import CapsuleBrain
from ..core.config import CapsuleBrainConfig


class BrainMonitor(tk.Frame):
    def __init__(self, master: tk.Tk, brain: CapsuleBrain, update_interval: float = 1.0) -> None:
        super().__init__(master)
        self.brain = brain
        self.update_interval = update_interval
        self.pack(fill="both", expand=True)
        # Memory pressure label
        self.pressure_var = tk.StringVar()
        ttk.Label(self, textvariable=self.pressure_var).pack(anchor="w")
        # Emotion labels
        self.emotion_vars = [tk.StringVar() for _ in range(self.brain.cfg.emotion.num_channels)]
        for i, var in enumerate(self.emotion_vars):
            ttk.Label(self, textvariable=var).pack(anchor="w")
        # Workspace listbox
        self.workspace_list = tk.Listbox(self)
        self.workspace_list.pack(fill="both", expand=True)
        # Skill probabilities listbox
        ttk.Label(self, text="Skill Selection").pack(anchor="w")
        self.skill_list = tk.Listbox(self)
        self.skill_list.pack(fill="both", expand=True)
        # Memory usage listbox
        ttk.Label(self, text="Top Memory Slots").pack(anchor="w")
        self.mem_usage_list = tk.Listbox(self)
        self.mem_usage_list.pack(fill="both", expand=True)
        # Start update loop
        self.after(int(self.update_interval * 1000), self.update)

    def update(self) -> None:
        # Retrieve average state from the brain
        pressure = self.brain.pmm.compute_pressure()
        self.pressure_var.set(f"Memory Pressure: {pressure:.3f}")
        # For demonstration we use a zero batch of appropriate size (3×64×64)
        device = next(self.brain.parameters()).device
        B = 1
        pixels = torch.zeros(B, 3, 64, 64, device=device)
        out = self.brain(pixels, env=None, prev_state=None)
        emotion = out["emotion"].mean(dim=0).cpu().numpy()
        for i, var in enumerate(self.emotion_vars):
            var.set(f"Emotion[{i}]: {emotion[i]:.3f}")
        workspace = out["workspace"].mean(dim=0).cpu().numpy()
        self.workspace_list.delete(0, tk.END)
        for i, val in enumerate(workspace):
            self.workspace_list.insert(tk.END, f"{i}: {val:.3f}")
        # Update skill probabilities
        with torch.no_grad():
            # Softmax over action capsule logits (without meta controller and safety for monitoring)
            logits = self.brain.action_capsule(out["workspace"])
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        self.skill_list.delete(0, tk.END)
        for name, prob in zip(self.brain.skills.keys(), probs):
            self.skill_list.insert(tk.END, f"{name}: {prob:.2f}")
        # Show top memory slots by usage
        usage = self.brain.pmm.usage.cpu().numpy()  # type: ignore
        # Get indices of top 10 used slots
        top_idx = usage.argsort()[::-1][:10]
        self.mem_usage_list.delete(0, tk.END)
        for idx in top_idx:
            self.mem_usage_list.insert(tk.END, f"Slot {idx}: {usage[idx]:.3f}")
        self.after(int(self.update_interval * 1000), self.update)


def main() -> None:
    brain = CapsuleBrain(CapsuleBrainConfig())
    root = tk.Tk()
    root.title("Capsule Brain Monitor")
    BrainMonitor(root, brain)
    root.mainloop()


if __name__ == "__main__":
    main()
