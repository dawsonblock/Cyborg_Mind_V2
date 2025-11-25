"""External AI tutor bridge.

This module defines a simple interface to query an external large
language model (LLM) such as OpenAI's GPT or Anthropic's Sonnet.  The
Capsule Brain uses the tutor as part of its curriculum to
provide explanations, hints or additional information during
training.  By decoupling the LLM interface into its own class the
system can be configured to use different providers or run offline.
"""

from __future__ import annotations

from typing import Optional

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore


class TutorBridge:
    """Query an external language model for assistance.

    Usage:
        bridge = TutorBridge(model="gpt-4", api_key="sk-…")
        answer = bridge.ask("Explain the concept of capsules in neural networks.")

    If the ``openai`` package is not installed or an API key is not
    provided the bridge returns a placeholder response.
    """

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key
        if openai is not None and api_key is not None:
            openai.api_key = api_key

    def ask(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        """Send a prompt to the tutor and return the generated response.

        Args:
            prompt: The textual prompt or question.
            temperature: Sampling temperature controlling randomness.
            max_tokens: Maximum number of tokens in the reply.

        Returns:
            The tutor's response as a string.  If the API is not
            available a placeholder explanation is returned.
        """
        if openai is None or self.api_key is None:
            # No API; return a simple placeholder response
            return (
                "[Tutor unavailable] The tutor bridge is not configured with an API key."
                " Please install the 'openai' package and set your API key to use this feature."
            )
        # Otherwise call the completion API
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful tutor."}, {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"[Tutor error] {e}"