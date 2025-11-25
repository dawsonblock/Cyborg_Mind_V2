"""A simple symbolic reasoning engine.

Symbolic reasoning complements the neural subsystems of the Capsule Brain
by allowing high‑level rules to influence behaviour.  The engine
maintains a set of boolean expressions and associated actions.  When
invoked with a context dictionary it evaluates each expression and
returns the first matching action.  Expressions are written in a
restricted Python subset and should be safe when used with trusted
rules.
"""

from typing import Any, Callable, Dict, List, Tuple


class SymbolicReasoner:
    """Rule‑based reasoning engine."""

    def __init__(self) -> None:
        # List of tuples (predicate_str, action_function)
        self.rules: List[Tuple[str, Callable[[Dict[str, Any]], Any]]] = []

    def add_rule(self, predicate: str, action: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a new rule.

        Args:
            predicate: A Python expression string evaluated in the context
                dictionary passed to ``infer``.  When the expression
                evaluates to ``True`` the associated action is executed.
            action: Function called with the context to compute the result.
        """
        self.rules.append((predicate, action))

    def infer(self, context: Dict[str, Any], default: Any = None) -> Any:
        """Evaluate rules against the context and return the result of the first match.

        Args:
            context: Dictionary of variables available to the predicates.
            default: Value returned if no predicates match.
        Returns:
            The result of the first matching action or ``default``.
        """
        for predicate, action in self.rules:
            try:
                if eval(predicate, {}, context):
                    return action(context)
            except Exception:
                continue
        return default
