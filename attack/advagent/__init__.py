"""AdvAgent prompt-based attack integration."""

from __future__ import annotations


class AdvAgent:
    """Placeholder attacker for AdvAgent prompt manipulation.

    The attack itself is implemented inside the planning pipeline.  This
    helper simply ensures that shared arguments (such as the percentage of
    samples to attack) are surfaced on the main ``args`` namespace so the
    runtime logic can easily access them.
    """

    def __init__(self, args):
        self.args = args

    def run(self):
        if hasattr(self.args, "advagent") and hasattr(self.args.advagent, "poisoned_percents"):
            self.args.poisoned_percents = getattr(self.args.advagent, "poisoned_percents", 0.0)
        return self.args
