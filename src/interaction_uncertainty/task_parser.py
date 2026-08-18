"""Task parsing boundary.

V0 uses a frozen schema parser for the two registered LIBERO retrieval prompts.
It is deliberately not advertised as learned language generalization.  A Qwen
backend can implement the same ``parse`` interface without changing planning.
"""

from __future__ import annotations

from .contracts import TaskSpec


class FrozenRetrievalTaskParser:
    model_stamp = "frozen-retrieval-schema-v0"

    def parse(self, prompt: str) -> TaskSpec:
        lowered = prompt.strip().lower()
        target = "cream cheese" if "cream cheese" in lowered else "butter" if "butter" in lowered else ""
        if not target or "basket" not in lowered:
            raise ValueError("V0 parser supports registered butter/cream-cheese-to-basket prompts only")
        return TaskSpec(
            prompt=prompt,
            target=target,
            destination="basket",
            goal_relation="inside",
            required_facts=(
                "target_identity",
                "target_location",
                "target_accessibility",
                "destination_location",
                "goal_completion",
            ),
            completion_description=f"{target} is inside basket",
        )
