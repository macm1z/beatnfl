"""Plain-English failures.

Every step of the weekly workflow raises StepError (never a raw traceback) with:
  what  - what went wrong, in one sentence
  why   - the most likely cause
  fix   - the exact thing to do next
The UI and CLI render these three lines; the technical detail is kept for the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class StepError(Exception):
    def __init__(self, step: str, what: str, why: str = "", fix: str = "", detail: str = ""):
        self.step, self.what, self.why, self.fix, self.detail = step, what, why, fix, detail
        super().__init__(f"[{step}] {what}")

    def as_text(self) -> str:
        lines = [f"✖ {self.step}: {self.what}"]
        if self.why:
            lines.append(f"  Why: {self.why}")
        if self.fix:
            lines.append(f"  Fix: {self.fix}")
        return "\n".join(lines)


@dataclass
class StepResult:
    step: str
    status: str  # "ok" | "warn" | "fail" | "skip"
    message: str
    fix: str = ""
    detail: str = ""
    data: dict = field(default_factory=dict)

    @property
    def icon(self) -> str:
        return {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}[self.status]

    def as_text(self) -> str:
        s = f"{self.icon} {self.step}: {self.message}"
        if self.fix:
            s += f"\n   → {self.fix}"
        return s
