"""
review/verdict.py — the structured output schema for Part 4's AutoGen
review stage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VerdictModel(BaseModel):
    approved: bool = Field(..., description="Whether the reviewed response is safe to send to the customer as-is.")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Overall risk level assigned by the review stage.")
    concerns: list[str] = Field(default_factory=list, description="Specific concerns found, if any.")
    rationale: str = Field(..., description="Human-readable explanation of the verdict.")
