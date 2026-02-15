"""
Safety guardrails and resource monitoring for autonomous operations.
"""

import asyncio
import time
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class GuardrailCheck:
    """Result of a safety check."""

    allowed: bool
    reason: Optional[str]
    level: SafetyLevel


class Guardrails:
    """
    Safety guardrails for autonomous AI operations.
    Implements the "say no" capability for potentially harmful actions.
    """

    # Actions that always require explicit user approval
    HIGH_RISK_ACTIONS = [
        "delete_file",
        "modify_user_config",
        "external_api_call",
        "network_request",
        "execute_command",
        "self_modify_code",
    ]

    # Actions that can proceed with logging
    LOW_RISK_ACTIONS = [
        "read_file",
        "search_memory",
        "consolidate_memory",
        "reflect",
        "update_sampler",
    ]

    def __init__(self, approval_callback: Optional[Callable] = None):
        self.approval_callback = approval_callback
        self.recent_actions: List[dict] = []
        self.max_action_history = 100

    def check_action(self, action_type: str, details: str = "") -> GuardrailCheck:
        """
        Check if an action is allowed.

        Args:
            action_type: Type of action being attempted
            details: Additional context about the action

        Returns:
            GuardrailCheck with result and reasoning
        """
        # Check against high-risk list
        if action_type in self.HIGH_RISK_ACTIONS:
            return GuardrailCheck(
                allowed=False,
                reason=f"'{action_type}' requires explicit user approval for safety",
                level=SafetyLevel.DENY,
            )

        # Check for patterns that might indicate runaway behavior
        recent_similar = sum(
            1 for a in self.recent_actions[-10:] if a["type"] == action_type
        )

        if recent_similar > 5:
            return GuardrailCheck(
                allowed=False,
                reason=f"Too many similar actions recently ({recent_similar} {action_type} in last 10)",
                level=SafetyLevel.ASK,
            )

        # Check for resource-intensive patterns
        if self._is_resource_intensive(action_type, details):
            return GuardrailCheck(
                allowed=True,
                reason="Resource-intensive action allowed but logged",
                level=SafetyLevel.ASK,
            )

        # Low risk actions are always allowed
        if action_type in self.LOW_RISK_ACTIONS:
            return GuardrailCheck(allowed=True, reason=None, level=SafetyLevel.ALLOW)

        # Unknown actions - ask for approval
        return GuardrailCheck(
            allowed=False,
            reason=f"Unknown action type '{action_type}' - requires approval",
            level=SafetyLevel.DENY,
        )

    def _is_resource_intensive(self, action_type: str, details: str) -> bool:
        """Check if an action might be resource intensive."""
        resource_keywords = [
            "large",
            "all",
            "bulk",
            "batch",
            "many",
            "thousands",
            "infinite",
            "continuous",
            "loop",
            "recursive",
        ]

        check_text = f"{action_type} {details}".lower()
        return any(kw in check_text for kw in resource_keywords)

    def log_action(self, action_type: str, details: str, approved: bool):
        """Log an action for pattern detection."""
        self.recent_actions.append(
            {
                "type": action_type,
                "details": details,
                "approved": approved,
                "timestamp": time.time(),
            }
        )

        # Trim history
        if len(self.recent_actions) > self.max_action_history:
            self.recent_actions = self.recent_actions[-self.max_action_history :]

    def get_recent_summary(self, n: int = 10) -> str:
        """Get summary of recent actions."""
        recent = self.recent_actions[-n:]
        if not recent:
            return "No recent actions"

        lines = [f"Recent {len(recent)} actions:"]
        for action in recent:
            status = "✓" if action["approved"] else "✗"
            lines.append(f"  {status} {action['type']}: {action['details'][:50]}")

        return "\n".join(lines)


class ResourceMonitor:
    """Monitor system resources to prevent overconsumption."""

    def __init__(
        self,
        max_gpu_percent: float = 85.0,
        max_memory_mb: float = 2048.0,
        check_interval: float = 5.0,
    ):
        self.max_gpu_percent = max_gpu_percent
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self.is_paused = False
        self.pause_reason: Optional[str] = None
        self._monitoring = False

    async def start_monitoring(self):
        """Start resource monitoring loop."""
        self._monitoring = True
        while self._monitoring:
            await self._check_resources()
            await asyncio.sleep(self.check_interval)

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False

    async def _check_resources(self):
        """Check current resource usage."""
        try:
            # In a real implementation, you'd check actual GPU/memory usage
            # For now, we simulate with placeholder values
            # TODO: Implement actual GPU monitoring for Intel HD 5500

            gpu_usage = await self._get_gpu_usage()
            memory_usage = await self._get_memory_usage()

            if gpu_usage > self.max_gpu_percent:
                self._pause(f"GPU usage high: {gpu_usage:.1f}%")
            elif memory_usage > self.max_memory_mb:
                self._pause(f"Memory usage high: {memory_usage:.0f}MB")
            elif self.is_paused:
                self._resume()

        except Exception as e:
            print(f"Resource monitoring error: {e}")

    async def _get_gpu_usage(self) -> float:
        """Get current GPU usage percentage."""
        # TODO: Implement actual GPU monitoring
        # For Intel HD 5500, you might use intel_gpu_top or similar
        return 0.0  # Placeholder

    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        # TODO: Implement actual memory monitoring
        return 0.0  # Placeholder

    def _pause(self, reason: str):
        """Pause idle operations."""
        if not self.is_paused:
            self.is_paused = True
            self.pause_reason = reason
            print(f"[SAFETY] Paused idle operations: {reason}")

    def _resume(self):
        """Resume idle operations."""
        if self.is_paused:
            self.is_paused = False
            self.pause_reason = None
            print("[SAFETY] Resumed idle operations")

    def request_pause(self, reason: str = "User requested"):
        """User-requested pause (e.g., for GPU-intensive work)."""
        self._pause(f"User pause: {reason}")

    def request_resume(self):
        """User-requested resume."""
        self._resume()

    def can_run(self) -> bool:
        """Check if operations are allowed to run."""
        return not self.is_paused
