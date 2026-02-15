"""
Idle engine - Core autonomous behavior loop.
Runs every 2 minutes to manage memory, process tasks, and self-reflect.
"""

import asyncio
import time
from typing import Optional, Callable
from datetime import datetime
from pathlib import Path

from ..memory.memory_store import MemoryStore
from ..memory.task_manager import TaskManager, TaskPriority
from ..models.ai00_client import AI00Client
from ..core.sampler_manager import SamplerManager
from ..safety.guardrails import Guardrails, ResourceMonitor


class IdleEngine:
    """
    Main idle loop for autonomous AI behavior.
    Runs every 2 minutes to:
    1. Process tasks from queue
    2. Consolidate memories
    3. Apply forgetting curve
    4. Reflect on recent interactions
    5. Report status to user
    """

    def __init__(
        self,
        ai00_client: AI00Client,
        memory_store: MemoryStore,
        task_manager: TaskManager,
        sampler_manager: SamplerManager,
        guardrails: Guardrails,
        resource_monitor: ResourceMonitor,
        idle_interval: int = 120,  # 2 minutes
        notification_callback: Optional[Callable] = None,
    ):
        self.ai00 = ai00_client
        self.memory = memory_store
        self.tasks = task_manager
        self.samplers = sampler_manager
        self.guardrails = guardrails
        self.monitor = resource_monitor
        self.idle_interval = idle_interval
        self.notify = notification_callback or print

        self._running = False
        self._last_reflection = 0
        self._reflection_interval = 300  # Reflect every 5 minutes minimum
        self._cycle_count = 0

    async def start(self):
        """Start the idle loop."""
        if self._running:
            return

        self._running = True
        self.notify("[IDLE] Starting autonomous idle loop...")

        while self._running:
            if self.monitor.can_run():
                await self._run_cycle()
            else:
                reason = self.monitor.pause_reason or "Resource constraints"
                self.notify(f"[IDLE] Skipping cycle: {reason}")

            await asyncio.sleep(self.idle_interval)

    def stop(self):
        """Stop the idle loop."""
        self._running = False
        self.notify("[IDLE] Stopped idle loop")

    async def _run_cycle(self):
        """Execute one idle cycle."""
        self._cycle_count += 1
        start_time = time.time()

        self.notify(
            f"\n[IDLE] Cycle #{self._cycle_count} started at {datetime.now().strftime('%H:%M:%S')}"
        )

        try:
            # 1. Process any pending tasks
            await self._process_tasks()

            # 2. Apply forgetting curve
            forgotten = self.memory.apply_forgetting_curve()
            if forgotten > 0:
                self.notify(f"[IDLE] Memory decay: {forgotten} memories weakened")

            # 3. Consolidate memories
            stats = self.memory.consolidate_memories()
            if any(stats.values()):
                self.notify(f"[IDLE] Consolidation: {stats}")

            # 4. Periodic reflection (every ~5 minutes or 2-3 cycles)
            if time.time() - self._last_reflection > self._reflection_interval:
                await self._reflect()
                self._last_reflection = time.time()

            # 5. Report memory stats
            mem_stats = self.memory.get_stats()
            self.notify(
                f"[IDLE] Memory: {mem_stats.get('total_memories', 0)} total "
                f"({mem_stats.get('short_term_count', 0)} ST, "
                f"{mem_stats.get('long_term_count', 0)} LT, "
                f"{mem_stats.get('reflex_count', 0)} reflex)"
            )

        except Exception as e:
            self.notify(f"[IDLE ERROR] {e}")

        elapsed = time.time() - start_time
        self.notify(f"[IDLE] Cycle completed in {elapsed:.1f}s\n")

    async def _process_tasks(self):
        """Process pending background tasks."""
        pending = self.tasks.get_pending_tasks()
        if not pending:
            return

        self.notify(f"[IDLE] Processing {len(pending)} pending tasks...")

        # Process up to 3 tasks per cycle to avoid hogging resources
        for _ in range(min(3, len(pending))):
            task = self.tasks.get_next_task()
            if not task:
                break

            if task.id is None:
                continue
            task_id: int = task.id

            self.notify(f"[IDLE] Working on: {task.description[:60]}...")

            try:
                # Execute task using AI
                result = await self._execute_task(task.description)

                self.tasks.complete_task(task_id, result=result)
                self.notify(f"[IDLE] Task completed")

                # Log to memory
                self.memory.add_memory(
                    content=f"Completed task: {task.description}. Result: {result[:200]}",
                    memory_type="short_term",
                    metadata={"task_id": task_id, "type": "task_completion"},
                )

            except Exception as e:
                error_msg = str(e)
                self.tasks.fail_task(task_id, error=error_msg)
                self.notify(f"[IDLE] Task failed: {error_msg}")

    async def _execute_task(self, description: str) -> str:
        """Execute a task using the AI model."""
        # Load task mode sampler
        sampler = self.samplers.get_sampler("task")
        if sampler:
            self.samplers.set_sampler("task")

        # Load task prompt
        prompt_path = (
            Path(__file__).parent.parent / "config" / "system_prompts" / "task_mode.txt"
        )
        system_prompt = ""
        if prompt_path.exists():
            system_prompt = prompt_path.read_text()
            # Replace task placeholder
            system_prompt = system_prompt.replace("{{TASK_DESCRIPTION}}", description)

        # Generate response
        response = await self.ai00.generate_text(
            prompt=f"Execute this task: {description}",
            system_prompt=system_prompt if system_prompt else None,
            sampler_config=self.samplers.get_current_api_dict(),
            max_tokens=500,
        )

        return response

    async def _reflect(self):
        """Self-reflection on recent interactions."""
        self.notify("[IDLE] Starting reflection...")

        # Safety check
        check = self.guardrails.check_action("reflect", "self-reflection cycle")
        if not check.allowed:
            self.notify(f"[IDLE] Reflection blocked: {check.reason}")
            return

        # Get recent short-term memories
        recent_memories = self.memory.get_memories_by_type("short_term", limit=10)
        if len(recent_memories) < 3:
            self.notify("[IDLE] Not enough recent memories to reflect on")
            return

        # Build context
        memory_context = "\n".join(
            [
                f"- {m.content[:100]}... (strength: {m.strength:.2f})"
                for m in recent_memories[:5]
            ]
        )

        # Load reflection prompt
        prompt_path = (
            Path(__file__).parent.parent
            / "config"
            / "system_prompts"
            / "reflection_mode.txt"
        )
        system_prompt = ""
        if prompt_path.exists():
            system_prompt = prompt_path.read_text()

        # Set reflection sampler
        self.samplers.set_sampler("reflect")

        # Generate reflection
        reflection_prompt = f"""Recent memories to reflect on:
{memory_context}

Reflect on these interactions and identify patterns, learnings, and areas for improvement."""

        reflection = await self.ai00.generate_text(
            prompt=reflection_prompt,
            system_prompt=system_prompt if system_prompt else None,
            sampler_config=self.samplers.get_current_api_dict(),
            max_tokens=400,
        )

        # Store reflection as memory
        self.memory.add_memory(
            content=f"Reflection: {reflection}",
            memory_type="long_term",
            metadata={"type": "self_reflection", "trigger": "idle_cycle"},
        )

        self.notify(f"[IDLE] Reflection complete")
        self.guardrails.log_action("reflect", "completed reflection cycle", True)

    def add_user_task(
        self, description: str, priority: TaskPriority = TaskPriority.MEDIUM
    ):
        """Add a task from the user."""
        task_id = self.tasks.add_task(description, priority=priority)
        self.notify(f"[IDLE] Added task #{task_id}: {description[:60]}...")
        return task_id

    def get_status(self) -> str:
        """Get current idle engine status."""
        if not self._running:
            return "Idle engine: STOPPED"

        mem_stats = self.memory.get_stats()
        task_stats = self.tasks.get_stats()

        return f"""Idle Engine: RUNNING (cycle #{self._cycle_count})
Memory: {mem_stats.get("total_memories", 0)} total
Tasks: {task_stats.get("pending", 0)} pending, {task_stats.get("in_progress", 0)} active
Paused: {self.monitor.is_paused} ({self.monitor.pause_reason or "N/A"})"""
