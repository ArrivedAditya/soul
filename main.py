#!/usr/bin/env python3
"""
Soul - A personal AI companion with autonomous self-reflection and memory.

Usage:
    python main.py                          # Start interactive mode
    python main.py --idle                   # Start with idle loop
    python main.py --status                 # Show current status
    python main.py --pause                  # Pause idle operations
    python main.py --resume                 # Resume idle operations
    python main.py task "description"       # Add a background task
"""

import asyncio
import argparse
from typing import Optional
from pathlib import Path

from soul.models.ai00_client import AI00Client
from soul.memory.memory_store import MemoryStore
from soul.memory.task_manager import TaskManager, TaskPriority
from soul.core.sampler_manager import SamplerManager
from soul.core.idle_engine import IdleEngine
from soul.safety.guardrails import Guardrails, ResourceMonitor


class Soul:
    def __init__(self):
        print("Initializing Soul...")

        # Initialize components
        self.ai00 = AI00Client()
        self.memory = MemoryStore()
        self.tasks = TaskManager()
        self.samplers = SamplerManager()
        self.guardrails = Guardrails()
        self.monitor = ResourceMonitor()
        self.idle_engine = None

        print("Components initialized.")

    # Chat mode
    async def start_interactive(self, enable_idle: bool = False):
        # some very simple ascii art
        print("\n" + "=" * 60)
        print("  Soul - Personal AI Companion")
        print("  Type 'help' for commands, 'exit' to quit")
        print("=" * 60 + "\n")

        # Check ai00 connection
        if not await self.ai00.check_health():
            print("Warning: ai00-server not responding.")
            print("Please start ai00-server first.\n")
        else:
            print("Connected to ai00-server and it's ready to serve.\n")

        # Idle Mode with cancellation support
        idle_task = None
        idle_lock = asyncio.Lock()
        self._idle_cancel_event = asyncio.Event()

        if enable_idle:
            self.idle_engine = IdleEngine(
                ai00_client=self.ai00,
                memory_store=self.memory,
                task_manager=self.tasks,
                sampler_manager=self.samplers,
                guardrails=self.guardrails,
                resource_monitor=self.monitor,
                user_interrupt_event=self._idle_cancel_event,
            )
            idle_task = asyncio.create_task(self.idle_engine.start())
            print("Idle engine started (2-minute cycles)")
            print("Commands will interrupt idle tasks\n")

        # Main interaction loop
        try:
            await self._chat_loop(idle_lock=idle_lock)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            if idle_task and self.idle_engine:
                self.idle_engine.stop()
                idle_task.cancel()
                try:
                    await idle_task
                except asyncio.CancelledError:
                    pass

            self.memory.close()
            self.tasks.close()
            print("Goodbye!")

    async def _chat_loop(self, idle_lock: Optional[asyncio.Lock] = None):
        """Main chat interaction loop with idle interruption support."""
        # Set default sampler
        self.samplers.set_sampler("chat")

        # Load master personality
        prompt_path = (
            Path(__file__).parent
            / "soul"
            / "config"
            / "system_prompts"
            / "master_personality.txt"
        )
        system_prompt = ""
        if prompt_path.exists():
            system_prompt = prompt_path.read_text()

        conversation_history = []

        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            # Signal idle engine to interrupt current generation
            if hasattr(self, "_idle_cancel_event") and self._idle_cancel_event:
                self._idle_cancel_event.set()
                # Small delay to let idle tasks check the event
                await asyncio.sleep(0.1)
                self._idle_cancel_event.clear()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "help":
                self._show_help()
                continue
            elif user_input.lower() == "status":
                print(self._get_status())
                continue
            elif user_input.lower() == "pause":
                self.monitor.request_pause("User requested")
                print("✓ Idle operations paused")
                continue
            elif user_input.lower() == "resume":
                self.monitor.request_resume()
                print("✓ Idle operations resumed")
                continue
            elif user_input.lower().startswith("task "):
                task_desc = user_input[5:]
                if self.idle_engine:
                    self.idle_engine.add_user_task(task_desc)
                else:
                    self.tasks.add_task(task_desc)
                    print(f"✓ Task added: {task_desc[:50]}...")
                continue
            elif user_input.lower() == "memory":
                self._show_memory_stats()
                continue
            elif user_input.lower() == "tasks":
                self._show_task_stats()
                continue

            # Regular chat
            conversation_history.append({"role": "user", "content": user_input})

            # Store in memory
            self.memory.add_memory(
                content=f"User said: {user_input}",
                memory_type="short_term",
                metadata={"type": "user_input"},
            )

            # Generate response
            print("Soul: ", end="", flush=True)

            try:
                response = await self.ai00.chat_completion(
                    messages=conversation_history[-10:],  # Last 10 messages
                    system_prompt=system_prompt if system_prompt else None,
                    sampler_config=self.samplers.get_current_api_dict(),
                    max_tokens=1000,
                    stream=True,
                )

                conversation_history.append({"role": "assistant", "content": response})

                # Store response in memory
                self.memory.add_memory(
                    content=f"I responded: {response[:200]}",
                    memory_type="short_term",
                    metadata={"type": "ai_response"},
                )

            except Exception as e:
                print(f"[Error: {e}]")

    def _show_help(self):
        """Show help message."""
        print("""
Commands:
  help          Show this help message
  status        Show current status
  pause         Pause idle operations
  resume        Resume idle operations
  task <desc>   Add a background task
  memory        Show memory statistics
  tasks         Show task statistics
  exit/quit     Exit the program

Regular input will be sent to the AI for response.
""")

    def _get_status(self) -> str:
        """Get system status."""
        lines = ["\n--- Status ---"]

        if self.idle_engine:
            lines.append(self.idle_engine.get_status())
        else:
            lines.append("Idle engine: NOT RUNNING")

        mem_stats = self.memory.get_stats()
        lines.append(f"Memory: {mem_stats.get('total_memories', 0)} total memories")

        task_stats = self.tasks.get_stats()
        lines.append(f"Tasks: {task_stats.get('pending', 0)} pending")

        lines.append(f"Sampler: {self.samplers.current_sampler or 'default'}")
        lines.append(f"Paused: {self.monitor.is_paused}")
        lines.append("--------------\n")

        return "\n".join(lines)

    def _show_memory_stats(self):
        """Show memory statistics."""
        stats = self.memory.get_stats()
        print("\n--- Memory Statistics ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("-------------------------\n")

    def _show_task_stats(self):
        """Show task statistics."""
        stats = self.tasks.get_stats()
        pending = self.tasks.get_pending_tasks()

        print("\n--- Task Statistics ---")
        for status, count in stats.items():
            print(f"  {status}: {count}")

        if pending:
            print("\nPending tasks:")
            for task in pending[:5]:  # Show first 5
                print(f"  - {task.description[:60]}...")
        print("-----------------------\n")

    def add_task(self, description: str):
        """Add a background task."""
        task_id = self.tasks.add_task(description, priority=TaskPriority.MEDIUM)
        print(f"Added task #{task_id}: {description}")

    def cleanup(self):
        """Cleanup resources."""
        self.memory.close()
        self.tasks.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Soul - Personal AI Companion")
    parser.add_argument(
        "--idle", action="store_true", help="Enable idle loop for autonomous behavior"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current status and exit"
    )
    parser.add_argument("--pause", action="store_true", help="Pause idle operations")
    parser.add_argument("--resume", action="store_true", help="Resume idle operations")
    parser.add_argument(
        "--task", type=str, metavar="DESCRIPTION", help="Add a background task"
    )

    args = parser.parse_args()

    soul = Soul()

    try:
        if args.status:
            print(soul._get_status())
        elif args.pause:
            soul.monitor.request_pause("CLI request")
            print("Idle operations paused")
        elif args.resume:
            soul.monitor.request_resume()
            print("Idle operations resumed")
        elif args.task:
            soul.add_task(args.task)
        else:
            await soul.start_interactive(enable_idle=args.idle)
    finally:
        soul.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
