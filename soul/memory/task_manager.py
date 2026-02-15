"""
Task manager for background task queue with SQLite persistence.
"""

import sqlite3
import json
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a background task."""

    id: Optional[int]
    description: str
    status: TaskStatus
    priority: TaskPriority
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    result: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.id is None:
            self.id = int(time.time() * 1000)
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)
        if isinstance(self.priority, int):
            self.priority = TaskPriority(self.priority)
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)


class TaskManager:
    """SQLite-based task queue manager."""

    def __init__(self, db_path: str = "soul_memory.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """Initialize task table."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 2,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                result TEXT,
                error TEXT,
                metadata TEXT DEFAULT '{}',
                CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled'))
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_status ON tasks(status)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_priority ON tasks(priority)"
        )

        self.conn.commit()

    def add_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a new task to the queue."""
        now = time.time()
        meta = metadata or {}

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO tasks (description, status, priority, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                description,
                TaskStatus.PENDING.value,
                priority.value,
                now,
                json.dumps(meta),
            ),
        )

        self.conn.commit()
        return cursor.lastrowid or 0

    def get_next_task(self) -> Optional[Task]:
        """Get highest priority pending task."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM tasks 
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        """)

        row = cursor.fetchone()
        if row:
            task = self._row_to_task(row)
            task_id = task.id
            if task_id is not None:
                # Mark as in_progress
                now = time.time()
                self._update_status(task_id, TaskStatus.IN_PROGRESS, started_at=now)
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = now
                return task
        return None

    def _update_status(
        self,
        task_id: int,
        status: TaskStatus,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Update task status."""
        cursor = self.conn.cursor()

        updates = ["status = ?"]
        params: List[Any] = [status.value]

        if started_at is not None:
            updates.append("started_at = ?")
            params.append(started_at)
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)
        if result is not None:
            updates.append("result = ?")
            params.append(result)
        if error is not None:
            updates.append("error = ?")
            params.append(error)

        params.append(task_id)

        cursor.execute(
            f"""
            UPDATE tasks SET {", ".join(updates)} WHERE id = ?
        """,
            params,
        )

        self.conn.commit()

    def complete_task(self, task_id: int, result: Optional[str] = None):
        """Mark task as completed."""
        self._update_status(
            task_id, TaskStatus.COMPLETED, completed_at=time.time(), result=result
        )

    def fail_task(self, task_id: int, error: str):
        """Mark task as failed."""
        self._update_status(
            task_id, TaskStatus.FAILED, completed_at=time.time(), error=error
        )

    def cancel_task(self, task_id: int):
        """Cancel a pending task."""
        self._update_status(task_id, TaskStatus.CANCELLED)

    def get_task(self, task_id: int) -> Optional[Task]:
        """Get specific task by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_task(row)
        return None

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks sorted by priority."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM tasks 
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
        """)
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def get_active_tasks(self) -> List[Task]:
        """Get all in-progress tasks."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM tasks 
            WHERE status = 'in_progress'
            ORDER BY started_at ASC
        """)
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def get_completed_tasks(self, limit: int = 50) -> List[Task]:
        """Get recent completed/failed tasks."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM tasks 
            WHERE status IN ('completed', 'failed')
            ORDER BY completed_at DESC
            LIMIT ?
        """,
            (limit,),
        )
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, int]:
        """Get task statistics."""
        cursor = self.conn.cursor()

        stats = {}
        for status in TaskStatus:
            cursor.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = ?", (status.value,)
            )
            stats[status.value] = cursor.fetchone()[0]

        return stats

    def cleanup_old_tasks(self, days: int = 30):
        """Delete completed tasks older than specified days."""
        cutoff = time.time() - (days * 24 * 3600)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            DELETE FROM tasks 
            WHERE status IN ('completed', 'failed', 'cancelled')
            AND completed_at < ?
        """,
            (cutoff,),
        )

        self.conn.commit()

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert database row to Task object."""
        return Task(
            id=row["id"],
            description=row["description"],
            status=TaskStatus(row["status"]),
            priority=TaskPriority(row["priority"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            result=row["result"],
            error=row["error"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self):
        """Close database connection."""
        self.conn.close()
