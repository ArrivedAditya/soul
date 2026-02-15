"""
Memory system with SQLite backend and Ebbinghaus forgetting curve.
"""

import sqlite3
import json
import time
import math
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path


@dataclass
class Memory:
    """Represents a single memory entry."""

    id: Optional[int]
    content: str
    memory_type: str  # 'short_term', 'long_term', 'reflex'
    strength: float  # 0.0 to 1.0
    created_at: float
    last_accessed: float
    access_count: int
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.id is None:
            self.id = int(time.time() * 1000)  # Simple ID generation
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)


class MemoryStore:
    """SQLite-based memory storage with forgetting curve support."""

    def __init__(self, db_path: str = "soul_memory.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """Initialize database tables."""
        cursor = self.conn.cursor()

        # Main memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                CHECK (memory_type IN ('short_term', 'long_term', 'reflex'))
            )
        """)

        # Create indexes for efficient queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strength ON memories(strength)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)")

        self.conn.commit()

    def add_memory(
        self,
        content: str,
        memory_type: str = "short_term",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a new memory."""
        now = time.time()
        metadata = metadata or {}

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO memories (content, memory_type, strength, created_at, 
                                last_accessed, access_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (content, memory_type, 1.0, now, now, 1, json.dumps(metadata)),
        )

        self.conn.commit()
        return cursor.lastrowid or 0

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_memory(row)
        return None

    def get_memories_by_type(
        self, memory_type: str, limit: Optional[int] = None, min_strength: float = 0.3
    ) -> List[Memory]:
        """Get all memories of a specific type above strength threshold."""
        cursor = self.conn.cursor()

        query = """
            SELECT * FROM memories 
            WHERE memory_type = ? AND strength >= ?
            ORDER BY strength DESC, last_accessed DESC
        """
        params = [memory_type, min_strength]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def access_memory(self, memory_id: int) -> None:
        """Mark a memory as accessed (strengthens it)."""
        now = time.time()
        cursor = self.conn.cursor()

        cursor.execute(
            """
            UPDATE memories 
            SET access_count = access_count + 1,
                last_accessed = ?,
                strength = MIN(1.0, strength + 0.1)
            WHERE id = ?
        """,
            (now, memory_id),
        )

        self.conn.commit()

    def update_strength(self, memory_id: int, new_strength: float) -> None:
        """Update memory strength directly."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memories SET strength = ? WHERE id = ?",
            (max(0.0, min(1.0, new_strength)), memory_id),
        )
        self.conn.commit()

    def update_type(self, memory_id: int, new_type: str) -> None:
        """Change memory type (e.g., short_term -> long_term)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memories SET memory_type = ? WHERE id = ?", (new_type, memory_id)
        )
        self.conn.commit()

    def delete_memory(self, memory_id: int) -> None:
        """Delete a memory."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()

    def apply_forgetting_curve(
        self, decay_rates: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Apply Ebbinghaus forgetting curve to all memories.
        Returns number of memories below threshold.
        """
        default_rates = {
            "short_term": 0.3,  # Fast decay
            "long_term": 0.05,  # Slow decay
            "reflex": 0.01,  # Very slow decay (habits are hard to change)
        }
        rates = decay_rates or default_rates

        cursor = self.conn.cursor()
        now = time.time()

        # Get all memories
        cursor.execute("SELECT * FROM memories")
        memories = cursor.fetchall()

        forgotten_count = 0

        for row in memories:
            memory = self._row_to_memory(row)
            elapsed_hours = (now - memory.last_accessed) / 3600

            # Ebbinghaus formula: R = e^(-t/S)
            # Where R = retention, t = time, S = relative strength of memory
            decay_rate = rates.get(memory.memory_type, 0.1)
            new_strength = memory.strength * math.exp(-elapsed_hours * decay_rate)

            if new_strength < 0.3:
                forgotten_count += 1

            cursor.execute(
                "UPDATE memories SET strength = ? WHERE id = ?",
                (new_strength, memory.id),
            )

        self.conn.commit()
        return forgotten_count

    def consolidate_memories(self) -> Dict[str, int]:
        """
        Consolidate memories based on access patterns.
        Returns stats of promotions/demotions.
        """
        stats = {"promoted": 0, "demoted": 0, "reflexed": 0}

        cursor = self.conn.cursor()

        # Promote short-term to long-term (high strength + multiple accesses)
        cursor.execute("""
            SELECT * FROM memories 
            WHERE memory_type = 'short_term' 
            AND strength > 0.7 
            AND access_count >= 3
        """)

        for row in cursor.fetchall():
            self.update_type(row["id"], "long_term")
            stats["promoted"] += 1

        # Promote long-term to reflex (very high access, consistent pattern)
        cursor.execute("""
            SELECT * FROM memories 
            WHERE memory_type = 'long_term' 
            AND access_count >= 10
            AND strength > 0.8
        """)

        for row in cursor.fetchall():
            # Check if memory has resistance threshold
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            if metadata.get("resistance", 0) < 5:  # Not too resistant
                self.update_type(row["id"], "reflex")
                stats["reflexed"] += 1

        # Demote weak memories
        cursor.execute("""
            SELECT * FROM memories 
            WHERE memory_type = 'long_term' 
            AND strength < 0.3
        """)

        for row in cursor.fetchall():
            self.update_type(row["id"], "short_term")
            stats["demoted"] += 1

        return stats

    def search_memories(
        self, query: str, memory_type: Optional[str] = None
    ) -> List[Memory]:
        """Simple keyword search in memory content."""
        cursor = self.conn.cursor()

        if memory_type:
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE content LIKE ? AND memory_type = ?
                ORDER BY strength DESC
            """,
                (f"%{query}%", memory_type),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM memories 
                WHERE content LIKE ?
                ORDER BY strength DESC
            """,
                (f"%{query}%",),
            )

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        cursor = self.conn.cursor()

        stats = {}

        # Count by type
        cursor.execute(
            "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
        )
        for row in cursor.fetchall():
            stats[f"{row[0]}_count"] = row[1]

        # Average strength by type
        cursor.execute(
            "SELECT memory_type, AVG(strength) FROM memories GROUP BY memory_type"
        )
        for row in cursor.fetchall():
            stats[f"{row[0]}_avg_strength"] = round(row[1], 3) if row[1] else 0

        # Total memories
        cursor.execute("SELECT COUNT(*) FROM memories")
        stats["total_memories"] = cursor.fetchone()[0]

        return stats

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            strength=row["strength"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self):
        """Close database connection."""
        self.conn.close()
