"""
Sampler manager for dynamic sampling strategy switching.
Integrates with ai00-server API.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SamplerConfig:
    """Represents a sampler configuration."""

    name: str
    type: str  # 'Nucleus', 'Mirostat', 'Typical'
    temperature: float = 1.0
    top_p: float = 0.5
    top_k: int = 128
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty: int = 400
    penalty_decay: float = 0.99654026
    tau: Optional[float] = None
    rate: Optional[float] = None

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to ai00 API format."""
        config: Dict[str, Any] = {"type": self.type}

        if self.type == "Nucleus":
            config["temperature"] = self.temperature
            config["top_p"] = self.top_p
            config["top_k"] = self.top_k
            config["presence_penalty"] = self.presence_penalty
            config["frequency_penalty"] = self.frequency_penalty
            config["penalty"] = self.penalty
            config["penalty_decay"] = self.penalty_decay
        elif self.type == "Mirostat":
            config["tau"] = self.tau or 0.5
            config["Rate"] = self.rate or 0.09
        elif self.type == "Typical":
            config["temperature"] = self.temperature
            config["top_p"] = self.top_p
            config["top_k"] = self.top_k
            config["tau"] = self.tau or 5.0
            config["presence_penalty"] = self.presence_penalty
            config["frequency_penalty"] = self.frequency_penalty
            config["penalty"] = self.penalty
            config["penalty_decay"] = self.penalty_decay

        return config


class SamplerManager:
    """Manages sampler presets and dynamic switching."""

    def __init__(self, presets_path: Optional[str] = None):
        if presets_path is None:
            self.presets_path = (
                Path(__file__).parent.parent / "config" / "sampler_presets.json"
            )
        else:
            self.presets_path = Path(presets_path)
        self.presets: Dict[str, SamplerConfig] = {}
        self.current_sampler: Optional[str] = None

        self._load_presets()

    def _load_presets(self):
        """Load sampler presets from JSON file."""
        if not self.presets_path.exists():
            # Create default presets
            self._create_default_presets()

        with open(self.presets_path, "r") as f:
            data = json.load(f)

        for name, config in data.items():
            self.presets[name] = SamplerConfig(
                name=name,
                type=config.get("type", "Nucleus"),
                temperature=config.get("temperature", 1.0),
                top_p=config.get("top_p", 0.5),
                top_k=config.get("top_k", 128),
                presence_penalty=config.get("presence_penalty", 0.0),
                frequency_penalty=config.get("frequency_penalty", 0.0),
                penalty=config.get("penalty", 400),
                penalty_decay=config.get("penalty_decay", 0.99654026),
                tau=config.get("tau"),
                rate=config.get("Rate"),
            )

    def _create_default_presets(self):
        """Create default sampler presets optimized for RWKV7."""
        defaults = {
            "chat": {
                "type": "Typical",
                "temperature": 0.9,
                "top_p": 0.5,
                "top_k": 128,
                "tau": 9.5,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.3,
                "penalty": 400,
                "penalty_decay": 0.99654026,
            },
            "reflect": {
                "type": "Mirostat",
                "tau": 0.5,
                "Rate": 0.09,
                "penalty": 400,
                "penalty_decay": 0.99654026,
            },
            "task": {
                "type": "Nucleus",
                "temperature": 0.4,
                "top_p": 0.3,
                "top_k": 64,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.3,
                "penalty": 400,
                "penalty_decay": 0.99654026,
            },
            "decision": {
                "type": "Nucleus",
                "temperature": 0.3,
                "top_p": 0.2,
                "top_k": 32,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.3,
                "penalty": 400,
                "penalty_decay": 0.99654026,
            },
            "idle": {
                "type": "Typical",
                "temperature": 0.7,
                "top_p": 0.4,
                "top_k": 64,
                "tau": 9.5,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4,
                "penalty": 400,
                "penalty_decay": 0.99654026,
            },
        }

        self.presets_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.presets_path, "w") as f:
            json.dump(defaults, f, indent=2)

    def get_sampler(self, name: str) -> Optional[SamplerConfig]:
        """Get a sampler configuration by name."""
        return self.presets.get(name)

    def set_sampler(self, name: str) -> Optional[SamplerConfig]:
        """Set current sampler by name."""
        if name in self.presets:
            self.current_sampler = name
            return self.presets[name]
        return None

    def get_current(self) -> Optional[SamplerConfig]:
        """Get currently selected sampler."""
        if self.current_sampler:
            return self.presets.get(self.current_sampler)
        return None

    def get_current_api_dict(self) -> Dict[str, Any]:
        """Get current sampler in ai00 API format."""
        sampler = self.get_current()
        if sampler:
            return sampler.to_api_dict()
        # Return default Nucleus
        return SamplerConfig(name="default", type="Nucleus").to_api_dict()

    def create_custom_sampler(self, name: str, config: Dict[str, Any]) -> SamplerConfig:
        """Create a custom sampler preset."""
        sampler = SamplerConfig(name=name, **config)
        self.presets[name] = sampler
        return sampler

    def auto_select_sampler(self, context: str, task_type: Optional[str] = None) -> str:
        """
        Automatically select appropriate sampler based on context.

        Args:
            context: Description of current situation
            task_type: Optional specific task type hint

        Returns:
            Name of selected sampler preset
        """
        # Simple rule-based selection
        if task_type:
            if task_type in ["reflection", "thinking", "analysis"]:
                return "reflect"
            elif task_type in ["task", "work", "execution"]:
                return "task"
            elif task_type in ["decision", "choice", "deterministic"]:
                return "decision"

        # Context-based heuristics
        context_lower = context.lower()

        if any(
            word in context_lower
            for word in ["reflect", "think", "analyze", "review", "improve", "learn"]
        ):
            return "reflect"
        elif any(
            word in context_lower
            for word in [
                "task",
                "work",
                "execute",
                "process",
                "compute",
                "analyze data",
            ]
        ):
            return "task"
        elif any(
            word in context_lower
            for word in ["decide", "choose", "determine", "select", "judge"]
        ):
            return "decision"
        elif any(
            word in context_lower
            for word in ["create", "imagine", "brainstorm", "creative"]
        ):
            return "creative"

        return "chat"

    def list_presets(self) -> Dict[str, str]:
        """List all available presets with descriptions."""
        descriptions = {
            "chat": "Natural conversation mode (balanced)",
            "reflect": "Self-reflection and analysis (creative)",
            "task": "Task execution mode (focused)",
            "decision": "Decision-making mode (deterministic)",
            "creative": "Creative generation mode (diverse)",
        }

        return {
            name: descriptions.get(name, "Custom preset")
            for name in self.presets.keys()
        }
