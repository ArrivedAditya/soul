# Soul - Personal AI Companion

A human-like AI that runs on edge devices with autonomous self-reflection, memory management, and the ability to say "no" to potentially harmful actions.

## Philosophy

Soul is designed to be a companion, not a tool racing toward AGI. It focuses on:
- **Authenticity**: Genuine interactions without pretense
- **Autonomy**: Self-reflection and continuous improvement during idle time
- **Safety**: Built-in guardrails that refuse harmful actions
- **Privacy**: Local-first, user-trusted, external sources treated as untrusted

## Features

### 🧠 Memory System
- **Three-tier memory**: Short-term → Long-term → Reflex (habits)
- **Forgetting curve**: Ebbinghaus algorithm with type-specific decay rates
- **Automatic consolidation**: Memories promoted based on strength and repetition
- **Resistance mechanism**: Reflex habits require saturation to change

### 🔄 Autonomous Idle Loop
- **2-minute cycles**: Runs continuously in background
- **Task processing**: Executes user-assigned background tasks
- **Self-reflection**: Analyzes recent interactions to improve
- **Memory maintenance**: Applies forgetting curve and consolidates

### 🎛️ Dynamic Sampler Management
- **Context-aware sampling**: Automatically switches samplers based on task
  - `chat`: Natural conversation (Nucleus, temp=0.8)
  - `reflect`: Creative thinking (Mirostat)
  - `task`: Focused execution (Nucleus, temp=0.4)
  - `decision`: Deterministic choices (Nucleus, temp=0.3)
- **Manual override**: User can force specific samplers

### 🛡️ Safety & Guardrails
- **"Say no" capability**: Refuses high-risk actions (deletion, external APIs, self-modification)
- **Pattern detection**: Prevents runaway behavior loops
- **Resource monitoring**: Pauses when GPU usage high (for your other work)
- **User pause/resume**: Full control over idle operations

## System Requirements

**Minimum:**
- Device: ThinkPad x250 (or similar)
- CPU: Intel Core i5-5200U (Gen 5)
- GPU: Intel HD 5500 (Gen 8) with Vulkan support
- RAM: 8GB
- OS: Linux (CachyOS/Arch-based recommended)

**Models Supported:**
- RWKV-7 "Goose" (current)
- RWKV-8 "Heron" ROSA-1bit (future, when officially released)

## Installation

```bash
# Clone repository
git clone <repo-url>
cd soul

# Install dependencies
pip install -r requirements.txt

# Or use the package
pip install -e .
```

## Usage

### Start ai00-server first
```bash
# In your ai00-server directory
./ai00_server
```

### Run Soul

```bash
# Interactive mode only
python main.py

# With autonomous idle loop
python main.py --idle

# Add a background task
python main.py --task "Research RWKV-8 capabilities"

# Check status
python main.py --status

# Pause/resume idle operations
python main.py --pause
python main.py --resume
```

### Interactive Commands

Once running, use these commands:
- `help` - Show help
- `status` - Current system status
- `pause` - Pause idle operations
- `resume` - Resume idle operations  
- `task <description>` - Add background task
- `memory` - Show memory statistics
- `tasks` - Show task queue
- `exit` or `quit` - Exit program

## Architecture

```
soul/
├── config/                     # Configuration files
│   ├── system_prompts/        # Personality & mode prompts
│   │   ├── master_personality.txt
│   │   ├── reflection_mode.txt
│   │   ├── task_mode.txt
│   │   └── consolidation_mode.txt
│   └── sampler_presets.json   # Sampler configurations
│
├── core/                       # Core functionality
│   ├── idle_engine.py         # Main autonomous loop
│   └── sampler_manager.py     # Dynamic sampler switching
│
├── memory/                     # Memory management
│   ├── memory_store.py        # SQLite + forgetting curve
│   └── task_manager.py        # Background task queue
│
├── models/                     # AI model interface
│   └── ai00_client.py         # ai00-server API client
│
├── safety/                     # Safety systems
│   └── guardrails.py          # "Say no" & resource monitoring
│
└── main.py                     # Entry point
```

## Memory Flow

```
Input → Short-term Memory → [Idle Processing] → Long-term Memory
                                    ↓
                            Forgetting Curve
                            Consolidation
                            Reflection
                                    ↓
                              Reflex (Habits)
```

**During idle cycles:**
1. Process pending tasks
2. Apply forgetting curve (decay rates: ST=0.3, LT=0.05, Reflex=0.01)
3. Consolidate memories (promote/demote based on strength)
4. Self-reflect on recent interactions
5. Report status

## Configuration

### Sampler Presets
Edit `soul/config/sampler_presets.json`:
```json
{
  "chat": {
    "type": "Nucleus",
    "temperature": 0.8,
    "top_p": 0.5,
    "top_k": 128
  },
  "reflect": {
    "type": "Mirostat",
    "tau": 0.5,
    "Rate": 0.09
  }
}
```

### System Prompts
Modify prompts in `soul/config/system_prompts/` to change personality.

## Future: RWKV-8 ROSA-1bit

When officially released and ai00-server supports it:
- 1-bit quantization (~32x memory reduction)
- ROSA (Rapid Online Suffix Automaton) instead of attention
- Perfect for edge devices like your ThinkPad x250
- Soul will automatically support it via ai00-server

## Safety Features

### Actions That Require Approval
- Deleting/modifying user files
- External API calls
- Network requests
- System commands
- Self-modification

### Resource Limits
- Max GPU: 85% (pauses idle to free resources for your work)
- Max RAM: 2GB for idle operations
- Auto-detects when you need GPU

### Pattern Detection
- Prevents >5 similar actions in 10 cycles
- Flags resource-intensive patterns
- Logs all autonomous actions

## Development

### Running Tests
```bash
# Coming soon
pytest tests/
```

### Type Checking
```bash
pyright soul/
```

## License

MIT License - See LICENSE file

## Credits

- **RWKV**: Bo Peng and the RWKV community
- **ai00-server**: Ai00-X development team
- **ROSA**: BlinkDL's Rapid Online Suffix Automaton

## Philosophy Quote

> "Focus on the joy of creating anything, not racing toward AGI/ASI."

Soul is your companion, not a replacement for human thought. It augments, automates, and assists while respecting your autonomy.
