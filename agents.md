# RL-Engine — Agent Reference

SAC/PPO trainer. Drives Lynch, ingests history, writes checkpoints to `shared/models/`.

**Core purpose:** Communication with Lynch/NeonFC + transition parsing + "select algorithm & run". Originally sum-zero adversarial only; now supports flexible reward modes.

---

## Design Shift: Sum-Zero → Flexible Modes

| Aspect | Original (Sum-Zero) | New (Flexible) |
|--------|---------------------|----------------|
| **Reward assumption** | One role +r, other -r | Per-role independent rewards |
| **Algorithm** | Single SAC/PPO | `AdversarialSAC` / `SingleSAC` (and PPO variants) |
| **Critic** | Shared, zero-sum value | Mode-dependent |
| **Config** | Implicit | Explicit `mode` in `[training]` |

**Migration:** In progress. Both modes coexist; select via config.

---

## Algorithm Modes

| Mode | File | Reward Handling | Use Case |
|------|------|-----------------|----------|
| `adversarial_sac` | `algorithms/adversarial_sac.py` | Zero-sum: striker +r, goalkeeper -r (or vice versa) | Competitive scenarios |
| `single_sac` | `algorithms/single_sac.py` | Independent per-role rewards | Cooperative / single-agent |
| `adversarial_ppo` | `algorithms/adversarial_ppo.py` | Zero-sum | Competitive, on-policy |
| `single_ppo` | `algorithms/single_ppo.py` | Independent per-role | Cooperative, on-policy |

- **Shared critic** in all modes? Mode-dependent (adversarial shares; single may not)
- **Config switch:**
  ```toml
  [training]
  algorithm = "adversarial_sac"   # or "single_sac", "adversarial_ppo", "single_ppo"
  ```

---

## Training Loop (`trainer.py::run()`)

```python
# 1. Connect to Lynch (TCP :10003)
# 2. Send RunRequest {test_case, epochs, config, mode}
# 3. Wait for RunResponse {status, history_files, summary_file}
# 4. Read JSONL from shared/results/<SCENARIO>/history_*.jsonl
# 5. Parse transitions → build buffers (per role, mode-aware)
# 6. Train batch (selected algorithm)
# 7. Save checkpoint to shared/models/<role>.pt
# 8. Repeat for epochs (or until KeyboardInterrupt → save final)
```

| Step | Detail |
|------|--------|
| **Lynch connection** | `TestAbstraction` client — `host:port` from config |
| **Request payload** | `test_case`, `epochs`, full config (includes `algorithm` mode) |
| **Response** | Lynch returns when episodes done: history files + summary |
| **Ingestion** | Reads JSONL → builds replay/PPO buffers per role |
| **Training** | Dispatches to selected algorithm's `train(batch)` |
| **Checkpoint** | `SaveSystem.save(epoch, algorithm)` → atomic write per role |
| **Epochs** | `--epochs N` CLI arg, or infinite (Ctrl-C saves final) |

---

## Lynch Communication (Core)

| Channel | Protocol | Purpose |
|---------|----------|---------|
| **Request** `:10003` | JSON lines | `RunRequest` {test_case, epochs, config, algorithm_mode} |
| **Response** `:10003` | JSON lines | `RunResponse` {status, history_files[], summary_file} |
| **Timeouts** | Config | `connect_timeout` (5s), `ack_timeout` (5s) |

**JSONL Transition Schema** (written by Lynch, read by RL-Engine):
```json
{
  "cur_state": [16],
  "next_state": [16],
  "actions": {"goalkeeper": [4], "striker": [4]},
  "rewards": {"goalkeeper": 0.5, "striker": -0.5},   // per-role, sign depends on mode
  "done": true
}
```
- Parser handles both zero-sum (opposite signs) and independent rewards transparently

---

## Checkpoint Format (`save_system.py`)

```python
# Per role file: shared/models/<role>.pt
{
    "layer_sizes": [256, 256],
    "state_dict": {...}             # actor + critic (shared or per-mode) weights
}
```
- **Atomic write:** `.tmp` → `os.rename()`
- **One file per role:** `striker.pt`, `goalkeeper.pt` — NeonFC loads each

---

## Config Schema (`config.toml`)

```toml
[runner]
host = "127.0.0.1"
port = 10003
connect_timeout = 5.0
ack_timeout = 5.0

[training]
algorithm = "adversarial_sac"   # "adversarial_sac" | "single_sac" | "adversarial_ppo" | "single_ppo"
role_ids = ["striker", "goalkeeper"]
state_dim = 16
action_dim = 4
rollout_batch_size = 32
save_every_n_epochs = 10
test_case = "SHOOTOUT"          # must match Lynch scenario
input_module = "NeonFC"

[adversarial_sac]
hidden_sizes = [256, 256]
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
gamma = 0.99
tau = 0.005
initial_alpha = 0.2
auto_alpha = true
replay_capacity = 200_000
min_buffer_size = 1_000
batch_size = 256

[single_sac]
hidden_sizes = [256, 256]
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
gamma = 0.99
tau = 0.005
initial_alpha = 0.2
auto_alpha = true
replay_capacity = 200_000
min_buffer_size = 1_000
batch_size = 256

[paths]
save_dir = "runs/training"   # symlink → shared/models/
log_dir  = "runs/tensorboard"
```

**Tune:** `algorithm` (mode), `role_ids`, `rollout_batch_size`, `save_every_n_epochs`, `test_case`, per-algorithm hyperparams.

---

## Adding a New Algorithm

1. Implement `algorithms/<name>.py`:
   ```python
   class SharedCritic<Name>:
       def __init__(self, role_ids, state_dim, action_dim, config, ...): ...
       def train(self, batch): ...           # returns loss dict
       def save(self, path): ...             # returns {layer_sizes, state_dict}
       @classmethod
       def load(cls, path, ...): ...         # returns instance
   ```
2. Register in `main.py::build_algorithm()`:
   ```python
   elif t.algorithm == "<name>":
       from rl_engine.algorithms.<name> import SharedCritic<Name>
       return SharedCritic<Name>(...)
   ```

---

## Debug Steps

| Symptom | First Checks |
|---------|--------------|
| Won't connect to Lynch | Lynch running? `:10003` open? `[runner]` host/port match? |
| No history files from Lynch | Lynch completed episodes? `logs/LyNCh.log`. `shared/results/` symlink valid? |
| Training diverges / NaN | LR too high? `gamma`/`tau`? Gradient clipping? TensorBoard |
| Checkpoint not loaded by NeonFC | `shared/models/` has `.pt`? `neonfc.toml` path correct? `ModelUpdate` sent? |
| Wrong reward mode | `algorithm` in config matches expected mode? JSONL `rewards` sign as expected? |

---

## Run Commands

```bash
# From RL-Control root:
./run.sh RL-Engine
# Or direct:
poetry run train --config ../configs/rlengine.toml --epochs 100
```

---

> **Note:** See `RL-Control/agents.md` for orchestration, ports, shared dirs. See `LyNCh/agents.md` for harness interface.