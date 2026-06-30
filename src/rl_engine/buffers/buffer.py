"""
buffer.py
=========

Defines the Buffer Protocol -- the surface Trainer uses on any buffer,
regardless of whether it is a ReplayMemory (off-policy, persistent) or a
RolloutBuffer (on-policy, cleared after each update).

The two methods every buffer must have:
  * ingest_new_transitions -- read new lines from out.jsonl into the buffer
  * __len__                -- how many transitions are currently held

`sample` and `get_all`/`clear` are intentionally NOT part of this Protocol.
They belong to specific buffer kinds and Trainer already handles that split
via the algorithm's `buffer_kinds` dict -- putting them here would force
a false symmetry between two things that are genuinely different.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Protocol, Union, runtime_checkable


PathLike = Union[str, Path]


@runtime_checkable
class Buffer(Protocol):
    def __len__(self) -> int:
        ...

    def ingest_new_transitions(
        self, path: PathLike, role_filter: Optional[str] = None
    ) -> int:
        ...
