"""Self-play opponent pool of frozen TorchScript snapshots.

Holds a bounded set of past-policy snapshots on disk; the sim env
samples one per episode. A baseline snapshot (index 0) is never
evicted so the learner always faces a floor opponent and cannot
collapse into a degenerate cycle.
"""

from pathlib import Path
from typing import List

from agent.checkpoints import export_torchscript


class SnapshotPool:
    """Bounded pool of TorchScript opponent snapshots on disk."""

    def __init__(self, out_dir: Path, max_size: int = 6) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._paths: List[str] = []
        self._counter = 0

    def seed(self, snapshot_path: str) -> None:
        """Add an existing .pt as the permanent baseline opponent."""
        path = Path(snapshot_path)
        if not path.is_file():
            raise FileNotFoundError(f"snapshot not found: {path}")
        self._paths.append(str(path))

    def add(self, model) -> str:
        """Export the model as a snapshot; evict oldest non-base."""
        path = self.out_dir / f"snap_{self._counter:04d}.pt"
        self._counter += 1
        export_torchscript(model, path)
        self._paths.append(str(path))
        # keep index 0 (baseline) pinned; evict the oldest after it
        if len(self._paths) > self.max_size:
            evicted = self._paths.pop(1)
            snap = Path(evicted)
            if snap.parent == self.out_dir:
                snap.unlink(missing_ok=True)
        return str(path)

    def paths(self) -> List[str]:
        """Current opponent snapshot paths."""
        return list(self._paths)
