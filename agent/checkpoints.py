"""Checkpoint ranking, persistence, and TorchScript export.

Keeps the top-K models by eval score, but a candidate is ineligible
if its left/right success gap exceeds config.MAX_SIDE_GAP (the
anti-asymmetry floor v39 lacked). Each kept model writes an SB3 zip
(resume/fine-tune), a traced TorchScript actor (.pt, argmax per
branch, no SB3 at inference), and a metadata JSON whose feature_dim
is validated on load. A top_models.json manifest tracks the
ranking; promote_best() copies rank 1 to the submission name.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn

from agent import config

MANIFEST = "top_models.json"


@dataclass
class CheckpointMeta:
    """Persisted summary of one checkpoint (stem locates its files)."""

    stem: str
    score: float
    level: int
    overall_rate: float
    left_rate: float
    right_rate: float
    total_steps: int
    feature_dim: int
    action_nvec: List[int]


class _DeterministicActor(nn.Module):
    """Wrap an SB3 PPO policy as obs -> argmax action per branch.

    Holds the policy's feature extractor, actor MLP, and action
    head; the value net is dropped. Output is an int64 tensor of
    one index per discrete branch.
    """

    def __init__(self, policy, nvec: List[int]) -> None:
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.split_sizes = [int(n) for n in nvec]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)
        parts = torch.split(logits, self.split_sizes, dim=-1)
        idx = [torch.argmax(p, dim=-1) for p in parts]
        return torch.stack(idx, dim=-1)


def export_torchscript(model, path: Path) -> None:
    """Trace the deterministic actor and save a TorchScript .pt."""
    nvec = [int(n) for n in model.action_space.nvec]
    actor = _DeterministicActor(model.policy, nvec).eval()
    example = torch.zeros(
        (1, config.FEATURE_DIM), dtype=torch.float32
    )
    with torch.no_grad():
        traced = torch.jit.trace(actor, example)
    traced.save(str(path))


def load_meta(meta_path: Path) -> CheckpointMeta:
    """Load checkpoint metadata, validating the feature dimension."""
    data = json.loads(Path(meta_path).read_text())
    if data["feature_dim"] != config.FEATURE_DIM:
        raise ValueError(
            f"checkpoint feature_dim {data['feature_dim']} != "
            f"config.FEATURE_DIM {config.FEATURE_DIM} ({meta_path})"
        )
    return CheckpointMeta(**data)


def eligible(left_rate: float, right_rate: float) -> bool:
    """False if the per-side gap exceeds the anti-asymmetry floor."""
    return abs(left_rate - right_rate) <= config.MAX_SIDE_GAP


class CheckpointManager:
    """Maintain the top-K eligible checkpoints on disk."""

    def __init__(
        self,
        out_dir: Optional[Path] = None,
        top_k: int = config.TOP_K_MODELS,
    ) -> None:
        self.out_dir = Path(out_dir or config.CHECKPOINT_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self._entries: List[CheckpointMeta] = []
        self._counter = 0

    def _delete_files(self, stem: str) -> None:
        for suffix in (".zip", "_policy.pt", "_policy.pt.json"):
            (self.out_dir / f"{stem}{suffix}").unlink(missing_ok=True)

    def _write_manifest(self) -> None:
        manifest = {
            "entries": [asdict(m) for m in self._entries]
        }
        (self.out_dir / MANIFEST).write_text(
            json.dumps(manifest, indent=2)
        )

    def consider(
        self,
        model,
        *,
        score: float,
        level: int,
        overall_rate: float,
        left_rate: float,
        right_rate: float,
        total_steps: int,
    ) -> bool:
        """Save a candidate if eligible and within the top-K.

        Returns True if the candidate was kept.
        """
        if not eligible(left_rate, right_rate):
            return False
        if len(self._entries) >= self.top_k:
            worst = min(m.score for m in self._entries)
            if score <= worst:
                return False
        stem = f"ckpt_{self._counter:04d}"
        self._counter += 1
        meta = CheckpointMeta(
            stem=stem, score=score, level=level,
            overall_rate=overall_rate, left_rate=left_rate,
            right_rate=right_rate, total_steps=total_steps,
            feature_dim=config.FEATURE_DIM,
            action_nvec=[config.ACTION_CHOICES]
            * config.ACTION_BRANCHES,
        )
        model.save(self.out_dir / f"{stem}.zip")
        export_torchscript(
            model, self.out_dir / f"{stem}_policy.pt"
        )
        (self.out_dir / f"{stem}_policy.pt.json").write_text(
            json.dumps(asdict(meta), indent=2)
        )
        self._entries.append(meta)
        self._entries.sort(key=lambda m: m.score, reverse=True)
        for dropped in self._entries[self.top_k:]:
            self._delete_files(dropped.stem)
        self._entries = self._entries[: self.top_k]
        self._write_manifest()
        return True

    def promote_best(self) -> Optional[Path]:
        """Copy the rank-1 checkpoint to best_1.* for submission."""
        if not self._entries:
            return None
        best = self._entries[0]
        for suffix in (".zip", "_policy.pt", "_policy.pt.json"):
            shutil.copyfile(
                self.out_dir / f"{best.stem}{suffix}",
                self.out_dir / f"best_1{suffix}",
            )
        return self.out_dir / "best_1_policy.pt"
