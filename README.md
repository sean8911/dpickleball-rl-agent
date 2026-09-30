# dPickleBall Reinforcement-Learning Agent

A reinforcement-learning agent for
[dPickleBall](https://github.com/dPickleball/dPickleBallEnv), a two-player
Unity game where each side's paddle is driven by a Python policy. The
policy sees only a real-time camera frame and returns three discrete
actions. This repository holds the full pipeline used to build the agent,
from environment measurement through a calibrated simulator, curriculum
pre-training, Unity fine-tuning, and self-play, together with the
deployment code and the final checkpoint.

This agent placed third out of thirteen groups, taking bronze, in the dPickleBall competition.

> For the detailed, phase-by-phase development write-up, see
> **[TRAINING.md](TRAINING.md)**.

## Overview

The competition exposes a hard real-time interface. The policy returns
three action components, vertical, horizontal, and rotation, each in
`{0, 1, 2}`, within about 10 ms per step, and it observes only an
`84 x 168` RGB frame. Two decisions follow from those limits.

- **Object-centric features over raw pixels.** An OpenCV extractor turns
  each frame into a frozen 16-dimensional feature vector of ball and
  paddle positions, velocities, and a side flag. A small MLP over that
  vector runs in well under a millisecond, where a deep CNN over a frame
  stack would not, and the geometric features close most of the
  sim-to-real gap by construction.
- **A staged, gated pipeline.** Learning happens mostly in a cheap
  parallel Python simulator that is calibrated against the real game
  first. The expensive Unity build is reserved for measurement,
  fine-tuning, and validation.

One shared side-aware policy drives both paddles. The observation is in
absolute pixel coordinates, so the policy is strongest in the left-paddle
frame. The right paddle mirrors its frame about the image centre, runs the
policy, and mirrors the action back.

## Repository structure

```
dPickleBall/
├── teamX.py / teamY.py             self-contained submission, left and right
├── checkpoints/
│   └── best_1_policy.pt.json       the deployed model metadata
├── README.md  TRAINING.md
└── agent/                          training framework
    ├── config.py                   single source of truth for constants
    ├── features.py                 frozen 16-dim feature contract
    ├── extractor.py                OpenCV ball and paddle state extractor
    ├── checkpoints.py              checkpoint ranking, metadata, export
    ├── curriculum.py               five-level training curriculum
    ├── rewards.py                  event-based reward policy
    ├── opponents.py                scripted opponents and self-play pool
    ├── train_sim.py                Phase 2 sim curriculum PPO
    ├── finetune_unity.py           Phase 3 Unity fine-tune
    ├── finetune_vs_opponent.py     Phase 3 fine-tune vs a fixed opponent
    ├── selfplay.py                 Phase 4 self-play
    ├── train_anticamper.py         anti-camper experiment
    ├── train_hybrid.py             anti-camper plus self-play experiment
    ├── sim/physics.py              calibrated 30 FPS physics simulator
    ├── envs/                       sim and unity Gym wrappers
    ├── calibrate/                  Phase 0 and 1 recorder and physics fit
    ├── tools/                      evaluation and diagnostics
    ├── tests/                      unit tests
    ├── MEASUREMENTS.md             measured environment facts
    └── CALIBRATION.md              simulator calibration notes
```

## Installation

Python 3.10 in a fresh virtual environment.

```bash
pip install numpy opencv-python torch stable-baselines3 gymnasium
```

Inference needs only `numpy`, `opencv-python`, and `torch`. Training and
Unity interaction also need the
[ML-Agents](https://github.com/Unity-Technologies/ml-agents) Python
packages `mlagents-envs` and `mlagents`, plus the compiled dPickleBall
Unity build. The build is supplied separately by the competition, and
scripts take its path through `--env-path`.

## Running the agent

The match runner imports `TeamX` for the left paddle and `TeamY` for the
right paddle and calls `policy(observation, reward)` each step.

```python
from teamX import TeamX
from teamY import TeamY

left, right = TeamX(), TeamY()
action = left.policy(observation, reward)   # [vertical, horizontal, rotation]
```

The policy loads the TorchScript checkpoint, validates it against the
16-dim feature contract, warms up the JIT, and pins PyTorch to one thread
to stay inside the step budget. Measured p99 latency is about 0.7 ms.

The competition runner calls the policy in a background thread and steps
with the previous action as a fallback, so actions land about one step
late. Two deploy-time measures handle this. The observation frame is
copied on entry because the shared buffer may be reused mid-read, and a
small vertical deadband holds the paddle when it is already aligned with
the ball, which stops a lag-induced oscillation. See
[TRAINING.md](TRAINING.md) Phase 6.

## Reproducing the training pipeline

Run from this directory so that `agent` is importable. `<dp.exe>` is the
path to the Unity build and `<run>` is a recording directory produced by
Phase 0.

```bash
# Phase 0: measure the environment
python -m agent.calibrate.record_unity --env-path <dp.exe> \
    --mode random --steps 40000 --graphics
# Phase 1: calibrate the simulator, then gate it against real flight
python -m agent.calibrate.fit_physics --run-dir <run>
python -m agent.tools.replay_compare --run-dir <run>
# Phase 2: sim curriculum pre-training
python -m agent.train_sim --num-envs 8 --total-steps 1200000 \
    --out-dir agent/checkpoints_v2
python -m agent.tools.eval_checkpoint --model agent/checkpoints_v2/best_1.zip
# Phase 3: Unity fine-tuning
python -m agent.finetune_unity --env-path <dp.exe> \
    --pretrained agent/checkpoints_v2/best_1.zip \
    --out-dir agent/checkpoints_unity_v2
# Phase 4: self-play
python -m agent.selfplay --num-envs 8 \
    --pretrained agent/checkpoints_unity_v2/best_1.zip \
    --out-dir agent/checkpoints_selfplay
# Phase 5: deployment gate
python -m agent.tools.timing_check --model checkpoints/best_1_policy.pt
```

Deploy the self-play `best_1_policy.pt` to `checkpoints/`, then play the
submission with the official harness. Copy `teamX.py`, `teamY.py`, and
`checkpoints/` into `dPickleBallEnv/CompetitionScripts/` and run
`Competition.py`.

Diagnostics live in `agent/tools/`. `feature_audit` checks sim-to-deploy
feature parity, `unity_probe` measures rally competence, `sanity_rollout`
runs a heuristic environment check, and `detection_rate` reports extractor
accuracy.

## Tests

```bash
python -m pytest agent/tests
```

The Unity-dependent tests skip automatically when no build is present.

## Notes

- Training run outputs under `agent/checkpoints*/` and `agent/data/` are
  git-ignored and regenerated by the scripts above.
- Constants are centralised in `agent/config.py`. The feature layout is
  defined once in `agent/features.py` and validated against checkpoint
  metadata on load.
