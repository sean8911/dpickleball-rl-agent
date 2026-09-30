# Environment Measurements

All facts below were verified empirically against
`dPickleball BuildFiles\Training\Windows\dp.exe` via
`agent.calibrate.record_unity`. Items marked [doc] come from
docs/code only and were not independently re-verified.

## Launch
- Graphics required. `no_graphics=True` crashes the build with
  0xC0000005 before the first reset. Always launch with graphics.
- Competition build does not connect via the standard
  ML-Agents handshake (UnityTimeOutException on two ports/worker
  ids). All training and eval use the Training build.
- Throughput: ~47 env steps/s (graphics, this machine).
  1M Unity steps ~= 6 h wall-clock.

## Agents and observation
- Agents: `PAgent1?team=0?agent_id=0` = LEFT,
  `PAgent2?team=0?agent_id=2` = RIGHT.
- Visual obs `(3, 84, 168)` float32 in [0, 1], only on agent 0;
  both paddles must read the same image.
- Agent 0 additionally exposes a scalar obs `(1,)` (0.0 in all
  inspected frames; meaning unknown). Agent 1 has only the scalar.
- Between points the ball is absent from the frame (out of play);
  longest observed absence ~47 frames (~1.5 s).

## Actions (measured from 10k sweep steps)
- Branches `[vertical, horizontal, rotation]`, each {0, 1, 2}.
- Global/screen-space semantics, identical for both paddles:
  - vertical: 1 = up (dy<0), 2 = down (dy>0)
  - horizontal: 1 = screen-right (dx>0), 2 = screen-left (dx<0)
  - rotation: 1 = ccw, 2 = cw [doc; angle tracking in Phase 1]
- Measured speeds ~0.7-1.3 px/step (means depressed by holds
  against court bounds; Phase 1 filters bound-clamped steps).
- `test_paral.py`'s action comments are mislabeled; the README
  mapping is correct.

## Rewards and termination
- Reward is +1 to the scorer, 0 to the opponent on the scoring
  step; all other steps 0 for both. No -1 events observed
  (150 point events, 40k steps), not zero-sum at env level.
- `done` never fired and `env.agents` never emptied in 53k
  recorded steps spanning ~150 points under serve code 212. The
  env continues past 21 points; match end must be tracked
  externally from cumulative rewards (as Competition.py does).
- Stuck-ball >5 s concession rule: [doc] not yet isolated in data.

## Extractor (ported v39, `agent/extractor.py`)
- Paddle detection 100% (both sides), opponent 100%.
- Ball detected ~100% of frames where it is on court; all misses
  are out-of-play pauses.
- 0.24 ms mean / 0.51 ms p99 per frame, far inside the 10 ms
  policy budget.
- Court ROI (y 20-80, x 8-164) correctly excludes the orange
  score-area decorations at the top of the frame.

## Datasets (agent/data/recordings/)
- `random_run`: 40k steps held-random, 150 points.
- `sweep_run`: 10k steps single-branch probes.
- `smoke_test`: 3k steps (also the unit-test fixture).
- Format: `chunk_NNNN.npz` with `frames` uint8 (N,84,168,3),
  `actions` int8 (N,2,3) [left,right], `rewards` float32 (N,2),
  plus `meta.json` per run.
