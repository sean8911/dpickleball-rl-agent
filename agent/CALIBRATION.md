# Phase 1: Sim calibration results

System ID over 40k random-play Unity steps (tracks cached in
`data/recordings/random_run/tracks.npz`). Fitted values
live in `config.py` under "Calibrated sim physics"; the sim is
`sim/physics.py`. Reproduce with:

    python -m agent.calibrate.fit_physics --run-dir <run>
    python -m agent.tools.replay_compare  --run-dir <run>

## Fitted / assumed parameters
| Param | Value | Source |
|-------|-------|--------|
| Court ball x | [9.5, 163.5] | measured extremes |
| Wall y (top/bot) | 21.5 / 79.5 | measured extremes |
| Left paddle x | [21, 82] | measured |
| Right paddle x | [85, 147] | measured |
| Paddle y | [24, 78] | measured |
| Ball drag/step | 0.982 | median of 1471 flight samples |
| Gravity | 0 | measured vy-accel ~0 |
| Wall restitution | 1.0 | assumed elastic; 1 clean bounce agrees |
| Paddle v-speed | 0.82 px | bound-filtered mean |
| Paddle h-speed | 0.70 px | bound-filtered mean |
| Rotation | 0.06 rad | DEFAULT (unmeasurable, see below) |
| Hit model | v39 form | starting point, tuned later |

## Replay-compare gate (sim vs real Unity)
- **Free flight: PASS.** mean 0.36 px, p90 1.05 px over 316
  segments (gate < 2.0). Geometry + drag + integration validated.
  (Drag was fit on this data, so this is a consistency check.)
- **Wall bounce: DATA-LIMITED, not passed.** Random play produced
  only ~37 wall events, ~1 of them a clean isolated bounce; the
  rest are slow (sub-noise vy) or paddle-entangled (vx sign flips).
  The model is textbook elastic vy-reflection and every observed
  bounce flips vy with restitution ~1.0 (e.g. idx12774:
  vy 1.75 -> -1.75 exactly). Quantitative wall validation is
  deferred to Phase 3 (Unity fine-tune) and covered by domain
  randomization at curriculum level 5.

## Known limitations (carried into Phase 2/3)
- **Rotation rate unmeasured.** minAreaRect angle on the near-square
  paddle is too noisy; ccw/cw did not separate. Set to 0.06 rad/step
  as a placeholder; revisit if Phase 2 policies misuse rotation.
- **Hit response unfit.** Random play never strikes hard (all balls
  ~1 px/step), and a ball-chasing collector traps the ball via the
  5 s-stuck rule rather than rallying. Using the v39 hit form
  (out = 0.55*incoming + 1.05*paddle_motion); the residual is an
  explicit Phase 3 fine-tune target and a DR axis.
- **Fast-ball dynamics unobserved.** Drag/restitution are speed
  ratios, so slow-ball fits should extrapolate; verified only up to
  ~2.3 px/step.
