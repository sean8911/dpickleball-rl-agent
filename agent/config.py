"""Single source of truth for agent constants.

Every magic number in the package lives here. Calibrated physics
parameters (Phase 1) land here as well; until then only verified
interface facts and recording settings are defined.
"""

from pathlib import Path

# observation tensor as delivered by ML-Agents: (C, H, W), float [0, 1]
IMG_C = 3
IMG_H = 84
IMG_W = 168

# three discrete branches (vertical, horizontal, rotation), each {0,1,2}
ACTION_BRANCHES = 3
ACTION_CHOICES = 3

# frozen feature contract; layout in features.py, validated on load
FEATURE_DIM = 16

# serve code: last digit = mode (1L/2R/3rand), lead digits = points
DEFAULT_SERVE_CODE = 212

# recording (Phase 0)
PACKAGE_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PACKAGE_DIR / "data" / "recordings"
REC_CHUNK_STEPS = 2000
# hold each probe action this long so displacement is measurable
SWEEP_HOLD_STEPS = 30

# extractor HSV thresholds (proven v39 values)
BALL_HSV_LO = (20, 80, 80)
BALL_HSV_HI = (45, 255, 255)
RACKET_ORANGE_LO = (5, 45, 45)
RACKET_ORANGE_HI = (30, 255, 255)
# red wraps around hue 0/179, so two ranges
RACKET_RED1_LO = (0, 45, 45)
RACKET_RED1_HI = (6, 255, 255)
RACKET_RED2_LO = (170, 45, 45)
RACKET_RED2_HI = (179, 255, 255)

# court region of interest; excludes score/logo border pixels
ROI_X1, ROI_X2 = 8, 164
ROI_Y1, ROI_Y2 = 20, 80

# blob-size limits (pixels^2) for detection
BALL_MIN_AREA = 3
BALL_MAX_AREA = 160
PADDLE_MIN_AREA = 15

# 8.0 not 4.0 so Unity frame-delta velocities do not saturate
BALL_VEL_NORM = 8.0  # px/step mapped to [-1, 1]
# a delta above this is occlusion recovery, not motion; zeroed
MAX_BALL_DELTA_PX = 6.0
PADDLE_FALLBACK_X_RIGHT = 0.82  # frac of width when not detected
PADDLE_FALLBACK_X_LEFT = 0.18
PADDLE_FALLBACK_Y = 0.55  # frac of height

# hold vertical within this many px of the ball (stops the hop)
Y_DEADBAND_PX = 3.0

# track segmentation / system ID (Phase 1)
TRACK_EVENT_SPEED_DELTA = 0.8  # px/step jump that marks an event
TRACK_EVENT_ANGLE_DELTA = 0.44  # rad (~25 deg) turn that marks one
TRACK_MIN_EVENT_SPEED = 0.3  # px/step; below this turns are noise
CONTACT_RADIUS_PX = 12.0  # ball-paddle distance to credit a hit
WALL_PROX_PX = 6.0  # distance to court y-extreme for wall bounces
MIN_FLIGHT_FRAMES = 5  # shortest usable free-flight segment
SERVE_GAP_FRAMES = 5  # ball absence that separates points
PADDLE_BOUND_MARGIN_PX = 2.0  # exclude steps pinned at bounds

# calibrated sim geometry (px @ 30 FPS); see MEASUREMENTS.md
WALL_Y_TOP = 21.5
WALL_Y_BOT = 79.5
SCORE_X_LEFT = 9.5
SCORE_X_RIGHT = 163.5
LEFT_PADDLE_X = (21.0, 82.0)
RIGHT_PADDLE_X = (85.0, 147.0)
PADDLE_Y_RANGE = (24.0, 78.0)
COURT_MID_X = (SCORE_X_LEFT + SCORE_X_RIGHT) / 2.0  # 86.5

# no gravity (measured); drag = median per-step speed decay
BALL_RADIUS = 3.0
# 0.998 not measured 0.982 (noise-dominated, would stall serves)
BALL_DRAG = 0.998
# elastic; wall-bounce data too noisy, verified by replay-compare
WALL_RESTITUTION = 1.0

# measured per held step, bound-filtered
PADDLE_V_SPEED = 0.82
PADDLE_H_SPEED = 0.70
# 0.0: extractor angle can't encode tilt, so rotation is off
PADDLE_ROT_SPEED = 0.0
# measured 0.142 rad/step; used only by the rotation experiment
ROTATION_SPEED = 0.14  # rad/step when rotation is enabled (measured)
PADDLE_ANGLE_RANGE = (1.0, 2.14)  # rad, around vertical (pi/2)
PADDLE_LEN = 12.0
PADDLE_WIDTH = 6.5

# v39 form out = a*incoming + b*paddle; residual absorbed by DR
HIT_INCOMING_COEF = 0.55
HIT_PADDLE_COEF = 1.05
HIT_SPEED_MIN = 0.6
HIT_SPEED_MAX = 3.7

# measured Unity slow-ball boost; used only by the impulse experiment
SLOW_IMPULSE_THRESHOLD = 0.95  # incoming speed below this = slow
SLOW_IMPULSE_FLOOR = 1.0  # min outgoing of a (weakly) driven slow hit
SLOW_IMPULSE_GAIN = 1.3  # extra outgoing speed per unit push_quality
SLOW_IMPULSE_CAP = 2.3  # max outgoing speed from the impulse (~p95)

# Serve: ball launched from near a paddle toward the opponent.
SERVE_SPEED_RANGE = (0.9, 1.7)

# level-5 jitter; drag/restitution clamped <=1.0 (>1 unplayable)
DR_DRAG_RANGE = (0.996, 1.0)
DR_RESTITUTION_RANGE = (0.92, 1.0)
DR_SPEED_FRAC = 0.12

# episode cap fits the return target even at the slowest level
MAX_EPISODE_STEPS = 900  # 30 s hard cap
OWN_SIDE_TIMEOUT_STEPS = 150  # 5 s on own side -> fail (real rule)
SELF_START_PROB = 0.5  # else launcher/returner serves at the agent
TRIAL_RETURN_TARGET = 5  # returns in an episode = trial success

# a return only counts past mid; structural anti-camping rule
RETURN_CLEAR_MARGIN_PX = 4.0

# past-center margin so an off-center rally doesn't trip a stall
COMPETITIVE_STALL_DEADZONE_PX = 20.0

# Curriculum gates.
EVAL_TRIALS = 50
PASS_SUCCESS_RATE = 0.80
PER_SIDE_PASS_RATE = 0.75
MAX_LEVEL = 5

# Reward terms (event-based, lean; no per-step shaping).
RETURN_REWARD = 1.0
FAIL_REWARD = -1.0
CONTACT_BONUS = 0.2  # one-shot per incoming ball
CONTACT_ANNEAL_LEVEL = 3  # contact bonus reaches 0 at this level

# potential-based ball-advance shaping; farm-resistant
BALL_ADVANCE_COEF = 0.5

# PPO (SB3) defaults.
NUM_ENVS = 8
PPO_LR = 3e-4
PPO_N_STEPS = 1024
PPO_BATCH = 256
PPO_EPOCHS = 8
PPO_GAMMA = 0.995
PPO_GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_ENT_COEF = 0.01
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_NET_ARCH = (128, 128)
TRAIN_INTERVAL = 50_000
MAX_TOTAL_STEPS = 2_000_000
SEED = 42

# Checkpointing.
CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"
TOP_K_MODELS = 3
# ineligible if |left_rate - right_rate| exceeds this
MAX_SIDE_GAP = 0.15
