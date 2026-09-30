"""Event-based reward terms for the sim curriculum.

Deliberately lean: a counted return, a fail, and a one-shot
first-contact bonus that anneals to zero by a configured level. No
per-step approach/push shaping (v39's reward-farming trap) and no
net-camping penalty stream (camping is handled structurally in the
env: returns must clear past mid, and self-start episodes give a
camper no ball to intercept).
"""

from agent import config


def contact_bonus_scale(level: int) -> float:
    """Linear anneal of the first-contact bonus, 1.0 -> 0.0.

    Reaches 0 at config.CONTACT_ANNEAL_LEVEL so later levels reward
    only genuine returns, not mere contact.
    """
    anneal = config.CONTACT_ANNEAL_LEVEL
    if level >= anneal:
        return 0.0
    return max(0.0, 1.0 - (level - 1) / max(1, anneal - 1))


def step_reward(
    *,
    counted_return: bool,
    failed: bool,
    first_contact: bool,
    level: int,
) -> float:
    """Sum the event rewards that fired this step.

    Args:
        counted_return: ball cleared past mid after an agent hit.
        failed: agent's side was scored on or timed out.
        first_contact: agent's first hit of the current incoming
            ball (bonus paid once per incoming).
        level: current curriculum level (drives contact anneal).
    """
    reward = 0.0
    if counted_return:
        reward += config.RETURN_REWARD
    if failed:
        reward += config.FAIL_REWARD
    if first_contact:
        reward += config.CONTACT_BONUS * contact_bonus_scale(level)
    return reward
