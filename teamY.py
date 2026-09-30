"""Competition submission: right paddle.

Runs teamX's policy mirrored into the left frame.
"""

from teamX import _Policy


class TeamY(_Policy):
    """Competition entry point: right paddle."""

    def __init__(self) -> None:
        super().__init__(side="right", mirror_obs=True)
