"""Clean-slate dPickleBall RL training framework.

Staged pipeline: Unity measurement -> sim calibration -> sim
curriculum pretrain -> Unity fine-tune -> self-play -> deployment.
"""
