"""kaggle-tools: battle-tested utilities for Kaggle competitions.

Modules are small and independent; import what you need::

    from kaggle_tools.env import competition_dir, is_competition_rerun
    from kaggle_tools.submission import ProgressiveSubmission
    from kaggle_tools.ensemble import rank_blend, hill_climb
    from kaggle_tools.cv import spatial_group_folds, rare_safe_folds
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kaggle-tools")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
