"""
Curriculum learning scheduler for gradually increasing task difficulty.
"""

import numpy as np
from typing import Optional


class CurriculumScheduler:
    """
    Time-based curriculum scheduler that progressively increases occluder density.

    The scheduler divides training into stages, with each stage having a specific
    number of trees (occluders). As training progresses, the difficulty increases
    by adding more trees to the environment.
    """

    def __init__(
        self,
        min_trees: int = 0,
        max_trees: int = 8,
        num_stages: int = 5,
        total_iterations: Optional[int] = None,
        iterations_per_stage: Optional[int] = None,
    ):
        """
        Initialize the curriculum scheduler.

        Args:
            min_trees: Minimum number of trees (easiest level)
            max_trees: Maximum number of trees (hardest level)
            num_stages: Number of curriculum stages
            total_iterations: Total training iterations (mutually exclusive with iterations_per_stage)
            iterations_per_stage: Iterations per stage (mutually exclusive with total_iterations)
        """
        assert (total_iterations is None) != (
            iterations_per_stage is None
        ), "Must specify exactly one of total_iterations or iterations_per_stage"

        self.min_trees = min_trees
        self.max_trees = max_trees
        self.num_stages = num_stages

        # Calculate tree count for each stage (linear progression)
        self.stage_trees = np.linspace(
            min_trees, max_trees, num_stages, dtype=int
        ).tolist()

        # Calculate iterations per stage
        if total_iterations is not None:
            self.iterations_per_stage = max(1, total_iterations // num_stages)
        else:
            self.iterations_per_stage = max(1, iterations_per_stage)

        self.current_iteration = 0
        self.current_stage = 0

    def get_num_trees(self) -> int:
        """Get the number of trees for the current curriculum stage."""
        return self.stage_trees[self.current_stage]

    def get_stage(self) -> int:
        """Get the current curriculum stage (0-indexed)."""
        return self.current_stage

    def get_progress(self) -> float:
        """Get curriculum progress as a fraction [0, 1]."""
        return (
            self.current_stage / (self.num_stages - 1) if self.num_stages > 1 else 1.0
        )

    def step(self) -> bool:
        """
        Advance the curriculum by one iteration.

        Returns:
            bool: True if the stage changed, False otherwise
        """
        self.current_iteration += 1

        # Calculate which stage we should be in based on iteration count
        new_stage = min(
            self.current_iteration // self.iterations_per_stage, self.num_stages - 1
        )

        stage_changed = new_stage != self.current_stage
        self.current_stage = new_stage

        return stage_changed

    def reset(self):
        """Reset the curriculum to the beginning."""
        self.current_iteration = 0
        self.current_stage = 0

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(stage={self.current_stage}/{self.num_stages-1}, "
            f"trees={self.get_num_trees()}, "
            f"iteration={self.current_iteration})"
        )

    def get_info(self) -> dict:
        """Get curriculum information as a dictionary for logging."""
        return {
            "curriculum/stage": self.current_stage,
            "curriculum/num_trees": self.get_num_trees(),
            "curriculum/progress": self.get_progress(),
            "curriculum/iteration": self.current_iteration,
        }
