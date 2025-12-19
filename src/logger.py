"""
Reusable TensorBoard logging module for training and evaluation.

This module provides a clean, reusable interface for TensorBoard logging
that can be used across training, evaluation, and other components.

Usage Examples:
    Basic usage with RunContext:
        >>> from src.logger import RunContext, TensorboardLogger
        >>> ctx = RunContext(base_dir="logs", run_name="experiment_1")
        >>> logger = TensorboardLogger.create(ctx=ctx, enabled=True)
        >>> logger.log_scalar("train/reward", 0.95, step=100)
        >>> logger.log_dict("losses", {"policy": 0.1, "value": 0.05}, step=100)
        >>> logger.close()

    Usage without RunContext (direct log directory):
        >>> logger = TensorboardLogger.create(log_dir="logs/tensorboard", enabled=True)
        >>> logger.log_scalar("eval/score", 0.8, step=1)
        >>> logger.close()

    Disabled logging (returns NoOpLogger):
        >>> logger = TensorboardLogger.create(enabled=False)
        >>> logger.log_scalar("anything", 1.0, step=1)  # Does nothing
        >>> logger.close()  # Does nothing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class Logger(Protocol):
    """Protocol defining the logger interface."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        ...

    def log_dict(self, prefix: str, values: Mapping[str, Any], step: int) -> None:
        """Log a dictionary of values with a common prefix."""
        ...

    def close(self) -> None:
        """Close the logger and flush any pending writes."""
        ...


class RunContext:
    """
    Manages run-specific directories and paths.

    This class is responsible for creating and managing directories
    for a single run, following the Single Responsibility Principle.
    """

    def __init__(
        self,
        base_dir: str | Path,
        run_name: str | None = None,
        create_subdirs: bool = True,
    ):
        """
        Initialize a run context.

        Args:
            base_dir: Base directory for all runs
            run_name: Optional name for this specific run. If None, uses timestamp.
            create_subdirs: Whether to create subdirectories immediately
        """
        self.base_dir = Path(base_dir)
        self.run_name = run_name or self._generate_run_name()
        self.run_dir = self.base_dir / self.run_name
        self.tb_run_dir = self.run_dir / "tensorboard"

        if create_subdirs:
            self.ensure_dirs()

    @staticmethod
    def _generate_run_name() -> str:
        """Generate a unique run name based on timestamp."""
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.tb_run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)


class NoOpLogger:
    """
    Null logger that does nothing.

    Used when logging is disabled or TensorBoard is not available.
    Follows the Null Object Pattern.
    """

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """No-op: does nothing."""
        pass

    def log_dict(self, prefix: str, values: Mapping[str, Any], step: int) -> None:
        """No-op: does nothing."""
        pass

    def close(self) -> None:
        """No-op: does nothing."""
        pass


class TensorboardLogger:
    """
    TensorBoard logger implementation.

    Wraps torch.utils.tensorboard.SummaryWriter with a clean interface
    that can be used across training, evaluation, and other components.
    """

    def __init__(self, writer: SummaryWriter, ctx: RunContext | None = None):
        """
        Initialize the TensorBoard logger.

        Args:
            writer: The SummaryWriter instance to use
            ctx: Optional run context for reference
        """
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard is not available. Install it with: pip install tensorboard"
            )
        self._writer = writer
        self.ctx = ctx

    @classmethod
    def create(
        cls,
        ctx: RunContext | None = None,
        log_dir: str | Path | None = None,
        enabled: bool = True,
    ) -> Logger:
        """
        Factory method to create a logger instance.

        Args:
            ctx: RunContext instance. If provided, uses ctx.tb_run_dir for logging.
            log_dir: Alternative log directory if ctx is not provided.
            enabled: Whether logging is enabled. If False, returns NoOpLogger.

        Returns:
            Logger instance (TensorboardLogger or NoOpLogger)
        """
        if not enabled:
            return NoOpLogger()

        if SummaryWriter is None:
            return NoOpLogger()

        # Determine log directory
        if ctx is not None:
            ctx.ensure_dirs()
            log_path = ctx.tb_run_dir
        elif log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("Either 'ctx' (RunContext) or 'log_dir' must be provided")

        writer = SummaryWriter(log_dir=str(log_path))
        return cls(writer, ctx)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value to TensorBoard.

        Args:
            tag: Tag name for the scalar
            value: Scalar value to log
            step: Step number (e.g., training iteration, episode number)
        """
        self._writer.add_scalar(tag, float(value), int(step))

    def log_dict(self, prefix: str, values: Mapping[str, Any], step: int) -> None:
        """
        Log a dictionary of values with a common prefix.

        Args:
            prefix: Prefix to add to all tag names
            values: Dictionary of tag -> value mappings
            step: Step number
        """
        for key, value in values.items():
            if value is None:
                continue
            try:
                # Try to convert to float, skip if not possible
                float_value = float(value)
                tag = f"{prefix}/{key}" if prefix else key
                self.log_scalar(tag, float_value, step)
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue

    def close(self) -> None:
        """Close the writer and flush any pending writes."""
        self._writer.close()
