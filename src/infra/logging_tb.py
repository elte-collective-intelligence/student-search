from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol

from torch.utils.tensorboard import SummaryWriter

from src.infra.config import RunContext


class Logger(Protocol):
    run: Optional[RunContext]

    def log_scalar(self, tag: str, value: float, step: int) -> None: ...
    def log_dict(self, prefix: str, values: Mapping[str, Any], step: int) -> None: ...
    def close(self) -> None: ...


class NoOpLogger:
    run: Optional[RunContext] = None

    def log_scalar(self, *_: Any, **__: Any) -> None:
        return

    def log_dict(self, *_: Any, **__: Any) -> None:
        return

    def close(self) -> None:
        return


class TensorboardLogger:
    def __init__(self, writer: SummaryWriter, ctx: RunContext):
        self._w = writer
        self.run = ctx

    @classmethod
    def create(cls, ctx: RunContext, enabled: bool = True) -> Logger:
        if not enabled or SummaryWriter is None:
            return NoOpLogger()

        # ctx is responsible for directory creation, but safe to ensure here too.
        ctx.ensure_dirs()

        writer = SummaryWriter(log_dir=str(ctx.tb_run_dir))
        return cls(writer, ctx)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._w.add_scalar(tag, float(value), int(step))

    def log_dict(self, prefix: str, values: Mapping[str, Any], step: int) -> None:
        for k, v in values.items():
            if v is None:
                continue
            self.log_scalar(f"{prefix}/{k}", float(v), step)

    def close(self) -> None:
        self._w.close()
