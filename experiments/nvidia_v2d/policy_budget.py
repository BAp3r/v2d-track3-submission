"""冻结成对实验预算并按完整课程比例缩短训练；不修改上游配置。"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    total_steps: int = Field(gt=0, strict=True)
    world_count: int = Field(gt=0, le=4096, strict=True)
    wall_limit_s: int = Field(default=7200, gt=0, le=7200, strict=True)
    seeds: tuple[int, ...] = (42, 43, 44)
    motion_speed: float = Field(default=1.0, ge=1, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def complete_collections(self) -> TrainingBudget:
        if self.total_steps % self.world_count:
            raise ValueError("训练步数必须是完整环境采集批次")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("种子必须非空且不重复")
        return self


def scaled_curriculum(
    thresholds: list[int], voc_scales: list[float], *, total_steps: int, quantum: int
) -> list[int]:
    """保留各阶段比例；至少40%采集预算的课程目标VOC为0，不证明重置扶持关闭。"""
    if (
        not thresholds
        or len(thresholds) != len(voc_scales)
        or quantum < 1
        or total_steps % quantum
        or total_steps < (len(thresholds) + 1) * quantum
        or any(b <= a for a, b in zip([0, *thresholds[:-1]], thresholds, strict=True))
        or any(not math.isfinite(x) or not 0 <= x <= 1 for x in voc_scales)
    ):
        raise ValueError("课程必须有递增阈值和足够的完整采集批次")
    zero_index = next((i for i, scale in enumerate(voc_scales) if scale == 0), None)
    if zero_index is None or any(x != 0 for x in voc_scales[zero_index:]):
        raise ValueError("课程末段的目标VOC必须持续为0")
    result = [
        max(quantum, round(t / thresholds[-1] * total_steps / quantum) * quantum) for t in thresholds
    ]
    if any(b <= a for a, b in zip([0, *result[:-1]], result, strict=True)):
        raise ValueError("预算过小，缩放后课程阶段重叠")
    # FixedCurriculum uses bisect_right: stage i begins at threshold i-1.
    zero_start = result[zero_index - 1] if zero_index else 0
    if total_steps - zero_start < 0.4 * total_steps:
        raise ValueError("目标VOC为0的课程预算不足40%")
    return result


def training_overrides(
    budget: TrainingBudget,
    *,
    seed: int,
    parquet: str,
    output: str,
    thresholds: list[int],
    voc_scales: list[float],
) -> list[str]:
    """两后端共享这些覆盖项；路径为单个argv而非shell拼接。"""
    if seed not in budget.seeds:
        raise ValueError("种子不在冻结实验设计中")
    if any(c in parquet + output for c in "'\n\r"):
        raise ValueError("Hydra路径不能含引号或换行")
    schedule = scaled_curriculum(
        thresholds, voc_scales, total_steps=budget.total_steps, quantum=budget.world_count
    )
    interval = max(
        budget.world_count, budget.total_steps // 10 // budget.world_count * budget.world_count
    )
    return [
        "experiment=sharpa_flash_sac",
        f"task.parquet='{parquet}'",
        "task.motion_speed=1.0",
        "task.motion_start_frame=0",
        "task.motion_end_frame=-1",
        f"scene.world_count={budget.world_count}",
        f"reset.seed={seed}",
        f"training.total_environment_steps={budget.total_steps}",
        f"curriculum.thresholds={schedule}",
        f"training.checkpoint.save_interval_environment_steps={interval}",
        "logging.mode=disabled",
        f"output_dir='{output}'",
        f"hydra.run.dir='{output}/hydra'",
        "hydra.output_subdir=.hydra",
    ]
