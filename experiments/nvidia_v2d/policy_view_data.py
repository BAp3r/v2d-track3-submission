"""实验策略对照只读数据；运动学参考与真实策略执行始终分开标记。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, StrictBool, StrictInt, model_validator

from experiments.nvidia_v2d.vlm_view_data import confined_file

Vec3 = tuple[float, float, float]
Kind = Literal["kinematic_reference", "policy_rollout"]
Assistance = Literal["none", "assisted", "unknown"]


class HandScene(BaseModel):
    side: Literal["left", "right"]
    points: list[list[Vec3] | None]
    edges: list[tuple[int, int]]
    valid_frames: list[StrictBool | None] | None = None
    source_codes: list[StrictInt | None] | None = None


class ObjectScene(BaseModel):
    name: str
    vertices: list[Vec3]
    faces: list[tuple[int, int, int]]
    poses: list[list[float]]


class Scene(BaseModel):
    schema_version: Literal["nvidia_v2d.policy_scene.v1"] = "nvidia_v2d.policy_scene.v1"
    kind: Kind
    assistance: Assistance = "unknown"
    unit: Literal["meter"] = "meter"
    timestamps_s: list[float] = Field(min_length=1, max_length=10000)
    hands: list[HandScene]
    objects: list[ObjectScene]
    valid_frames: list[StrictBool | None] | None = None

    @model_validator(mode="after")
    def consistent_frames(self) -> Scene:
        times = self.timestamps_s
        if self.valid_frames is not None and len(self.valid_frames) != len(times):
            raise ValueError("有效性标记与时间轴长度不符")
        if any(not math.isfinite(x) or x < 0 for x in times):
            raise ValueError("时间戳必须有限且非负")
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("时间戳必须严格递增")
        for hand in self.hands:
            if len(hand.points) != len(times):
                raise ValueError("手部与时间轴长度不符")
            for quality in (hand.valid_frames, hand.source_codes):
                if quality is not None and len(quality) != len(times):
                    raise ValueError("逐手质量标记与时间轴长度不符")
            if hand.source_codes is not None and any(
                code is not None and code not in range(5) for code in hand.source_codes
            ):
                raise ValueError("逐手来源编码无效")
            present = [points for points in hand.points if points is not None]
            count = len(present[0]) if present else 0
            if any(not points or len(points) != count for points in present):
                raise ValueError("手部拓扑在帧间变化")
            if any(i < 0 or (present and i >= count) for edge in hand.edges for i in edge):
                raise ValueError("手部骨架索引越界")
            if any(not math.isfinite(v) for frame in present for xyz in frame for v in xyz):
                raise ValueError("手部包含非有限数值")
            if hand.valid_frames is not None and any(
                valid is True and points is None
                for valid, points in zip(hand.valid_frames, hand.points, strict=True)
            ):
                raise ValueError("有效手帧必须包含有限几何")
        for obj in self.objects:
            if len(obj.poses) != len(times) or any(len(pose) != 16 for pose in obj.poses):
                raise ValueError("物体变换必须与时间轴一致")
            if any(not math.isfinite(v) for pose in obj.poses for v in pose):
                raise ValueError("物体变换非有限")
            if any(not math.isfinite(v) for xyz in obj.vertices for v in xyz):
                raise ValueError("物体网格非有限")
            if any(i < 0 or i >= len(obj.vertices) for face in obj.faces for i in face):
                raise ValueError("物体面索引越界")
        return self


class PolicyVariant(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    status: Literal["pending", "running", "succeeded", "failed", "blocked", "timed_out"]
    kind: Kind = "kinematic_reference"
    assistance: Assistance = "unknown"
    backend: Literal["official", "multitip"] | None = None
    seed: int | None = Field(default=None, ge=0)
    scene_url: str | None = None
    sha256: str | None = None
    failure_reason: str | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)


class PolicyEpisode(BaseModel):
    variants: list[PolicyVariant] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PolicyStore:
    def __init__(self, index: Path):
        self.index = index

    def _entry(self, episode: str) -> dict:
        if not self.index.exists():
            return {}
        path = confined_file(self.index, self.index.parent)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("索引过大")
        return dict(json.loads(path.read_text()).get("episodes", {}).get(episode, {}))

    def episode(self, episode: str) -> PolicyEpisode:
        entry = self._entry(episode)
        variants = []
        for raw in entry.get("variants", []):
            variant = PolicyVariant.model_validate(raw)
            if raw.get("scene_path") and variant.status == "succeeded":
                variant.scene_url = f"/api/compare/policy-scene/{episode}/{variant.id}"
            else:
                variant.scene_url = None
            variants.append(variant)
        return PolicyEpisode(variants=variants, notes=entry.get("notes", []))

    def scene(self, episode: str, backend: str) -> Scene:
        raw = next((r for r in self._entry(episode).get("variants", []) if r["id"] == backend), None)
        if raw is None or raw.get("status") != "succeeded" or not raw.get("scene_path"):
            raise KeyError("尚无可显示场景")
        relative = Path(raw["scene_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("场景路径越界")
        path = confined_file(self.index.parent / relative, self.index.parent)
        if path.stat().st_size > 24 * 1024**2:
            raise ValueError("场景超过显示上限")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != raw.get("sha256"):
            raise ValueError("场景SHA不一致")
        result = Scene.model_validate_json(content)
        if result.kind != raw.get("kind", "kinematic_reference"):
            raise ValueError("场景类型与索引不一致")
        return result
