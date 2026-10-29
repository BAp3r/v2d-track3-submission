"""从实际追加记录生成无依赖SVG曲线；不插值补造缺失训练或评估历史。"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_records(path: Path, *, max_plot_points: int = 2000) -> list[dict[str, Any]]:
    """流式读取并有界抽样画图；全部原始记录不变，首末点始终保留。"""
    if max_plot_points < 2:
        raise ValueError("max_plot_points must be at least 2")
    records: list[tuple[int, dict[str, Any]]] = []
    stride = 1
    last = None
    with path.open() as stream:
        for index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except ValueError:
                if line.endswith("\n"):
                    raise
                # Only a final unterminated write can be incomplete.
                continue
            last = (index, row)
            if index % stride == 0:
                records.append(last)
            if len(records) > max_plot_points:
                stride *= 2
                records = [(i, value) for i, value in records if i % stride == 0]
    if last is not None and (not records or records[-1][0] != last[0]):
        records.append(last)
    return [row for _, row in records]


def _svg(records: list[dict[str, Any]], names: list[str]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{max(100, 220 * len(names) + 45)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#263448">',
        '<text x="20" y="24">Actual samples vs environment steps; display sampled at 2000 points; full history in JSONL/CSV.</text>',
    ]
    for panel, name in enumerate(names):
        top = 50 + panel * 220
        series: dict[str, list[tuple[int, float | None]]] = defaultdict(list)
        for row in records:
            if name in row["metrics"]:
                identity = row.get("context", {}).get("series_id", row.get("session_id", "legacy"))
                series[identity].append((row["environment_steps"], row["metrics"][name]))
        finite = [(x, y) for points in series.values() for x, y in points if y is not None and math.isfinite(y)]
        parts.append(f'<text x="20" y="{top}">{html.escape(name)}</text>')
        if not finite:
            parts.append(f'<text x="70" y="{top + 80}">No finite samples; inspect raw diagnostics.</text>')
            continue
        xmin, xmax = min(x for x, _ in finite), max(x for x, _ in finite)
        ymin, ymax = min(y for _, y in finite), max(y for _, y in finite)
        dx, dy = max(1, xmax - xmin), (ymax - ymin) or 1
        for y, value in ((top + 20, ymax), (top + 160, ymin)):
            parts += [f'<text x="4" y="{y + 4}">{value:.4g}</text>', f'<path d="M 75 {y} H 920" stroke="#d8dee8"/>']
        parts += [f'<text x="75" y="{top + 180}">{xmin}</text>', f'<text x="920" y="{top + 180}" text-anchor="end">{xmax}</text>']
        for index, points in enumerate(series.values()):
            segments: list[list[str]] = [[]]
            for x, y in points:
                if y is None or not math.isfinite(y):
                    segments.append([])
                    continue
                segments[-1].append(f"{75 + 845 * (x - xmin) / dx:.2f},{top + 160 - 140 * (y - ymin) / dy:.2f}")
            color = ("#2563ad", "#ac4026", "#218260", "#794aaa")[index % 4]
            for segment in filter(None, segments):
                if len(segment) == 1:
                    x, y = segment[0].split(",")
                    parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
                else:
                    parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    return "\n".join(parts + ["</g></svg>\n"])


def export_curves(path: Path) -> list[Path]:
    records = read_records(path)
    names = sorted({name for row in records for name in row["metrics"]})
    groups = {
        "loss": [name for name in names if "loss" in name],
        "evaluation": [name for name in names if name.startswith("evaluation/")],
        "training": [name for name in names if "loss" not in name and not name.startswith("evaluation/")],
    }
    outputs = []
    for group, selected in groups.items():
        if not selected:
            continue
        destination = path.with_name(f"{path.stem}.{group}.svg")
        temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(_svg(records, selected))
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        outputs.append(destination)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps([str(path) for path in export_curves(args.metrics_jsonl)]))


if __name__ == "__main__":
    main()
