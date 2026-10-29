"""实验标注可视化入口；所有数据和媒体端点只读，无处理任务副作用。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from experiments.nvidia_v2d.policy_view_data import PolicyEpisode, PolicyStore, Scene
from experiments.nvidia_v2d.vlm_view_data import ComparisonStore, RawCallView, Snapshot, ViewerConfig


def create_viewer(config: ViewerConfig, dist: Path | None = None) -> FastAPI:
    store = ComparisonStore(config)
    app = FastAPI(title="实验标注对照", docs_url=None, redoc_url=None)
    policy = PolicyStore(config.policy_index) if config.policy_index else None

    @app.get("/api/compare/policy/{key}", response_model=PolicyEpisode)
    def policy_episode(key: str) -> PolicyEpisode:
        try:
            return policy.episode(key) if policy else PolicyEpisode()
        except (OSError, ValueError):
            raise HTTPException(503, "策略结果正在更新") from None

    @app.get("/api/compare/policy-scene/{key}/{backend}", response_model=Scene)
    def policy_scene(key: str, backend: str) -> Scene:
        try:
            if policy is None:
                raise KeyError(key)
            return policy.scene(key, backend)
        except (KeyError, OSError, ValueError):
            raise HTTPException(404, "场景尚未发布或校验失败") from None

    @app.get("/api/compare", response_model=Snapshot)
    def snapshot() -> Snapshot:
        try:
            return store.snapshot()
        except (OSError, ValueError):
            raise HTTPException(503, "结果正在更新，请稍后刷新") from None

    @app.get("/api/compare/media/{key}")
    def media(key: str) -> FileResponse:
        try:
            return FileResponse(store.media(key), media_type="video/mp4")
        except (KeyError, OSError, ValueError):
            raise HTTPException(404, "视频不存在") from None

    @app.get("/api/compare/calls/{key}/{model_id}", response_model=list[RawCallView])
    def calls(key: str, model_id: str) -> list[RawCallView]:
        try:
            return store.calls(key, model_id)
        except KeyError:
            raise HTTPException(404, "视频或模型不存在") from None
        except (OSError, ValueError):
            raise HTTPException(503, "响应正在更新，请稍后刷新") from None

    if dist is not None:
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(dist / "vlm-compare.html")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30372)
    args = parser.parse_args()
    config = ViewerConfig.model_validate(json.loads(args.config.read_text()))
    import uvicorn

    uvicorn.run(create_viewer(config, args.dist), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
