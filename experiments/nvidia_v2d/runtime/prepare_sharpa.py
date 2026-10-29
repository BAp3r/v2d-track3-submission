"""Snapshot fixed official sources and verify selected LFS assets into a NEW root.

Only the Python standard library and Git are required. This does not build/run
containers, start training, or alter the supplied source checkout or its cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "7eb9721088615fe9201634a6be5cf48b049d1090"
UPSTREAM_URL = "https://github.com/nvidia-isaac/video_to_data.git"
LOCK_SHA256 = "91f31dd1a6f72f8d2eadcf6d321d99f826e19db975adaac93c5364641be3d767"
ASSET_BASE = "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/"
MESH_ROOTS = (
    "robotic_grounding/flash_chord/src/flash_chord/assets/meshes/sharpawave/",
    ASSET_BASE + "meshes/sharpa_wave/",
)
FIXTURE_ROOT = ASSET_BASE + "human_motion_data/ego_recon/"
FIXTURE_FILES = {
    "reconstructed_stage/tissue_box_simple_support.usda",
    "processed/tissue_box_simple_visual.stl",
    "processed/tissue_box_simple_material_0.png",
    "processed/tissue_box_simple.mtl",
    "processed/tissue_box_simple.obj",
    "processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave/77ed746eca2449aeb395e51151f26d3c-0.parquet",
}
LFS_POINTER = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_environment() -> dict[str, str]:
    return {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_OPTIONAL_LOCKS": "0"}


def archive_command(repository: Path, revision: str = UPSTREAM_COMMIT) -> list[str]:
    return [
        "git",
        "-C",
        str(repository),
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.required=false",
        "archive",
        "--format=tar",
        revision,
    ]


def new_root(destination: Path) -> Path:
    """Never resume into or overwrite existing directories, including symlinks."""
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("输出目录不能经过符号链接")
    destination.mkdir(parents=True, exist_ok=False)
    return destination.resolve()


def parse_pointer(data: bytes) -> tuple[str, int]:
    match = LFS_POINTER.fullmatch(data)
    if match is None:
        raise ValueError("资产不是完整 Git LFS 指针，拒绝覆盖已有资产")
    return match[1].decode(), int(match[2])


def download_verified(
    url: str,
    destination: Path,
    expected_sha256: str,
    size: int,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    """Publish only verified bytes; retain failed partial download for diagnosis."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("已有资产不覆盖")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or not 0 < size < 1024**3:
        raise ValueError("资产校验信息无效")
    temporary = destination.with_suffix(".download")
    digest, count = hashlib.sha256(), 0
    with temporary.open("xb") as target, opener(url, timeout=90) as response:
        while block := response.read(min(1024 * 1024, size - count + 1)):
            target.write(block)
            digest.update(block)
            count += len(block)
            if count > size:
                raise ValueError("Git LFS 下载大小不符")
    if count != size or digest.hexdigest() != expected_sha256:
        raise ValueError("Git LFS 下载大小或 SHA256 不符")
    # Hard-link publication is atomic and fails rather than replacing a new file.
    os.link(temporary, destination)


def collect_plan(source: Path) -> list[dict[str, Any]]:
    candidates = [
        path for prefix in MESH_ROOTS for path in (source / prefix).rglob("*") if path.is_file()
    ]
    candidates.extend(source / FIXTURE_ROOT / relative for relative in FIXTURE_FILES)
    plan = []
    for path in sorted(candidates):
        if path.is_symlink() or not path.is_file():
            raise ValueError("资产候选缺失或为符号链接")
        data = path.read_bytes()
        if not data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            continue
        digest, size = parse_pointer(data)
        plan.append({"path": path.relative_to(source).as_posix(), "sha256": digest, "size": size})
    if len(plan) != 464 or len({item["sha256"] for item in plan}) != 266:
        raise ValueError("固定提交的 Sharpa/tissue_box_simple 资产闭包数量不符")
    return plan


def prepare(destination: Path, repository: Path | None = None) -> dict[str, Any]:
    root = new_root(destination)
    receipt: dict[str, Any] = {
        "status": "running",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_url": UPSTREAM_URL,
        "source_checkout_modified": False,
        "download_method": "官方 GitHub LFS；下载板仅支持 HuggingFace/ModelScope，不支持该来源",
        "root": str(root),
        "failure_reason": None,
    }
    try:
        if repository is None:
            repository = root / "source-repository.git"
            subprocess.run(
                ["git", "init", "--bare", str(repository)], check=True, env=git_environment()
            )
            subprocess.run(
                ["git", "-C", str(repository), "fetch", "--depth=1", UPSTREAM_URL, UPSTREAM_COMMIT],
                check=True,
                env=git_environment(),
            )
        resolved = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", f"{UPSTREAM_COMMIT}^{{commit}}"],
            env=git_environment(),
            text=True,
        ).strip()
        if resolved != UPSTREAM_COMMIT:
            raise ValueError("上游固定提交未找到")
        archive = root / "source.tar"
        with archive.open("xb") as stream:
            subprocess.run(
                archive_command(repository), stdout=stream, check=True, env=git_environment()
            )
        source = root / "upstream"
        source.mkdir()
        with tarfile.open(archive) as contents:
            contents.extractall(source, filter="data")
        lock = source / "robotic_grounding/flash_chord/uv.lock"
        if sha256(lock) != LOCK_SHA256:
            raise ValueError("官方 uv.lock SHA256 不符")
        plan = collect_plan(source)
        cache = root / "lfs-objects"
        cache.mkdir()
        for item in plan:
            cached = cache / item["sha256"]
            if not cached.exists():
                url = (
                    "https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/"
                    + UPSTREAM_COMMIT
                    + "/"
                    + urllib.parse.quote(item["path"], safe="/=")
                )
                download_verified(url, cached, item["sha256"], item["size"])
            if sha256(cached) != item["sha256"] or cached.stat().st_size != item["size"]:
                raise ValueError("新目录中的 LFS 缓存校验不符")
        for item in plan:
            target = source / item["path"]
            if parse_pointer(target.read_bytes()) != (item["sha256"], item["size"]):
                raise ValueError("新快照资产已改变，拒绝覆盖")
            target.write_bytes((cache / item["sha256"]).read_bytes())
        receipt.update(
            status="succeeded",
            source_archive_sha256=sha256(archive),
            official_uv_lock_sha256=LOCK_SHA256,
            files=plan,
            verified_file_count=len(plan),
            unique_lfs_objects=266,
        )
    except Exception as error:
        receipt.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        (root / "runtime-source-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--upstream", type=Path, help="已有上游仓库，只读使用；省略则在新目录建立裸仓库"
    )
    args = parser.parse_args()
    result = prepare(args.destination, args.upstream)
    print(json.dumps({"status": result["status"], "root": result["root"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
