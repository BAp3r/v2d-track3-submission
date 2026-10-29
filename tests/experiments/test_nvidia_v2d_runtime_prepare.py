import hashlib
import io
from pathlib import Path

import pytest

from experiments.nvidia_v2d.runtime.prepare_sharpa import (
    UPSTREAM_COMMIT,
    archive_command,
    download_verified,
    git_environment,
    new_root,
    parse_pointer,
    prepare,
)


def test_lfs_hash_failure_never_publishes_or_replaces_asset(tmp_path):
    destination = tmp_path / "asset"
    with pytest.raises(ValueError, match="SHA256"):
        download_verified(
            "https://example.invalid",
            destination,
            "0" * 64,
            5,
            opener=lambda *_args, **_kwargs: io.BytesIO(b"wrong"),
        )
    assert not destination.exists()
    assert destination.with_suffix(".download").read_bytes() == b"wrong"


def test_existing_directory_rejected_before_any_git_or_network(tmp_path):
    sentinel = tmp_path / "do-not-change"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        prepare(tmp_path, Path("/does/not/exist"))
    assert sentinel.read_text() == "original"
    assert not (tmp_path / "runtime-source-receipt.json").exists()


def test_existing_asset_and_symlink_root_are_rejected(tmp_path):
    destination = tmp_path / "asset"
    destination.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        download_verified("https://example.invalid", destination, "0" * 64, 5)
    assert destination.read_bytes() == b"original"
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        new_root(alias / "child")


def test_verified_asset_requires_both_exact_size_and_hash(tmp_path):
    content = b"verified"
    digest = hashlib.sha256(content).hexdigest()
    destination = tmp_path / "asset"
    download_verified(
        "https://example.invalid",
        destination,
        digest,
        len(content),
        opener=lambda *_args, **_kwargs: io.BytesIO(content),
    )
    assert destination.read_bytes() == content
    with pytest.raises(ValueError, match="大小"):
        download_verified(
            "https://example.invalid",
            tmp_path / "oversize",
            digest,
            len(content) - 1,
            opener=lambda *_args, **_kwargs: io.BytesIO(content),
        )


def test_snapshot_disables_lfs_process_and_smudge_and_pins_revision():
    argv = archive_command(Path("/read-only-upstream"))
    assert argv[-1] == UPSTREAM_COMMIT
    for option in ("filter.lfs.process=", "filter.lfs.smudge=", "filter.lfs.required=false"):
        assert option in argv
    assert git_environment()["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert git_environment()["GIT_OPTIONAL_LOCKS"] == "0"


def test_nonpointer_existing_asset_cannot_be_materialized():
    with pytest.raises(ValueError, match="拒绝覆盖"):
        parse_pointer(b"existing mesh data")
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 12\n"
    assert parse_pointer(pointer) == ("a" * 64, 12)
