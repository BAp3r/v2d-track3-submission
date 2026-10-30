# Ego orchestration environment

Use uv for the host wrappers. ML dependencies stay in Docker images.

```bash
# From the repository root. Override this path to use another local filesystem.
export UV_CACHE_DIR=/tmp/v2d-uv-cache
export UV_PROJECT_ENVIRONMENT=/tmp/v2d-ego-venv
uv sync --project reconstruction/environments/ego --locked
cd reconstruction
uv run --project environments/ego --locked python modules/v2d_pipelines/run_ego_reconstruction.py --help
```

`uv.lock` pins the host dependencies. Editable package paths resolve from this
project directory. The explicit environment/cache paths also avoid mergerfs
installation failures observed in the default home cache and workspace venv.

Local model-layout directories may contain absolute symlinks into NAS roots.
Expose those roots read-only at their original paths inside all module containers:

```bash
export V2D_READONLY_MOUNTS=/home/fmc3/FermiBotNas/huggingface/models:/home/fmc3/FermiBotNas/models
```

Top-level ACE input symlinks are also resolved before mounting.
