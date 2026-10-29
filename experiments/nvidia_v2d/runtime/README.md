# Sharpa / FlashCHORD 独立环境重建

这里只保存必要源码和版本参数，不保存模型、生成资产、容器镜像、下载缓存或凭据。运行需要新的输出目录，建议放在 `/var/tmp` 等空间充足的磁盘。不会连接实体手，也不会自动启动训练。

## 固定来源

- 官方上游：`https://github.com/nvidia-isaac/video_to_data.git`。
- 提交：`7eb9721088615fe9201634a6be5cf48b049d1090`。
- 官方 FlashCHORD `uv.lock` SHA256：`91f31dd1a6f72f8d2eadcf6d321d99f826e19db975adaac93c5364641be3d767`，不改锁文件。
- 资产从该提交内的 Git LFS 指针取得大小与 SHA256，再从官方 GitHub LFS 内容地址下载，逐个验证后物化。范围为两处 Sharpa mesh、官方 tissue_box_simple 的六个 LFS 文件；固定闭包为 464 文件、266 个唯一对象。URDF/MJCF 等普通源文件随提交快照保留。
- 下载板目前仅支持 HuggingFace / ModelScope，不能承担 GitHub LFS、Docker 与 PyPI 下载；该脚本仅使用上述官方 GitHub 来源，不读取令牌。

## 只读快照与资产

Python 3.11.8 以上及 Git 即可执行，无 Python 第三方依赖。以下命令从实验分支仓库根目录执行，将 `NEW_ROOT` 改为尚不存在的绝对目录：

```bash
python -m experiments.nvidia_v2d.runtime.prepare_sharpa \
  --destination NEW_ROOT \
  --upstream /absolute/path/to/existing/video_to_data
```

省略 `--upstream` 时，在新目录创建裸仓库并只拉取固定提交。提供已有仓库时只读使用；快照同时设置 `GIT_LFS_SKIP_SMUDGE=1`、禁用 `filter.lfs.process` / `filter.lfs.smudge`、设置 `filter.lfs.required=false`，避免往旧仓库的 LFS 缓存写入。没有 checkout、reset、清理或修改旧仓库的操作。

已有输出目录和资产一律拒绝覆盖。失败下载保留 `.download` 和失败收据，重试应选择另一新目录，不会自动删改已有产物。成功收据为 `NEW_ROOT/runtime-source-receipt.json`。所有 LFS 对象全部验证后，才替换本轮新快照中的对应指针。

## 官方锁容器

`Dockerfile.legacy-builder` 来自上述固定提交的官方 Dockerfile，只移除 BuildKit 专属 `RUN --mount=type=cache` 语法；安装命令与锁保持不变。本轮已验证镜像为 `fmc3-flashchord:20261027-001`，镜像 ID 为 `sha256:7fdce3ad594fbad8b5983fde7846908be2ed6179ba4ebcfd1edefbcbcf4c628c`。

```bash
docker build --target dev \
  -f experiments/nvidia_v2d/runtime/Dockerfile.legacy-builder \
  -t fmc3-flashchord:reproduced \
  NEW_ROOT/upstream/robotic_grounding/flash_chord
```

固定参数为 CUDA 基础镜像标签 `13.0.2-devel-ubuntu24.04`、uv `0.11.21`、Python `3.11.15`。官方锁中运行栈包含 FlashCHORD `0.4.0`、JAX `0.10.2`、Newton `1.2.0`、Warp `1.14.0`、MuJoCo `3.8.1`、PyArrow `24.0.0`。基础镜像使用版本标签、apt 系统包来自当时的软件源，因此不宣称跨时间构建的镜像 ID 必然逐字节相同；Python 依赖由官方 frozen lock 固定。

训练需要 NVIDIA Container Toolkit 与兼容驱动；构建不需占用 GPU。实际运行应只读挂载输入、资产、代码，单独可写挂载本轮输出和缓存。具体训练参数、预算、三种子和无辅助评估由 `policy_experiment.py` 管理，本脚本不替代训练门控。

## CPU 几何与重定向依赖

以下两个环境必须独立，使用 Python 3.11：

```bash
UV_CACHE_DIR=NEW_CACHE uv venv --python 3.11 NEW_GEOMETRY_ENV
UV_CACHE_DIR=NEW_CACHE uv pip install --python NEW_GEOMETRY_ENV/bin/python \
  -r experiments/nvidia_v2d/runtime/requirements-geometry.txt

UV_CACHE_DIR=NEW_CACHE uv --no-config venv --python 3.11 NEW_MULTITIP_ENV
UV_CACHE_DIR=NEW_CACHE uv --no-config pip install --python NEW_MULTITIP_ENV/bin/python \
  -r experiments/nvidia_v2d/runtime/requirements-retarget-runtime.txt
UV_CACHE_DIR=NEW_CACHE uv --no-config pip install --python NEW_MULTITIP_ENV/bin/python \
  -r experiments/nvidia_v2d/runtime/requirements-torch-cu124.txt \
  --index https://download.pytorch.org/whl/cu124
```

选择空间充足的新缓存目录。`--no-config` 和优先级更高的 `--index` 用于避免项目默认索引遮蔽官方 cu124 构建；不修改平台锁文件。几何清单包含 rtree 与 pytest，和重定向环境分开安装。

完整重定向依赖分成上述62项基础依赖与21项Torch依赖，全部固定本轮实际版本；历史 `requirements-multitip.txt` 和 `requirements-adapter.txt` 仍保留，不应单独当成完整环境。这里固定版本而未附全部wheel哈希，不宣称跨平台逐字节重建。

本轮最初通过 `.pth` 借用项目 Torch `2.5.1+cu124`。现已补建 `/var/tmp/fmc3-sharpa-validation-20261027-001/retarget-runtime-standalone-001`，Python 3.11.14、83项完整依赖均在该环境内，没有导入项目site-packages。19项标定/真实Sharpa测试通过；两路分别重跑000044全部249帧，输出Parquet SHA256与原002完全相同，关节、FK、腕轨迹、残差、速度和来源标记差异均为0，仅求解耗时不同。

- 验证收据：`/var/tmp/fmc3-sharpa-validation-20261027-001/verification/standalone-retarget-receipt-001.json`。
- 新输出：`/var/tmp/fmc3-sharpa-validation-20261027-001/retarget-independent/public44-{official,multitip}-001/`。
- 测试凭据：`/var/tmp/fmc3-sharpa-validation-20261027-001/verification/standalone-retarget-tests-001.xml`。

官方 `retarget/hand_kinematics.py` 直接导入Torch，`retarget/utils.py` 在导入时编译TorchScript；即使离线IK输入是NumPy，仍不能省去Torch。该路径使用CPU，本次设置 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` 验证，未占用训练GPU。保留同版cu124构建，不需要TorchVision或TorchAudio；cu124依赖约数GB，磁盘需预留缓存和环境空间。

完整平台/查看器仍使用仓库自己的 uv 项目环境。MANO 属于已有授权资产，保持只读使用，不随此目录复制；从 MANO 重建到 loaded parquet 的桥接依赖与输入凭据由桥接适配器管理。

几何审计建议设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。新快照内的机器人源资产根目录为 `NEW_ROOT/upstream/robotic_grounding/source/robotic_grounding/robotic_grounding/assets`。

CPU 回归覆盖错误 LFS hash / 大小不发布、已有目录 / 资产 / 符号链接拒绝、固定提交与禁用 smudge 参数。不会下载真实大文件或构建容器：

```bash
uv run pytest tests/experiments/test_nvidia_v2d_runtime_prepare.py
```
