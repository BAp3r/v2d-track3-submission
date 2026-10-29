# Sharpa 单样本无辅助闭环实验

本目录保存迁入 `video_to_data` 的独立实验：重建质量核查、官方/自有 multi-tip 重定向、
FlashCHORD 策略训练、真实闭环评估与执行视频。根目录 `progress.md` 是持续实验记录。
最新结果见 [零成功率定位与单样本修复](ZERO_SUCCESS_REPAIR_20261029.md)，
双看板启动与校验见 [SwanLab / TensorBoard 使用说明](OBSERVABILITY.md)。
原始视频、重建输入和历史 checkpoint 只读；所有新训练和诊断输出写入
`.e2e/zero_success_20261027/`，不得把生成资产、视频、模型或本地环境提交到 Git。

Public44 的249帧 RGB、预测深度、手质量和重建诊断输入已复制到
`.e2e/zero_success_20261027/inputs/public44-source/`；该目录的 `local-input-index.json`
保存旧路径与本地路径、逐文件 SHA-256 和时间序列完整性，`README.md` 提供本地复现命令。
历史 manifest 和 checkpoint 中的绝对路径保持原样，通过索引或只读挂载解析。

保留的源码不依赖旧 Dashboard 的 `backend` 包。与本实验无关的 Dashboard 作业适配器、
原始采集传感器导出、VLM 推理服务和 DROID 调度模块已保存到本轮输出下的
`migration-archive/`；清单记录原路径、原始副本和 SHA-256，旧项目未被修改。
`vlm_view_data.py`、`vlm_compare_audit.py` 和 `vlm_compare_viewer.py` 是策略查看器仍需的独立模块，
保留历史命名；查看器 API 可独立运行，前端构建产物通过 `--dist` 显式提供。

## 代码范围

| 模块 | 职责 |
| --- | --- |
| `sharpa_episode.py`、`sharpa_bridge*.py` | 输入、时间轴、坐标、资产与官方重定向契约 |
| `sharpa_multitip*.py`、`sharpa_calibration.py` | 自有 multi-tip 重定向和运动学校准 |
| `hand_quality*.py`、`hand_object_diagnostics.py`、`sharpa_geometry_audit.py` | 有效帧、手物关系、拟合/重定向误差 |
| `postprocess_*.py`、`depth_source_ab.py` | 身份、遮挡、深度、支撑面与固定尺度重拟合 |
| `policy_experiment*.py`、`policy_run.py`、`policy_gate.py` | 独立作业、输入绑定、成对训练与无辅助全序列评估 |
| `policy_metric_logging.py`、`policy_metric_curves.py` | 本地真实 loss、训练和评估曲线 |
| `policy_demo_capture.py`、`policy_rollout_export.py`、`policy_scene_export.py` | 策略实际执行状态、视频/场景数据；不是参考轨迹回放 |
| `runtime/` | 固定上游容器、CPU 几何和独立重定向环境说明 |

## 独立 CPU 环境与验证

Python 3.12 的本地验证依赖见 `runtime/requirements-standalone.txt`。
从仓库根目录使用 uv 建立一个新环境；已有环境可直接执行安装和测试步骤：

```bash
uv venv --python 3.12 .e2e/zero_success_20261027/venv
uv pip install --python .e2e/zero_success_20261027/venv/bin/python \
  -r experiments/nvidia_v2d/runtime/requirements-standalone.txt
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .e2e/zero_success_20261027/venv/bin/python -m pytest tests/experiments -q
```

禁用 pytest 自动加载插件，避免宿主 ROS 等环境注入无关测试依赖。
该环境验证 CPU 契约、几何算法、作业、日志和查看器。依赖真实 GPU/FlashCHORD 或
外部 Sharpa/Pinocchio 资产的测试按原测试标记跳过；CPU 测试通过不代表操作成功。
训练使用固定 FlashCHORD 镜像，重定向使用独立环境，见 [runtime/README.md](runtime/README.md)。
不把这些运行栈混入宿主轻量测试环境。

## Loss 与评估曲线

新 `policy_experiment` 作业的训练和恢复均使用本地指标入口，每个采集批次保存
官方 learner 实际输出的 actor/critic/temperature loss、跳过更新比例、奖励、误差和课程状态。
loss 是该批次内实际优化器更新的均值，不能视为每个 minibatch 的独立 loss。

`train-metrics.jsonl` 和长表 `train-metrics.csv` 每条记录均 flush/fsync；恢复验证另写
`resume-metrics.*`。启动参数和源码哈希写入关联 session JSON；显式追加恢复保留原记录和重复步数，
各 session 分开。NaN/Inf 保留类型和字段名，JSON 数值写 null。

`.loss.svg` 和 `.training.svg` 在首条记录、checkpoint 和退出时更新。
曲线流式读取并最多抽样显示约2000个真实点，首末点保留；全部逐批次记录始终保存在 JSONL/CSV。
异常结束后已落盘记录保留；强杀后可重新生成曲线：

```bash
PYTHONPATH=. .e2e/zero_success_20261027/venv/bin/python \
  -m experiments.nvidia_v2d.policy_metric_curves /absolute/path/train-metrics.jsonl
```

对单独训练命令启用同一记录器（在固定 FlashCHORD 容器内）：

```bash
python -m experiments.nvidia_v2d.policy_metric_logging \
  scripts/train_flash_sac.py /run/train-metrics.jsonl <Hydra 参数>
```

训练结束后 `policy_experiment` 按步数对每个实际保存的策略 checkpoint 运行完整无辅助闭环，
分别保存 `evaluations/<steps>/` 的评估和逐步辅助尺度证据；最终 checkpoint 保留原有
`unassisted.*` 输出位置。`evaluation-metrics.jsonl`、`.csv`、`.evaluation.svg` 只追加
通过实际 VOC/物体控制尺度核查的真实结果。每个点绑定 checkpoint、实际参考 Parquet 的 SHA-256、
评估源码、阈值、重置方式和样本数。调用者通过 `scene_context` 记录外部支撑面哈希与碰撞配置；
不同条件不混画。评估仍受总作业时限约束，没评估完的 checkpoint 不填成功率、不补插值结果。

历史六组大训练未记录的 loss 无法恢复；已有16k记录只代表那两次短测。
训练能运行、参考跟踪、输入结构合格和真实无辅助操作成功分别报告。
