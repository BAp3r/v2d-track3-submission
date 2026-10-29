# 训练与评估曲线记录

`policy_observability.py` 是独立 CPU sidecar。从已有 `train-metrics.jsonl` 与
`evaluation-metrics.jsonl` 回填，然后持续追踪追加行，同时写入 SwanLab 与 TensorBoard。
原始 JSONL、checkpoint 和训练参数保持原样。新训练默认通过 `policy_logged_launch.py`
完成依赖预检、ready 握手、运行中存活检查和退出后的 drain。

## 独立环境与命令

在独立虚拟环境安装 `runtime/requirements-observability.txt`，在仓库根目录执行。
本轮验证版本为 SwanLab 0.10.3、swanboard 0.1.10b2、tensorboardX 2.6.5、TensorBoard 2.21.0。
`OBS_PYTHON` 指向该环境的 Python，`RUN` 指向单次训练输出目录。

```bash
"$OBS_PYTHON" -m experiments.nvidia_v2d.policy_observability check --mode local
"$OBS_PYTHON" -m experiments.nvidia_v2d.policy_observability observe \
  --run-dir "$RUN" --output-dir "$RUN/observability" \
  --status-file "$RUN/observability-status.json" \
  --ready-file "$RUN/observability-ready.json" \
  --stop-file "$RUN/execution.json" \
  --swanlog-root "$OBS_ROOT/swanlog" \
  --tensorboard-root "$OBS_ROOT/tensorboard" \
  --swanlab-url http://127.0.0.1:5092 \
  --tensorboard-url http://127.0.0.1:6006 --mode local
```

历史导入加 `--once`。再次导入使用新的 `--output-dir` 和 `--status-file`，已有状态文件
拒绝覆盖。多个训练可共用 `--swanlog-root` 和 `--tensorboard-root`，每次导入使用独立 run。
默认项目 `sharpa-unassisted-repair`；`--mode online` 请求私有项目，使用 SDK 自身已有认证。
sidecar 不读取、写入或打印凭据。`check` 只验证依赖；在线认证与 run 创建在第一条指标时发生，
失败即退出，尚未在本机验证云端上传。

训练的不同 `session_id` 分别建立 run，避免恢复训练的重叠 step 混在一起；评估按
`context.series_id` 独立建立 run，允许在训练完成后回填较早 checkpoint 的评估 step。
训练与评估的横轴均为原始 `environment_steps`。TensorBoard 使用原始 `wall_time_unix`；
SwanLab SDK 不提供覆盖事件时间的接口，原始时间另存为
`observability/source_wall_time_unix`，SwanLab 原生事件时间是导入时间。

预热时上游会记录 loss=0，但此时没有优化器更新，不能解释为零误差或已经收敛。
需同时查看 `replay/size_transitions` 与本轮 `replay_minimum`。例如 006 的 fresh replay
最小值为131072，前513次采集未更新，第514次才开始；真实更新计数另由 checkpoint 审计核对。

## 完整性与退出

ready 代表依赖和文件追踪已准备，SDK run 在第一条指标时初始化。发现 `execution.json`
后继续排空，默认至少两秒无新增行后关闭。`status.json` 或指定状态文件保留行数、标量数、
首末 step、SDK run ID、目录、URL 和错误。worker 日志在 sidecar 输出目录。

正常关闭会重新读取 SwanLab datastore 与 TensorBoard event，逐指标校验全部 step 和值的
SHA256、计数；TensorBoard 还校验原始 walltime。TensorBoard scalar 存储为 float32，
按实际格式比较。`persistence_verification.all_scalars_verified=true` 表示本地存储完整。
SwanLab 看板可能抽样显示图表，原始存储不抽样；本地验证不代表云端已经独立读回验证。

非有限值不会替换为零：保留 `nonfinite-metrics.jsonl`、TensorBoard 文本以及计数指标，
最终状态为 `completed_with_nonfinite`，退出码为 2。截断/替换源文件、损坏 JSONL、
同一 session 内重复或逆序 step、SDK 错误、持久化不一致均明确失败。

## 本地 SwanLab 看板兼容修复

SwanLab 第一个 local run 初始化后才创建 `swanlog/runs.swanlab`。若在此之前启动
`swanlab watch`，服务会继续运行但未绑定数据库，页面报 3500；DB 已创建后重启该服务。

swanboard 0.1.10b2 的项目摘要按显示顺序 `tag.sort` 读取目录，而当前 SDK 的指标目录
由 `tag.folder` 指定。结果为第一项 TypeError、后续摘要错位；单指标图表使用 folder，
原始指标完整。以下修复只修改独立虚拟环境的摘要读取函数，原 SHA 不匹配时拒绝修改。

```bash
"$OBS_PYTHON" -m experiments.nvidia_v2d.runtime.patch_swanboard_summary \
  --receipt "$OBS_ROOT/swanboard-summary-patch.json"
```

完成后重启该环境的 SwanLab 看板。脚本幂等，收据包含完整补丁及原/新 SHA256；不修改
数据库、指标或 frozen observer/launcher 源码。

- 原 `controller/project.py` SHA256：`b0d94c724bee6f36bdc3a0840577c86c71645fc98b96aa8b28302c959acc222c`。
- 修复后 SHA256：`d262ce3befeaa4564f3fc0145881f1fc8405ea38163ba7e6350a14343448e2cd`。
