# 2026-10-27 Sharpa 手物几何验证

自有 multi-tip 的指尖目标误差更小，但本轮物体表面邻近度没有同步改善。两条路线数值契约通过，**几何质量仍未知**；开放物体网格使穿透不可测。当前只允许明确标记的机制验证训练，不能据此宣布抓持更可靠、策略质量通过或自有路线胜出。

## 数据与审计范围

- 共同输入：`/var/tmp/fmc3-sharpa-validation-20261027-001/episodes/public44-005`。
- 两组正式结果：同一实验根目录下 `retarget/public44-official-002`、`retarget/public44-multitip-002`。
- 249 帧，20 FPS，时间戳从 0 到 12.4 秒；源视频按帧时长计 12.45 秒。
- 每侧 512 个固定、按面积采样的 Sharpa 原始 MJCF 碰撞 STL 表面点。它们不是仿真凸包接触点，不是整只手的精确最小距离。
- 使用 `preview.json` 的物体世界姿态与已米制 `mesh.obj`，不再次缩放物体、腕轨迹或手部采样点。
- 逐帧距离、来源、无效标记保留在 `geometry-audit/public44-*-002/surface-audit.npz`；输入哈希保留在对应 `manifest.json`。

## 结果

表中距离是每帧 512 个采样点到物体表面的最小值，再对全序列取中位数。5 mm 阈值仅用于邻近筛查。

| 路线 | 手 | 最近采样点距离中位数 | 5 mm 内帧数 |
|---|---|---:|---:|
| 官方 IK | 左 | 6.17 mm | 117 / 249 |
| 官方 IK | 右 | 16.22 mm | 58 / 249 |
| 自有 multi-tip | 左 | 13.66 mm | 59 / 249 |
| 自有 multi-tip | 右 | 23.03 mm | 48 / 249 |

分时间窗口的最近采样点距离中位数如下，单位为 mm：

| 时间（秒） | 官方左 | 官方右 | 自有左 | 自有右 |
|---|---:|---:|---:|---:|
| 0–2 | 28.58 | 54.16 | 28.90 | 52.66 |
| 2–4 | 7.47 | 25.81 | 14.90 | 35.12 |
| 4–6 | 1.17 | 0.60 | 7.13 | 2.46 |
| 6–8 | 0.41 | 0.67 | 0.32 | 4.32 |
| 8–10 | 2.83 | 9.43 | 9.77 | 19.19 |
| 10–12.4 | 409.64 | 370.03 | 403.69 | 374.14 |

末段两条路线都远离物体，且幅度接近。需要结合 RGB 判断是操作完成后的离开，还是手物重建/物体轨迹异常；本报告不把它直接归因于重定向。最靠近表面的采样点也可能位于错误手指或掌面，因此距离更小本身仍不证明接触更正确。

`white_pot` 原网格有 5043 个顶点、8052 个三角面，**不是封闭网格**。本轮不修补、不凸包化，穿透字段明确为未知。即使封闭网格的 512 个样本全部未穿透，也只能表示采样点未发现穿透，不能证明整个表面无穿透。接触延迟同样没有可信接触真值；首次进入 5 mm 阈值不能冒充真实接触起点。

## 6、7、8 秒同源帧与重投影复核

原始 NAS MP4 离线解码的第 120、140、160 帧 PTS 分别为 6、7、8 秒，均显示双手抓持已抬起的锅沿。三帧与既有重建对应 PNG 的 RGB 像素完全一致（MAE=0），HaMeR 帧号也一致。原视频仍为 249 帧、20 FPS，SHA256 为 `5018dbc5c724b3c9e830a771e5509f81b8e23e0bafad343071ab5e848527f030`。此前浏览器 seek 后截图显示的是滞后的已解码画面，不能用作重建错时证据；查看器已按实际呈现的视频帧同步三维视图，并修复暂停时 seek 的竞态。

逆场景变换与相机变换重投影，未发现这三个时刻的整体帧偏移或坐标错配：世界组合最大矩阵元素差约 1.2e-7，场景对齐差约 2.2e-16，两路物体轨迹与公共几何差约 1.4e-7。实际查看三张四格图后，物体投影与真实锅位置相符，但锅柄长度、局部轮廓及部分手指仍有偏差。物体投影与 SAM2 掩膜的 IoU 分别为 0.695、0.691、0.725；它们来自同一管线，不是独立真值，也不能推广为全序列质量通过。

上表“6–8 秒”是 40 帧窗口的中位数，并非 6 秒瞬时距离。三个时刻的逐帧最近采样点距离如下：

| 时刻 | 官方左 / 右 | 自有左 / 右 |
|---|---:|---:|
| 6 秒 | 8.725 / 0.223 mm | 13.707 / 0.899 mm |
| 7 秒 | 0.092 / 9.953 mm | 0.077 / 19.182 mm |
| 8 秒 | 8.752 / 3.330 mm | 17.526 / 8.659 mm |

图像及凭据：[6 秒重投影](/var/tmp/fmc3-sharpa-validation-20261027-001/geometry-reprojection-001/reprojection-000120.png)、[7 秒重投影](/var/tmp/fmc3-sharpa-validation-20261027-001/geometry-reprojection-001/reprojection-000140.png)、[8 秒重投影](/var/tmp/fmc3-sharpa-validation-20261027-001/geometry-reprojection-001/reprojection-000160.png)、[完整中文复核](/var/tmp/fmc3-sharpa-validation-20261027-001/geometry-reprojection-001/中文复核报告.md)、[数值与哈希凭据](/var/tmp/fmc3-sharpa-validation-20261027-001/geometry-reprojection-001/reprojection-receipt.json)。所有复核使用 CPU，源输入前后哈希一致。开放网格的穿透及真实接触质量仍为未知；冻结的机制验证训练不因此变更。

## 门控约定

`sharpa_geometry_audit.build_training_gate` 将数值合格和几何验收分开：

- 默认用途 `policy_quality` 不放行；有限采样结果不能自动把 `geometry_quality_accepted` 改成真。
- 显式用途 `mechanism_validation`，且上游数值契约、每帧有效性、有限值和输入不变检查全部通过，才设置 `training_ready=true`。
- 该放行始终保留 `geometry_quality_status=unknown`、`geometry_quality_accepted=false`、`policy_quality_claim_allowed=false`。
- 缺失、拒绝、外推、非有限或未完整处理的手帧，不能通过“机制验证”绕过整段监督输入检查。
- 开放网格的 `penetration_status=unknown`；封闭且朝向一致网格也只得到 `sampled_only`。

本轮两份明确限于机制验证的凭证位于：

```text
/var/tmp/fmc3-sharpa-validation-20261027-001/geometry-audit/training-gates/public44-official-002.json
/var/tmp/fmc3-sharpa-validation-20261027-001/geometry-audit/training-gates/public44-multitip-002.json
```

每份绑定共同输入、几何审计及各自重定向清单的路径与 SHA256。失效的 `multitip-001` 左手结果仍保留，不进入正式训练。训练保存、恢复、无辅助评估与导出机制可以继续验证，策略质量与参赛效果另行验收。

## 环境与复现

独立 CPU 环境：`/var/tmp/fmc3-sharpa-validation-20261027-001/geometry-runtime/venv`。通过 uv 安装并在同目录 `requirements-resolved.txt` 固定实际版本：NumPy 2.4.6、SciPy 1.17.1、trimesh 5.2.0、rtree 1.4.1。无 GPU 使用，无系统安全设置修改。

从实验分支目录执行，输出目录必须是新的：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  /var/tmp/fmc3-sharpa-validation-20261027-001/geometry-runtime/venv/bin/python \
  -m experiments.nvidia_v2d.sharpa_geometry_audit \
  --retarget-dir /var/tmp/fmc3-sharpa-validation-20261027-001/retarget/public44-official-002 \
  --output-dir /var/tmp/fmc3-sharpa-validation-20261027-001/geometry-audit/public44-official-new
```

查询按 32 点分块，限制远距离点选择大量候选三角形时的临时内存。首次未限制 BLAS 线程的两个 CPU 进程由本任务停止，其日志保留；分块、单线程重跑成功。

服务测试 16 项通过，覆盖刚体变换与米制距离、封闭网格内部点深度、开放网格未知、多物体共享坐标、质量标记、无效样本、时间轴冲突、原文件哈希不变、拒绝覆盖及训练门控。ruff 检查通过。

## 官方环境短测和输入保护

官方容器为 `fmc3-flashchord:20261027-001`，固定上游提交 `7eb9721088615fe9201634a6be5cf48b049d1090`，使用官方 uv.lock。464 个 LFS 资产文件已按官方指针校验，272 项 FlashCHORD Sharpa URDF mesh 引用完整。下载板不支持 GitHub LFS/Docker；这些资产及容器依赖来自官方对应来源。

官方 `tissue_box_simple` 已完成 16 环境、1024 步、Adam 状态审计确认 critic 生效更新 98 次、actor 生效更新 49 次，以及保存、从 512 步恢复、无辅助 260 帧完整评估。MPPE 为 6.8964 cm、ManipTrans 成功率为 0，此结果只验证运行机制。完整证据见实验根目录下 `runtime-assets/运行环境与官方短测报告.md`。无辅助实际物体轨迹是 Parquet；已保留最初扩展名错误的原件并新增同字节 `.parquet` 副本，格式与数值凭证见 `runtime-assets/smoke-002/trajectory-validation.json`。

NAS 原数据与已有实验输入只读。几何审计输入和 mesh 的前后 SHA256 相同；最终复核凭证保存于实验根目录的 `receipts`。首次上游快照操作意外触发 LFS smudge，在原上游 `.git/lfs` 对象缓存新增了部分官方对象；该例外已披露并保留记录，未修改原上游跟踪文件。后续快照禁用过滤器，最终源文件复核直接计算 Git blob 摘要，不调用 LFS 过滤器。
