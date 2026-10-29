# 2026-10-27 后处理复现与验证

下午补验已重新读取 NAS 全帧深度和 public31 原视频，并新增 public00 可见支持实际重估。
最新结果见 [NAS 恢复补全报告](NAS_RECOVERY_20261027.md)；下文的未接线/待核验表述保留此前时间点，不代表最新状态。

本轮新增了可运行的共同场景组装、身份异常诊断、遮挡观测支持修复和全时间轴深度尺度/平移重估。所有输入只读，新产物均位于 `/var/tmp/fmc3-sharpa-validation-20261027-001/`；没有使用 GPU、ICP、实体手或仿真训练。共同坐标证明、数值门控通过和真实重建质量是三件不同的事，本轮没有把前两项冒称为第三项。

## 一、000000 与 000031：可以复用同一个原始 SLAM 世界

`postprocess_scene.py` 先验证源视频哈希、帧数/FPS、逐帧 RGB 文件摘要、SLAM 轨迹及逐帧位姿摘要、内参，再核对每个物体的 `camera_to_world @ object_to_camera == object_to_world`。相机数组必须逐元素一致；任何来源不同或复合不一致都会拒绝组装。输出直接保留原始世界，绝不独立居中，也不以 ICP 猜测对齐。

| 样本 | 物体 | 帧数 / 时间 | 最大坐标复合误差 | 结果 |
|---|---|---|---|---|
| public 000000 | 白锅、锅盖 | 373 / 20 FPS | 1.1921e-7 | 共同坐标证明通过，供审查 |
| public 000031 | 小刷子、沥水架 | 290 / 20 FPS | 1.1921e-7 | 共同坐标证明通过，供审查 |

产物：

- `postprocess-public00-scene-001/{input.json,manifest.json,scene.npz,assets/}`。
- `postprocess-public31-scene-001/{input.json,manifest.json,scene.npz,assets/}`。

`scene.npz` 保存共同相机、按对象排列的刚体变换、原始帧索引/时间、逐帧有效性和显式指定来源的手部参数。网格已经包含原重建的物体尺度，不再二次缩放。

两套对象任务中的手部输出并不完全相同。000000 左手平移最大差约 7.49 mm；000031 两套右手手指轴角参数最大差约 0.58 rad。因此本次分别显式选择白锅任务、小刷子任务作为唯一手部来源，保留其它来源逐字段差异，未混合、平均或隐式挑选。这个选择只定义审查数据版本，不证明该套手更准确。

000000 仍保留左手拒绝 13 帧、右手拒绝 2 帧和插值 3 帧。对象跟踪输出缺少足够的“真实观测/遮挡估计”来源证据；二者均未获完整训练许可。

## 二、000000：当前证据不支持互换标签，修复的是遮挡污染监督

`postprocess_identity.py` 对原传播掩膜执行只读检查：覆盖、连通分量、面积突变及相邻帧跨标签质心匹配。共 126 帧的两目标交集超过较小目标面积的 25%，集中在 **0–34** 和 **282–372** 帧；未出现满足预设阈值的质心交换候选。

实际查看第 20、35、90、281、282、345 帧 RGB 和两目标轮廓后，高交叠主要对应锅盖真实盖在锅上。第 90 帧分离时，两套轮廓仍分别对应锅和盖。这个检查不是全面身份真值，也不能排除其它三维跟踪问题；它不支持为了“完成身份修复”而硬交换标签。

诊断记录：`postprocess-public00-identity-001/manifest.json`。

中文图：`postprocess-public00-identity-001/figures/锅与锅盖身份诊断_v2.png`。

### 保守的可见深度支持

`postprocess_occlusion.py` 生成新观测支持：

- 多目标重叠像素标为 `ambiguous`。
- 每对象 `visible_depth_support = original_mask - ambiguous`。
- `ambiguous_membership` 保留每个歧义像素原本属于哪些目标。
- 逐像素验证支持互斥，以及 `visible_depth_support | ambiguous_membership == original_masks`。

373 帧全部通过上述验证及源哈希复核。180 帧存在任意非零交叠，累计排除 **25,651,292 个对象×像素观测**；这不是删除原数据，也不是删除同样数量的独立图像像素。

| 帧 | 原锅 / 盖像素 | 新锅 / 盖可见支持 | 歧义像素 |
|---|---:|---:|---:|
| 20 | 152471 / 105881 | 46620 / 30 | 105851 |
| 90 | 142449 / 64078 | 142449 / 64078 | 0 |
| 282 | 156795 / 105784 | 51289 / 278 | 105506 |
| 345 | 146984 / 95269 | 51814 / 99 | 95170 |

产物：`postprocess-public00-visible-support-001/manifest.json` 和 `frames/*.npz`。新支持仅用于尺度/接触观测筛选；没有修改语义标签、原掩膜、轨迹，没有填补遮挡区域，也没有推断哪个对象在前。盖合时盖的真实可见区域也可能被保守排除；在没有额外遮挡顺序证据前，这属于明确记录的观测损失。

## 三、NAS：完成 4581 帧 CPU 尺度与平移候选重估

来源：`/home/fmc3/FermiBotNas/dataset/Ego_Data/recording_#box-20260706-b92c_20260721_144600`。

实现 `postprocess_depth.py`，使用已有 MoGe 逆深度、注册实测深度、原目标掩膜、未作深度缩放的 `mesh_pretransformed.obj` 和旧相机内物体旋转。没有只写诊断报告而省略重估步骤。

处理方法：

1. 以注册实测深度的设备彩色相机 K 为共同 K。MoGe 按各帧原 K 反投影并重新投影，使用最近深度缓冲，不填洞。两分支使用相同帧、掩膜、网格和算法。
2. 在同一掩膜、共同有效深度及固定旧网格投影的交集中建立三维对应。固定旋转，稳健估计每帧尺度与相机内平移。
3. 只有两分支同时通过预设门控的帧才参与标定，分别取各自整段固定尺度中值，不逐帧改变机器人/物体形态。
4. 固定整段尺度，重新估计每帧平移，再渲染新网格，检查重投影后的覆盖和深度残差。
5. 只把两路都通过的帧标成配对候选；保留所有真实时间、失败帧和拒绝原因。失败变换输出 NaN，不插值或外推。

默认运行参数：4 个 CPU 进程、每轴采样步长 4（320×180 工作网格）、共同网格重叠至少 128 像素、目标覆盖至少 25%、共同有效深度覆盖至少 50%、最大连通分量占比至少 85%、三维/重投影残差上限 3 cm、尺度范围 0.5–2、相对旧位姿平移变化上限 0.5 m。阈值在本次执行前固定，未为提高通过率放宽。

实跑目录：`postprocess-depth-full-001/`。输入/配置记录在 `input.json`；每帧初始拟合和拒绝原因在 `checkpoints/*.json`；重投影验收在 `manifest.json`；完整时间轴输出为 `refit.npz`。

| 项目 | 结果 |
|---|---:|
| 全部源帧 | 4581 |
| 两路同时通过第一遍 | 224 |
| 固定尺度后两路同时通过重投影 | 51 |
| 保留的失败帧 | 4530 |
| 预测深度分支固定网格尺度 | 0.8380106187 |
| 实测深度分支固定网格尺度 | 0.8125030371 |
| 配对候选预测深度 RMSE 中值 / P95 | 24.17 / 29.13 mm |
| 配对候选实测深度 RMSE 中值 / P95 | 20.49 / 26.68 mm |
| 运行时长 | 约 3 分 45 秒 |

这里的尺度是相对同一未作深度缩放网格的**候选**，不能直接与旧管线 `scale.json` 当作同一目标函数下的精度比较。固定旧旋转、预测深度产生的旧初值、掩膜污染和部分可见表面都可能造成偏差。

主要输入拒绝原因（允许一帧命中多项）：共同网格像素不足 778 帧、网格对掩膜覆盖不足 2775 帧、共同有效深度不足 1309 帧、掩膜明显碎裂或为空 815 帧。末段问题不能单靠更换深度源消除。

**本轮没有重新估计 SLAM 世界尺度、相机轨迹、手部轨迹或物体旋转，也没有验证掩膜的语义身份。** 输出只包含相机坐标中的物体候选，`geometry_quality_accepted` 与 `full_training_accepted` 均为 false。51 帧通过自动门控不等于这段录制已具备完整训练资格。

## 四、复现命令

在 `codex/experiments-nvidia-v2d` 的工作树根运行，使用现有 Python 环境，无需修改锁文件或同步依赖。每次输出目录必须全新。

```bash
.venv/bin/python -m experiments.nvidia_v2d.postprocess_scene \
  --input /var/tmp/fmc3-sharpa-validation-20261027-001/postprocess-public00-scene-001/input.json \
  --output /var/tmp/fmc3-sharpa-validation-20261027-001/postprocess-public00-scene-reproduce

.venv/bin/python -m experiments.nvidia_v2d.postprocess_occlusion \
  --scene /var/tmp/fmc3-sharpa-validation-20261027-001/postprocess-public00-scene-001/manifest.json \
  --output /var/tmp/fmc3-sharpa-validation-20261027-001/postprocess-public00-support-reproduce

.venv/bin/python -m experiments.nvidia_v2d.postprocess_depth \
  --reconstruction /home/fmc3/Documents/workspace1/FMC3_Ego_Collection_Upper_Dashboard/.ego-dashboard/integrations/nvidia-video-to-data/smoke-20261023-box-01/attempt-004/workspace/jobs/c75278b7-a50b-4472-a73c-17181671d7d0/reconstruction \
  --sensor /home/fmc3/Documents/workspace1/FMC3_Ego_Collection_Upper_Dashboard/.ego-dashboard/integrations/nvidia-video-to-data/smoke-20261023-box-01/diagnostics/sensor-depth-full \
  --output /var/tmp/fmc3-sharpa-validation-20261027-001/postprocess-depth-full-reproduce
```

## 五、测试与边界

18 项服务测试通过，覆盖同源世界证明、独立居中/不同 SLAM 来源拒绝、对象坐标复合、身份交换候选与空目标、遮挡支持互斥与逐像素来源还原、已知尺度/平移恢复、退化点云拒绝、完整时间轴、失败帧 NaN 保留及真实非固定帧间隔。Ruff 检查通过。

本轮未修改原录制、旧重建、原掩膜、共享环境、核心锁文件或其它仓库。下一步实际质量闭环需要把互斥可见支持接入对象尺度/接触观测，并重新审查遮挡前后的姿态和深度顺序；不应以自动换标签代替这些证据。

## 六、交付复核补充：000031 双手工具、NAS 连续性与 000000 接线边界

以下补测在提交 `72b4043` 后完成，不改写前文的历史组装结论。只使用原本地输入和新独立输出，未重跑 GPU、既有服务测试或策略训练。

**000031 已补齐实际角色与相对运动诊断。** 固定 `mini_sweeper` 的唯一手来源、原共享 SLAM 世界和已烘焙尺度，完成 290 帧两手腕/五指尖到两物体的精确无符号三角面距离，以及刷子相对架的运动；另以固定 256 个刷面样本检查其与架的邻近。RGB 抽查支持左手扶架、右手操作刷子、末段回放退手。

40–219 帧粗操作窗口内，左手→架最近指尖距离中值/P95 为 **1.3/3.7 mm**，右手→刷子为 **4.0/28.8 mm**；反配对中值分别为 452.6、107.7 mm。刷面采样→架的对应值为 0.79/43.82 mm。角色配对与 RGB 一致，但没有刷毛工作区定义、可靠接触真值或闭合网格，这些数字不代表接触成功率。

需复核的关键帧为 **185**（右腕相对刷子跳变）、**198**（左腕相对架跳变）和 **206**。第 206 帧 RGB 仍呈握刷子状态，右手最近指尖至刷子为 45.24 mm，全部 778 个 MANO 顶点中的最小距离为 27.45 mm；这是配准疑点，不是连续表面无接触证明。模型观测标记不能替代物理真值，完整训练门控仍拒绝。

- [000031 中文补充说明](/var/tmp/fmc3-sharpa-validation-20261027-001/public31-tool-review-001/双手工具验证说明.md)
- [双手角色 RGB 抽查](/var/tmp/fmc3-sharpa-validation-20261027-001/public31-tool-review-001/双手角色_RGB抽查.jpg)
- [相对运动与邻近曲线](/var/tmp/fmc3-sharpa-validation-20261027-001/public31-tool-review-001/双手角色_几何诊断.png)
- [关键帧手部投影](/var/tmp/fmc3-sharpa-validation-20261027-001/public31-tool-review-001/异常帧_手部投影.jpg)

**NAS 的 51 帧只占 4581 帧的 1.11%。** 最长连续段为 4474–4484，共 11 帧，原时间跨度仅 **0.333952 秒**；无法覆盖完整动作。已经完成的是全时轴、固定旋转下的两路尺度/平移候选重估；统一米制世界中的相机、手与物体完整对照尚未完成，不能把候选残差优势外推至整段。

**000000 的新 `visible_depth_support` 尚未接回尺度/接触观测重算。** 已完成的是遮挡污染支持修复和来源守恒验证，不是已经改善三维姿态或证实身份交换修复。下一步只需将可见支持接入现有观测筛选、对照残差，歧义/无支持区域继续拒绝；当前证据不支持硬交换锅与盖标签。

本次新增诊断复核了本地输入、290 帧 RGB 摘要和物化 MANO；NAS 目录只读访问超时，源 NAS 视频 SHA 沿用已有同源凭据，未宣称本次重新核验整个 NAS。详细交付差距与最小下一步见 [只读审阅记录](/var/tmp/fmc3-sharpa-validation-20261027-001/delivery-review-001/交付差距审阅.md)。
