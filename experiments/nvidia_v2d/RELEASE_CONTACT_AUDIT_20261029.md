# Sharpa 接触传递与释放阶段审计（2026-10-29）

本审计只读原始 parquet、checkpoint 配置与已保存 rollout；未改输入参考、动作、物理状态或成功判据。独立 CPU 重算脚本和带输入 SHA256 的完整数值在 [audit.py](../../.e2e/zero_success_20261027/audit/contact-chain-001/audit.py)、[summary.json](../../.e2e/zero_success_20261027/audit/contact-chain-001/summary.json)。槽映射及实际接触时序另见 [contact-mapping-audit.json](../../.e2e/zero_success_20261027/audit/release-contact-001/contact-mapping-audit.json)。

## 已证实的链路

1. `sharpa_bridge._contacts` 对物体表面固定采样 4096 点，以物体点到 MANO 顶点距离 **<1 cm** 推断接触；这是重建几何启发式，不是力传感器或人工接触 GT。接触点按最近 MANO link 中心归入 16 槽，每槽平均位置、投票 object part id。`tips_distance` 是另一项几何记录，不是奖励里“每根手指必须贴锅”的命令。
2. 官方 `SharpaHandKinematics` 只根据手腕、拇指中间点/尖端及另外四指的中间点/尖端求 IK。腕位/姿态权重各 0.2，指尖位置权重多为 1.0（小指 0.5）；未输入物体、桌面或接触约束。`sharpa_multitip` 更新 robot 字段后保留 MANO 字段。**本样本官方重定向前后的两手各 6 组 contact/tip 数组逐元素完全相同。**因此人手接触参考确实直接继承；但它随后作为整手的接触作用目标，而不是复制为机器人某个指节的硬约束。
3. FlashCHORD 的 reference 读取 MANO object contact 点/法向/id。位置与姿态重采样分别为线性/SLERP，接触标签使用最近邻。`objectives/wrench.py` 将每手、每物体的点和法向转换成 512 方向的 6D wrench support，取所有接触贡献的最大值；不要求 16 个 MANO link 与 17 个机器人槽逐项一致。
4. 实际仿真槽是 **right 17 在前、left 17 在后**。观测配置虽列 left/right，但按手名查索引重排。未发现左右颠倒。物理 shape0/shape1 力统一转换成“作用在物体上的力”：物体为 shape1 时取负，shape0 时保持正。参考使用物体面内向法向，与这个力的符号约定相容；未发现下游统一翻号。
5. 仿真接触按原始 contact count 判 active；wrench support 使用每槽合力的归一化方向，零合力方向则不产生 support。因此“count 非零”“承载非零力”“奖励 support 非零”并不等价。原 capture 没保存实际力，不能反推承载力大小。

## 源 9 秒以后的问题

所有时间以 parquet 的 20 fps 为准。源 frame 204（10.2 s）至 248（12.4 s）两手的参考接触均为零。源 9–10.2 s 左手基本只有 `link_thumb3`，frame 193–194 短暂掉线；右手直到 191 才间歇出现，191–192、195、197、199、201–203 活跃，其间反复闪断。

runtime 重采样实际是 `linspace(0, 12.4, 496)`，不是 `step * 0.025`。最后活跃 frame 406 对应 10.1705 s，**frame 407（10.1956 s）起双手参考 support 全零**。这也解释旧图末尾“12.375 s”的标注与实际参考末帧 12.4 s 的细小差别。

左拇指 frame 201（10.05 s）有一次已量化的法向翻转：object-local 法向与 frame 200 点积 **−0.997318**，与 frame 202 点积 **−0.980005**。201 的 `n·(hand_contact − object_contact)` 是 **+5.163 mm**，相邻两帧为负。全部 285 个有效接触槽的法向与对应最近 mesh triangle 的负面法向点积在约 `[0.9999999, 1.0000002]`，接触点到面最大距离约 2e−8 m：这不是 parquet 读取错误，而是上游选择了那个面的法向。

`contact_utils.find_object_contact_positions` 的 docstring 声称选择朝向手部的面，实际实现仅作距离 `argmin`，没有法向筛选。**薄锅沿内/外面跳选是合理候选解释，尚未由真实接触力确认。**Mesh winding consistent，但非 watertight；不能把正投影直接当作已证实穿透，也不能无条件翻转所有法向。

RGB 审核显示 10.5 s 已释放并放在桌上；同一时段参考 object root 从 frame 210 到 248 仍平移 `[+1.01,+6.86,−10.09] cm`，旋转约 0.1284 rad。参考平移有限差分速度的 median/p95/max 为 `0.0772/0.1272/0.1289 m/s`（源时间；半速播放约减半）。这是参考导数，不是 solver 速度。几何审计的桌面内 collision 底点在约 10.5 s 高于桌面 5.76 cm，末尾低于桌面 3.47 cm。单一常数 table height 无法同时解释这两帧。参考阶段与静态支撑存在矛盾，不能靠继续训练消除。

## 当前奖励、控制和物理配置

policy 2621440 checkpoint 的解析配置确认：有效课程权重 object keypoints 20、contact support 10、missed −1、unintended **0**、hand keypoints/joints 各 0.25；接触力惩罚和 force closure 关闭。参考无接触时，contact support 和 missed 项都归零，当前 unintended 权重也为零，因而**没有独立的释放接触惩罚**。物体跟踪奖励仍跟随上述下降参考。

Actor 是 residual pose，手腕位置 scale/clip `0.05/0.2 m`，归一化 rational 映射，EMA 0.3；手指 scale/clip `0.15/1.0 rad`。actor 不观测 object body velocity（critic 观测），也没有单独物体速度奖励；这些是可检验因素，不足以独立证明抛甩原因。腕平移物理驱动 kp100/kd12、原 45 N 力限保留；`action.wrist_kp=25` 属于另一处配置，不能误读为已覆盖物理平移关节配置。

物理为 MuJoCo-Warp/Newton solver，implicitfast、pyramidal cone、100 次迭代、50 line-search、100 Hz 物理（20 Hz × 5 substeps）、njmax2048/nconmax512、native MuJoCo contacts。所有碰撞 shape friction 1.0，物体 free-joint passive damping 0；wrench 几何近似使用 friction 0.1，是奖励近似参数，与仿真材料参数不同。物体 URDF mass 0.3 kg、对角 inertia 0.001 kg·m²；这些是配置值，未做真实物理标定。有限桌面顶面 z=0，圆柱半径约 0.30435 m，robot-support 碰撞开启。

当前 raw count 只覆盖声明的 17 个接触角色。每手另 4 个 MCP_VL 碰撞体未进入这些角色；全槽零计数不能证明所有手 shape 无接触。旧 capture 的 ground count 只记录 world body<0 的无限地面，有限桌面 body53 未被独立记录。不能仅据这些记录断言“无碰撞穿桌”或“零力释放”。现有轨迹中桌外下落拟合约 −9.81 m/s²，与失去有限支撑后的重力一致，具体拖拽冲量仍需新增完整 contact/force 捕获。

腕部重力核查有明确反证：`SharpaHandsConfig.gravity_compensation=True`，控制配置将 robot body gravcomp 设为 1、joint actgravcomp 设为 True；自定义腕增益 factory 保留此设置。冻结镜像中的 Newton 把这些属性写入 MuJoCo，MuJoCo-Warp 再将 `qfrc_gravcomp` 加到 actuator force，随后按 joint actfrcrange 限制到原 ±45 N。因此不能把 17–18 cm 的腕参考偏差直接称为“没有重力前馈”，也不应重复叠加 mg。[gravity-config.json](../../.e2e/zero_success_20261027/audit/wrist-gravity-001/gravity-config.json) 保存镜像内源码/资产 SHA 与关键行；验证范围是配置及后端传递实现，尚未直接捕获运行时 MuJoCo gravcomp 数组。

冻结镜像实际左/右手 URDF 总质量分别 1.245265/1.243265 kg，wrist-z 承载链（含 z dummy）重力约 12.217/12.197 N。即便假设完全无补偿、静止无接触，kp100 对应静差也约 12.2 cm；整锅 0.3 kg 全压单手的同类估计约 15.1 cm。这只是反事实量级计算，不能替代实际驱动记录。应将 `actual-reference = (actual-command) + (command-reference)` 分开，结合全部碰撞力、速度、actuator force 和重力项判断策略主动偏移、动态跟踪或力限饱和。只有驱动误差证据支持时才考虑 kp150/200 的单因素试验，保持 45 N 上限，并检查震荡/碰撞冲量；目前没有足够依据推荐直接提高 kp。

最新完整 capture 已保存命令与动作，实际分解见 [target-decomposition.json](../../.e2e/zero_success_20261027/audit/wrist-gravity-001/target-decomposition.json)。原 2.6m checkpoint 这次录制的末帧右腕 world0 只有 −1.7 mm 参考偏差（命令 +26.2 mm、跟踪 −27.9 mm），world5 为 −63.7 mm（命令 −51.5 mm、跟踪 −12.2 mm），没有复现旧批 17–18 cm 下沉。world4 则向上偏 +192.5 mm，其中命令已触及 +200 mm clip，跟踪只差 −7.5 mm。56 维动作按 rational+EMA 重建目标的误差 ≤5.5e−8 m，已验证 action 右 xyz `[0:3]`/左 `[28:31]` 与 joint target 右 `[28:31]`/左 `[0:3]` 的不同布局。不能把两批不同物理轨迹的同 world 编号拼成一个证据。

005 release 消融的首次腕失败在全部 8 个新捕获 world 中都是 step14–16（源约 0.35–0.40 s），远早于 release407。例 world0 左腕总误差 `[-123.3,+38.2,−89.7] mm`，分解为命令残差 `[+200,+200,−40.5] mm` 加跟踪误差 `[-323.3,−161.8,−49.2] mm`。这说明该训练后的策略已改变早期接近行为，不能把其失败仅归为撤手阶段物理碰撞。

修改 wrist clip 也必须保留对照：原 2.6m 完整 capture 中 world7 唯一全程通过原腕位/物体位置/物体姿态阈值，407 前每轴绝对 residual p95 为左 `[14.03,10.26,10.09] cm`、右 `[12.87,7.12,8.82] cm`。0.075 m clip 会触及此轨迹约 75% 的左右手帧；0.1 m 也触及约 44%/35%。持握段 step120:360 的比例更高。0.1 是二者中较小的干预，但尚未证明能保留抓持；0.075 不能视为只删除坏的撤手动作。当前 rational 映射的 input scale 依赖 clip，改变 clip 会改变所有非零输入到物理 residual 的映射，不仅是裁掉峰值。逐轴 0.1 m 上限允许 3D 范数达 0.173 m，逐轴 0.075 m 则为 0.130 m；二者都不能保证 actual-reference 不超阈值，因为还有跟踪误差。

## 最小可验证修复与边界

第一项控制消融已经实现于 [release_contact_objective.py](release_contact_objective.py)：保留原 `unintended_contact` 算法和 term 布局，只在显式 runtime frame407 起使用负权重（005 使用 −10），此前把该项置零后重新 compose。构建拒绝越界帧、disabled、非负权重，以及 release suffix 非零 reference wrench support；课程权重也不能静默变为非负。它不改动作、参考或成功判据。冻结镜像仅 CPU 的 9 个测试验证真实 Warp contact kernel、边界、其余 term 完全不变、reset、Hydra/checkpoint JSON 重载。005 的 3 个 checkpoint 各 32-world 评估共 0/96，而原 2.6m 的同规模对照为 1/32；**005 随后查出继承 replay 的奖励项语义污染，因此不能作为纯粹的 release gate 反证**，也不能把正确实现的在线包装视为已修好训练协议或物理参考。详细训练/评估记录仍由主实验流程保存。

### 005 replay 语义污染（已证实）

`training/flash_sac/replay.py` 保存未加权的 `discounted_objective_terms`；`_relabel_rewards` 使用当前课程权重再次点积，并乘 `frame_dt`。原权重为 0 时，非零 unintended 原始项仍保存在 replay 中。005 改为 −10 后，新在线 evaluator 虽在 frame407 前置零，继承的旧 committed 样本不会追溯重算原始项。

对原 2.6m 的真实 65536 条 replay 直接读取发现：起始 frame<405 的 61890 条中有 **42707 条（69.00%）** 旧 unintended 项非零，按 −10 和 frame_dt0.05 重算后，该组新增平均 −1.3896、最负 −3.8991 的奖励；这些 3-step 窗口全部在释放边界前，本应为零。起始405/406的249条跨边界样本单列，不从聚合项猜测各步贡献。已确认错误惩罚存在，但它对005失败的因果贡献仍须清除旧 replay 后比较。旧 pending payload 虽也可非零，恢复逻辑调用 `_flush_pending` 重置其 size/write_index，不能把旧 pending 数值直接算作被消费的错误样本。

旧65536 capacity只能保存256次256-world committed batch，小于完整496步；这证实容量不足以同时留住一整批同步完整轨迹，但不单独证明发生遗忘。新006设 fresh replay capacity262144/minimum131072，完整恢复参数/optim但不加载旧replay。源码确认 ready 只看新 replay.size，非累计environment_steps；3-step聚合先占2次pending，最早第514次新collection才可更新，和实际前513次loss0的记录一致。容量变大只提供保存全阶段的可能性，早终止仍可能造成阶段缺失，必须继续检查实际phase histogram。

006 的真实 `replay_3145728` 已读回验证：[旧 replay 数值](../../.e2e/zero_success_20261027/audit/replay-release-001/legacy2621440.json)、[006 数值](../../.e2e/zero_success_20261027/audit/replay-release-001/fresh006_3145728.json)、[复现脚本](../../.e2e/zero_success_20261027/audit/replay-release-001/audit_replay.py)。新 buffer 的 capacity=size=262144，246876 条起始 frame<405 的样本中 unintended 项**全部精确为0**。405/406跨界949条单列；frame≥407共有14319条（5.46%），其中450–495仍有3356条，phase实际覆盖0–495。晚段覆盖已存在，旧语义污染已清除；5.46%不能称为均匀阶段覆盖。新 pending 的490条释放前样本也全部为0。

参考修复应另开可回退候选：从已确认桌面稳定和已释放的 RGB 帧联合估计固定支撑平面与释放后的静止物体 pose，用相机运动解释图像中的视点变化；同时保留物体轮廓/2D重投影约束、支撑区域内不穿透和落桌接触约束，在落桌前与已验证的拇指/虎口接触时序连续衔接。不能把物体参考直接移到策略实际轨迹，更不能强制另四指贴锅。当前桌面绝对高度本身来自近似重建，因此不能把 z=0 当实测 GT。需要报告修正幅度、重投影变化、接触时序、桌面残差和速度，再决定候选是否可训练。

针对 frame201 法向，先可离线输出最近面的候选、面朝向、hand/object 间隙、前后帧一致性与 RGB 局部叠图；确认薄壁跳选后，再在重新推断接触时增加可解释的面侧/时序一致性约束。原始值应保留，对无法消歧的帧输出 unknown/confidence，不能以“平滑后更好训”代替证据。

## 后续接触感知 IK 候选（本轮不改）

当前官方 IK 保持人体指尖/腕部姿态相似，无法保证粗大机器人指面能以相同位置和方向夹持锅沿；整手 wrench 奖励也允许许多不同的机器人接触部位实现相似作用。因此“跟上了轨迹”仍可能表现为不自然的指面支撑、刮蹭或撤手拖拽。不能把它简化成左右/指序映射错误。

可单独生成、审核新的离线参考候选：

1. 使用已核实的拇指与掌侧/虎口夹持时段作为语义证据；在机器人 thumb PP/DP 与可用掌侧碰撞面上选择接触点，允许其在同一物理角色内换面，不固定 MANO 槽到 Sharpa 槽的一对一编号。其余四指仅保留姿态、关节限位与碰撞约束，绝不强制贴锅。
2. 在原 IK 姿态目标之外加入 object-local 接触点间隙和相容法向代价、真实碰撞几何的非穿透约束，以及腕/指运动的速度和加速度约束。接触目标必须先通过上述法向/时序审核；无确定证据的帧不作为硬接触目标。应保留人手与机器人尺寸差异所需的适应自由度。
3. 接近、承载、放回、撤手的权重依有证据的阶段切换。释放阶段移除吸引接触目标，检查撤离路径的最小间隙；放回阶段同时检查桌面支撑。若源手、物体和桌面本身矛盾，应先拒绝候选，而非靠 IK 推动物体参考来满足姿态。
4. 独立对比原 IK 和候选的 RGB 重投影、腕/指变化、接触点/法向残差、时序稳定性、物体/桌面穿透、速度及力限可达性，再做同条件无辅助物理 rollout。离线几何满足不等于力可行或闭环成功；原参考、原结果和验收阈值全部保留供对照。

这条方案尚未实现，不能与当前 frame407 释放奖励的单因素结果混在一起归因。
