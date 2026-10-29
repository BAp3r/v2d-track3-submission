# Public44 三路实际 rollout 与独立几何消融

用户于 2026-10-29 明确要求比赛主线继续推进：保留可运行版本，比较官方 IK、DexPilot-chain、multi-tip；几何独立消融，自然抓握仅作为附加质量。完整接触细节不再作为开训条件。

分支为 `codex/sharpa-unassisted-repair`。本轮产物根为仓库下 `.e2e/zero_success_20261027/threeway-rollout-20261029-001/`，后文路径均相对该目录。

## 保留的运行基线

`retained-baseline.json` 绑定原 011 的 policy/state、冻结代码、参考、支撑、角色映射和完整补录。其独立补录为 CHORD 26/32、完整释放 32/32、放稳 28/32、三项交集 24/32；右拇指 DP 接触仍为 0。010 及所有旧结果也保留。历史训练累计预算不同，不进入本轮同预算排名。

旧 012 联合几何/腕部奖励方案没有启动，已登记暂停原因；未完成的几何候选和失败证据原样保留。

## 预先固定的主线比较

`preregistration.json` 在任何新训练前冻结三份配置与全部运行源码，`protocol-mount-fix-001.json` 验证三路输入及基线 SHA。DexPilot 的首次启动因遗漏只读审计目录，在训练前失败、0 优化步；失败目录保留，attempt2 只补挂载，不改数据、训练参数或预算。

| 项目 | 三路共同配置 |
|---|---|
| 源数据 | loaded005，249 帧，20fps，同 calibration；全部非 robot 字段逐值一致 |
| 初始化 | 各自从零、seed42，fresh replay |
| 环境步数 | 每路 2,097,152 |
| SAC | 256world；replay262144/min131072/batch512；每 collection 8 次更新 |
| 物理 | 同冻结镜像、有限桌面、200Hz 积分、20Hz 控制、0.5 倍播放；腕 kp100/kd12 |
| 训练扶持 | 相同缩放 VOC 课程，关闭 reset boost |
| 评估 | 每 524288 步保存；每个检查点完整 496 步×32world，辅助逐步验证为 0 |
| 模型选择 | 最后 2097152 模型为主结果，全部四个检查点均报告 |

运行审计 `runtime-config-audit-001` 逐字段检查三路实际 Hydra，777 个叶字段只差 `task.parquet`；SAC/初始化种子、replay、训练源码和物理配置相同。预算是训练环境步和优化更新预算，不声称三种离线重定向算法具有相同目标函数或求解耗时。

DexPilot 使用固定 `dex-retargeting 0.5.0` 的 `DexPilotOptimizer`，加入 16 段指节链约束。历史自定义 chain 源码未找到，明确标注为重建适配器。其时序项采用平方和，multi-tip 采用平方均值，22DOF 下相同系数对应 22 倍强度差异；配置在产生评估结果前冻结，没有根据结果调参。完整算法、误差和速度说明见 `dexpilot-recovery.md` 及 `runtime-config-audit-001/retarget-quality.md`。

## 评估口径

主线直接比较实际物体轨迹：对共同物体参考，全序列位置误差≤0.1m、姿态误差≤0.5rad。另保留官方 harness 的 ADD/AUC、SPIDER、ManipTrans 和 CHORD。CHORD 含路线各自的腕参考，因此单独展示；这里的 object-only 指标不是另行宣称的官方比赛总分。

全部环境及失败后的轨迹都保留。非有限或被标记无效的 world 仍计入固定 32 分母，并记失败；上游的零填充不会被当作真实测量。图表保存全程误差、实际抬升高度、首次失败时间和全部四个检查点。只报告一个样本、一个成对 seed 的首轮结果。

自然抓握、拇指 MC/PP/DP 参与、末段释放和放稳使用最终模型的独立完整接触补录，明确与正式评估不是同一批 world，不混算交集。展示三路均预选 world0，不按成功挑视频。低接触自然度不阻止主线训练。

## 独立几何 2×2 消融

固定官方 IK、相同预算与物理配置，raw 沿用主线；另外三组为：

- normal：仅修左手 source143/201 slot3 的两处物体法向，接触点、IDs、unknown 标签不变。
- support：只应用 source204 以后的共同坐标和放回支撑参考修复，所有指关节 q 不变。
- normal-support：严格组合以上两项。

没有加入 contact-aware IK、关节平滑、角色奖励或腕命令惩罚。三组源接触仍为左200/右85个活动槽，99个 unknown 全保留。`geometry-ablation-001` 保存逐字段因子隔离与全帧 FK 审计；各 gate 的28个绑定文件经过真实只读容器核验。运行代码仅新增独立输入 gate 分支，训练、控制、物理和评估代码与主线相同。

support 会改变后段参考，因此消融的 own-reference 跟踪成绩与相同物理标准的真实放回/稳定分开报告，不能将目标改变解释为自然抓握改善。

## 持久记录

每次 collection 的完整 loss、奖励和诊断均保留 JSONL/CSV，四次正式评估逐项保存；SwanLab 本地5092与TensorBoard6006同时开启，结束后逐标量读回核验。总进度继续追加仓库 `progress.md`。本轮主线、几何消融和历史基线各自标记，不合并成一个成功率。
