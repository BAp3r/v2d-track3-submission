# ACE-Ego-Hand 接入状态复核

日期：2026-10-30。本轮只读审计既有运行产物及源码，未重训、重建或覆盖旧结果。

## 结论

**旧接入存在实际MANO姿态契约错误；后续修正版已跑通重建输出，但尚未完成Track3重定向→策略训练→实际rollout的对照验证。** 不能把“有入口/输出NPZ”称为竞赛方法验证完成，也不能说ACE一直停留在仅提议状态。

此前进展总报告遗漏ACE支线，应补充。此前“只用NAS V2D”的回答仅适用于本次竞赛任务/策略训练；整个项目重建实验还用过自采纸箱视频。

## 官方管线与当前ACE模式

NVIDIA上游统一入口的`--hand_tracking hamer`实际启用WiLoR/SAM2/HaMeR：WiLoR逐帧检测并提供MANO初值、SAM2追踪掩码、HaMeR按track回归。底层WiLoR路径也可以直接使用WiLoR的MANO输出，不应仅凭脚本名或hamer目录推断具体来源。

- [上游统一入口](https://raw.githubusercontent.com/nvidia-isaac/video_to_data/main/reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py)
- [上游基础流程](https://raw.githubusercontent.com/nvidia-isaac/video_to_data/main/reconstruction/modules/v2d_pipelines/run_ego_wilor.py)
- [官方v0.3发布说明](https://github.com/nvidia-isaac/video_to_data/releases/tag/v0.3.0)

本地新分支有两种不同用法：

| 模式 | ACE实际替代范围 |
|---|---|
| `--hand_tracking ace` | ACE输出MANO及相机平移，复用既有matching/alignment/interpolation并进入最终result |
| `--hand_tracking hamer --hand_detector ace` | ACE用于提示/检测支路，最终手pose仍来自HaMeR；不能当完整ACE姿态实验 |

ACE本身从视频预测双手MANO和相机系平移，有K-given/K-free；不直接恢复物体mesh、物体轨迹、接触力或机器人策略。官方claim门控也不保证所有帧有效。[ACE官方实现](https://github.com/ggxxii/ACE-Ego-Hand)。

## 实际实验与旧错误

视频：`/home/fmc3/FermiBotNas/dataset/Ego_Data/recording_#box-20260706-b92c_20260721_144600/color.mp4`，4,581帧、30Hz、152.7秒。不是V2D评分集。

1. 10月23日ACE/WiLoR对照曾完成基础输出和视频，但后来被正式标记为`invalidated_ace_pose_convention_20261026`。
2. 旧适配把ACE基于非flat MANO mean的手指残差直接送入flat-mean消费者，漏加手部平均姿态。同一native预测在frame2400右手造成31.125mm平均、90.254mm最大顶点差。这是格式/几何适配误差，不是相对真实手部的精度评测。
3. 原生pickle本身未受该导出错误影响；但其衍生的渲染、手mask提示、匹配、深度对齐、插值和最终bundle均受影响，不能仅替换raw JSON继续沿用旧缓存。
4. 10月26日修复采用官方完整视频连续VAE/tiled推理、INTER_AREA缩放、记录相机内参，修复MANO mean和左右手转换，增加版本/缓存契约；重跑下游至result及Three.js。
5. 同一右手样本修正后消费者顶点差降到浮点精度。左手物理MANO与镜像右手shape basis仍不同，抽样最大约7.78mm，不应把左右手都描述成逐顶点完全一致。

原始证据：[旧对照状态](/var/tmp/video_to_data/box_20260721_ace_comparison_20261023/run_summary.json)、[修正版完整报告](/var/tmp/video_to_data/box_20260721_ace_official_20261026/report.json)、[实际MANO消费者测试日志](/var/tmp/video_to_data/box_20260721_ace_official_20261026/pose_regression.log)。

## 当前哪些已验证，哪些仍缺

| 层级 | 状态 |
|---|---|
| ACE权重 | K-given/K-free文件存在于NAS，入口symlink可达 |
| 官方原生推理 | 实跑4,581帧K-given及K-free，保留pickle、参数及视频 |
| MANO输出适配 | 修正v2已实现、实际样本消费者回归通过；本轮cache校验True |
| matching/深度对齐/插值 | 修正版实跑完成，右手匹配真实记录540→2610；这些变化包含多项修复，不能全归因MANO mean |
| portable result/Three.js | 修正版存在且已实际检查NPZ：相机/物体有效4581帧，左手3703、右手2939；有效包含补帧，不是独立精度证据 |
| 独立3D精度/时序/接触评估 | 尚无GT精度结论，单纸箱样本不证明泛化 |
| ACE→Sharpa→SAC | 未找到本支线实际训练/rollout结果；不在三路重定向比较中 |
| Track3正式提交 | 本次不是ACE重建分支，不能用六榜成绩判断ACE好坏 |

## 新确认的后处理问题：原生ACE不等于最终训练参考

`integrated_fixed/result/manifest.json`明确最终手来源是`wilor_aligned_filled`，backend为ace。名称wilor是兼容目录，不代表此处仍运行了WiLoR网络。

frame2400右手实际记录：

| 阶段 | 相机平移Z | 内参fx |
|---|---:|---:|
| ACE原生导出/匹配 | .488835m | 689.178 |
| 下游对齐pre-dz | .536384m | 转向pipeline相机 |
| 深度对齐后 | .636530m | 756.134 |

对齐诊断`dz=.100146m`；三维平移总变化150.43mm，最终track hand_scale约1.16175。两套内参的转换及单目深度拟合是实际发生的适配，不直接等于程序错误；但它说明换成ACE后，下游仍会明显改变原生3D结果。需要单独验证同一场景内相机、物体与手的尺度/投影一致性，不能凭原生视频自然就断言最终参考自然。此前已知V2D深度时序问题也不会仅因换ACE自动消失。

## 下一项最小对照（仅提议，未启动）

冻结当前完整提交。选03及37，固定视频/物体重建/共同相机，比较WiLoR与修正版ACE；先不长训，分别保存原生、对齐后、机器人参考、实际状态四层。量化有效覆盖、相同可见帧重投影、腕速度/加速度、手物距离、首步腕偏移和实际物体误差。原生与对齐后应在共同坐标中比较，不能直接把不同相机系的数字当精度。

若ACE仅提升覆盖/视觉，未改善实际可执行性，不批量替换；若有实际运动收益，再做同预算SAC验证。自然手势仍是附加质量，不新增完美接触门禁。

## 视频和输出

- [官方K-given完整视频](/var/tmp/video_to_data/box_20260721_ace_official_20261026/official_native_k.mp4)
- [K-free完整视频](/var/tmp/video_to_data/box_20260721_ace_official_20261026/official_kfree.mp4)
- [旧输出与官方对照视频](/var/tmp/video_to_data/box_20260721_ace_official_20261026/old_vs_official_comparison.mp4)：其中旧列仅作失效适配诊断，不能作为有效质量结果。
- [修正版result.npz](/var/tmp/video_to_data/box_20260721_ace_official_20261026/integrated_fixed/result/result.npz)
- [修正版result来源manifest](/var/tmp/video_to_data/box_20260721_ace_official_20261026/integrated_fixed/result/manifest.json)
- [本地接入说明](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/reconstruction/modules/v2d_ace_hand/README.md)

当前ACE模块/入口改动在共享工作区尚未形成这个比赛分支的完整已提交版本；保留其他任务改动，本轮不擅自覆盖、打包或提交。技能文档仍未同步ACE用法，不能用旧技能说明代替当前实际源码。

## 补充：用户Windows原生实验，不与Linux接入混淆

用户指定源目录为`/home/fmc3/FermiBotNas/tmp/ACE-Ego-Hand`，实际存在完整results/options/checkpoints/scripts。之前Linux report中“Windows仅部分文件可用”的历史描述不代表当前NAS状态。

- Windows记录：`results/ego_125108/REPORT.md`记9月30日RTX3080 Laptop初测81帧；`results/ego_extended/README.md`记10月8日三段完整K-given，393/565/4581帧。input.json保留F:\workshop3路径，内参来自录制meta、编码832×480、tap15、连续VAE+tiled。
- NAS源码HEAD=97578680931b3f1c8111396c100d595b98857fb8；大量git diff为CRLF，忽略行尾后实质改动集中在4文件58新增/8删除，涉及显存加载/连续流式VAE等。原生viz_preds.py和mano_utils.py在忽略行尾后无修改；因此不能把我们另一条adapter漏MANO mean的错误直接归因到此Windows结果。未做全checkpoint哈希/全环境复现，仍不声称严格benchmark复现。
- 现存Windows批次README明确没有该批WiLoR对照，也没有3D真值，有限值、presence与速度诊断不是准确率。用户在其他实验的无遮挡WiLoR更合理观察作为需分层验证的证据保留。已视检一段comparison.jpg，原生可见手指有待核实的轮廓/姿态偏差。
- ACE论文Table S4：无-p腕对齐MPJPE，ARCTIC in-view WiLoR22.775/ACE15.423mm，HOT3D30.921/12.703mm；所以不能将全部优势归于出画面。in-view只需21关节有一点在视野内，不等于无遮挡；没有完整可见逐指子集结论。主表MPJPE-p/PA-p含漏检惩罚，PA还做Procrustes；OOS另分层，均不是机器人接触或抓握自然度。来源https://arxiv.org/html/2608.20308 。
- 解释：平均关节/覆盖指标、可见性分层、域差异及低分辨率全帧与手crop的细节差异，可与用户无遮挡WiLoR更合理同时成立；最后一点是架构推断，未量化因果。不能据论文平均分自动决定全量替换。最小验证为同帧可见/部分遮挡/出视野分层，原生各自渲染、相同K及共同坐标后分别比较关节与下游actual。
