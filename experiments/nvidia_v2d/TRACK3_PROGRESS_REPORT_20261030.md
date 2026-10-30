# FMC3 Robotics：Track3 进度、实验、视频与评分说明

日期：2026-10-30。分支：`codex/sharpa-unassisted-repair`。本报告以已保存配置、全部评估和官方评分为依据；本轮只整理和事后统计，没有新训练、修改参考或再次提交。

## 1. 当前进度与实际完成程度

**已完成首版完整有效提交；还没有完成在正式评测集上可靠抓取、放回和撤手的目标。** 局部 public44 已有真正无辅助成功，但不能推广成20集稳定成功，也不能与官方隐藏真值成功率混为一谈。

| 工作 | 状态 | 结果与限制 |
|---|---|---|
| 视频→重建mesh/手/物体轨迹 | 已实现、已运行 | 20集、36个episode-object；六个曾被误判缺失的结果从失败wrapper保留目录恢复；未拿模板网格替换 |
| 共同多物体场景 | 已实现、已验证坐标保持 | 双物体共用一次scene变换、一份手源；相对关系误差到浮点精度。绝对重建准确度仍未知 |
| official IK / DexPilot-chain / multi-tip | 已有公平训练比较 | 单一public44、单seed、同预算；三个最终模型完整任务均0/32，误差不同 |
| 原DexPilot-chain | 已找到并适配，离线及短物理已跑 | 与此前rebuilt-chain区分；未做同预算学习策略训练，不能冒称已验证原方法优胜 |
| 几何修复 | 已有独立2×2消融 | support单项有效；normal及组合未带来更好完整成功率 |
| 正式实际rollout与网格覆盖 | 已完成 | 20/20集、36/36对象、9,120行actual pose，固定world0，逐控制步物体辅助0 |
| 六榜提交 | 已评分成功 | AUC、SP-SR、MP-SR、RPE、MPPE、CD-O全部返回有效分数 |
| 参赛身份 | 六榜队名已统一 | FMC3 Robotics，账号Bap3r_681，原提交和分数保留；Private Group独立存在 |
| 完整赛事交付 | 尚未完成 | 最终报告PDF、最终提交表及Kaggle最终提交选择仍须核实/完成；官网同步不等同审核通过 |

完整提交本轮新增训练0。20集执行共5,181个控制时刻，原速20Hz，GPU执行耗时相加831.77秒，约13.86分钟。所有失败尾轨迹保留，数值有限不代表任务成功。

[完整执行收据](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/full-submission-20261030-001/complete-evidence.json)；[提交校验与哈希](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/full-submission-20261030-001/submission-readiness.json)；[公开生成代码commit](https://github.com/BAp3r/v2d-track3-submission/commit/c3e975141a3a5a5c497ecdbdf9fd99fe6b7facb7)。最终赛事表要求以实际生成结果的commit为准，不能用本报告新增文档的commit替换它。

## 2. 当前方案是什么

实际架构是：**RGB视频→单目深度/SLAM、手部MANO估计、物体mesh与位姿→米制共同场景→Sharpa重定向→FlashCHORD物理环境中的SAC残差策略→仿真actual物体状态→官方格式打包。**

“official”在本项目实验名中指官方Pink/Pinocchio重定向，**不表示完整复现了官方端到端baseline**。不能用我们这套SAC策略的低分推断官方baseline上限。目录名含hamer也不证明实际估计器就是HaMeR；public44输入含wilor_raw及wilor_mask来源，依实际provenance判断。

| 方案 | 实质差别 | 当前验证状态 |
|---|---|---|
| 官方IK | 自由腕、22指关节的几何IK；没有独立机器人接触重建目标 | public44同预算学习；03训练；20集正式提交均采用此重定向 |
| DexPilot-chain rebuilt | dex-retargeting 0.5.0 DexPilotOptimizer，加16段链向量约束 | public44同预算学习已验证；不等于原Dex_Teleop_demo方向链 |
| multi-tip | 指尖与指间几何约束，时序项采用平方均值 | public44同预算学习已验证；部分指尖几何更准并未保证任务成功 |
| 原dexpilot-chain-calibrated | 原方向链Huber、分指权重和伸指先验，适配至Sharpa | 两片段离线与短物理；尚未训练验证 |
| 手物接触/支撑修复 | 接触角色、法向、碰撞并集、释放后的物体支撑参考 | public44局部诊断和消融；不是全部20集通用已验证修复 |
| SPIDER / Open-CHOIR等候选 | 前者偏物理参考优化，后者含抓握深度修正与时序拟合 | 查到代码或接入线索，未接入本次提交；论文检索不算实验成功 |

完整提交复用两个actor：16个双物体episode使用03的131,072步actor；4个单物体episode使用public44的2,097,152步actor。**完成覆盖不等于逐任务训练或跨物体泛化已解决。** 保留011的高成功率模型带有public44特定参考/支撑和不同训练历史，未被当作通用20集策略。

## 3. 做过哪些实验，得到什么结论

### 3.1 最早两路三种子对照

official IK与multi-tip，各seed42/43/44，每组2,097,152环境步、64world，最终每组249帧×16world无辅助评估。六组共12,582,912步，容器内耗时相加108.57分钟；全部任务成功率0。相对重建参考的本地MPPE三种子均值13.520/13.398cm，不足以认定方法胜负。

已核实有真实Adam更新；但旧训练`logging.mode=disabled`，**没有逐步loss，不能从checkpoint补造历史loss**。后来的短测曲线单独标明。旧训练reset前20步还会恢复VOC扶持；后来新入口设置`reset.voc_decay_steps=0`。这不推翻当时最终评估逐步零辅助的记录，但旧训练不能被称为全程无辅助训练。

### 3.2 单样本物理、支撑和学习修复

| 实验 | 实际结果 | 可得结论 |
|---|---|---|
| 无限地面/有限桌面与腕部伺服诊断 | 地面与手初始位置存在矛盾；单纯过滤碰撞不足以修好全部偏移 | 支持物理场景必须与参考一致，不支持只改一个碰撞开关就能解决 |
| finite-training-001，262,144步 | 4个checkpoint均完整任务0；最终物体误差14.056cm；早期个别world抬升7.79cm | 拿起一点不等于任务学会 |
| 单纯续训004/008 | 004三点0/96；008四点9/3/13/7（每点32） | 不支持仅不断追加训练 |
| fresh replay/容量/预热006→007 | 冻结候选独立复评40/96 | 有单样本无辅助收益；多因素联合改变，不能归因单个参数 |
| 物理子步静态夹具 | 放稳0/4→4/4，独立重复4/4 | 支持积分精度影响静态稳定；静态夹具不是学习策略抓放 |
| 009参考、010语义、011接触对齐 | 独立最终补录CHORD分别21/32、31/32、26/32；全任务+释放+放稳分别19/32、21/32、24/32 | 有真实无辅助完整成功，但预算/参考不同，不是公平算法排名；右拇指DP仍0接触 |

011保留基线独立补录：26/32 CHORD、32/32释放、28/32放稳，三项交集24/32。正式四个检查点CHORD为15/8/13/25（每点32），波动明显，不能只报最好一次。010/011仍不算自然拇指抓握已修复。

[局部训练汇总和曲线](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/reference-repair-20261029-001/training-summary-001/report-002/summary.json)。

### 3.3 三路同数据、同预算实际物体比较

共同条件：public44 loaded005、249源帧/20Hz、seed42、各自从零；256world、SAC replay262144/min131072/batch512；各2,097,152步、每collection8次更新。有限桌面、200Hz物理、20Hz控制、0.5倍源播放；每524288步保存评估，4×32world，最后checkpoint为主结果。辅助逐步0；全部失败尾迹保留。

| 最后模型 | 完整物体成功 | CHORD | 平均位置误差 | 平均角误差 | 本地ADD-AUC |
|---|---:|---:|---:|---:|---:|
| 官方IK | 0/32 | 0/32 | 48.17cm | 35.92° | .3460 |
| DexPilot-chain rebuilt | 0/32 | 0/32 | 14.11cm | 11.46° | .6382 |
| multi-tip | 0/32 | 0/32 | 18.74cm | 12.39° | .6385 |

不是普遍拿不起来：official/multi-tip首次失败前物体原点抬升>20cm的world分别28/32、31/32。但multi-tip29/32在释放附近首次失败；DexPilot中间checkpoint出现过46.52m均误差，保留掉落长尾。最后较低误差不能证明训练稳定。此表不是官方隐藏GT成绩，也不是多seed统计胜负。

![三路loss](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/report-mainline-001/loss-curves.png)

![三路评估学习曲线](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/report-mainline-001/error-learning-curves.png)

三路rebuilt-chain与multi-tip的时序项分别平方和/平方均值；22DOF下相同系数有效强度不同。协议在结果前冻结；公平训练预算不表示离线目标函数和求解成本相同。后来找到的原chain不能追溯替换此前方法身份。

### 3.4 几何独立2×2消融

固定官方IK及上述训练预算。normal只改两处左手法向；support只改source204之后的共同坐标与放回支撑参考；指关节不变。raw复用主线。没有混入新接触IK或语义奖励。

| 条件 | 正式物体跟踪 /32 | 正式CHORD /32 | 独立补录放稳 /32 | 独立补录全任务且放稳 /32 |
|---|---:|---:|---:|---:|
| raw | 0 | 0 | 15 | 0 |
| normal | 0 | 0 | 7 | 0 |
| support | 14 | 8 | 24 | 12 |
| normal+support | 1 | 1 | 25 | 2 |

support在该样本确有实际完整成功收益；normal没有，组合也不优于support。support改变了参考，所以own-reference分数与统一物理放稳须分开看。各列若来自不同批次不能逐world拼交集。全部四个checkpoint保留。[完整实验报告](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/experiments/nvidia_v2d/THREEWAY_ROLLOUT_20261029.md)。

### 3.5 跨样本、共同场景与03训练

- 原chain在37右手指尖均误差9.33mm，官方7.97mm、multi-tip4.99mm；但37五种zero残差参考伺服均0/4、全shape手物接触0/604。几何更准没有转成实际物体收益。
- 对03/19各自逐对象PCA再拼接，会人为引入相对中心偏差均值33.62/33.34cm；一次共同scene变换保持到数值精度。本次正式提交已使用共同变换，不能继续将旧拼接风险当成现有漂移已定位根因。
- 03锅+勺：313帧，131,072步、64world、seed42；两个checkpoint各4world无辅助评估均0/4。最终位置10.120cm、角31.799°、本地AUC .24700，锅4.915cm、勺15.324cm。
- matched zero/policy另批对照：位置10.088/10.129cm，角30.887/31.849°，均0/4。末1/4训练episode平均1.74步；actor7155次、critic14308次实际更新。支持“优化器确实在跑”，不支持“当前训练带来任务改善”。

三路和几何六fresh训练合计12,582,912步、24次正式评估/768world，另6批32world完整接触补录。历史可恢复成本审计下界30,813,184步，加03为30,944,256步；不是项目全部成本，早期不完整日志、人工/离线成本另列，不把累计checkpoint步数简单相加。[逐run成本表](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/track3-reset-20261029-001/contract-cost-audit/runs.csv)。

## 4. 这次正式结果与漂移有多大

| 官方指标 | 本队分数 | 方向 |
|---|---:|---|
| AUC | 0.15786 | 高好 |
| SP-SR | 0.05000（5%） | 高好 |
| MP-SR | 0.00000 | 高好 |
| RPE | 25.01154cm | 低好 |
| MPPE | 20.82605cm | 低好 |
| CD-O | 0.51105cm | 低好 |

这些是六榜已评分成功的成绩。排名会变化，本报告不把旧名次当现榜；CD-O单项排名不能作为总排名。此前核查官网只汇总六指标、默认按AUC排序，无已公布六指标加权总分。未找到明确的官方Track3 baseline分数，不能说已超过或低于该baseline多少。

本轮直接读取**实际提交来源的20份diagnostic.npz**，对全部帧的actual与重建参考计算位置差，再逐集等权平均：**25.6809cm**；最高episode14为66.21cm，其次33为40.36cm。03为10.14cm，37为12.95cm。这是新增的事后诊断，没有修改rollout，也不等于官方隐藏真值MPPE或RPE。

![本次提交的逐集本地误差与03/37全时序](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/progress-report-20261030-001/submitted-rollout-local-errors.png)

[逐集CSV](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/progress-report-20261030-001/submitted-rollout-local-errors.csv)；[重算脚本](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/progress-report-20261030-001/build_stats.py)；[原始来源SHA与汇总](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/progress-report-20261030-001/summary.json)。

## 5. 为什么漂移这么多：已有证据与未知部分

“漂移”需区分输入参考漂移、实际状态跟踪偏差、释放失稳，以及相对隐藏GT的误差。物体不动而参考移动也会产生大误差；不能仅凭误差曲线称物体在数值发散。重建参考误差与控制误差也不能直接把两个均值相加解释官方MPPE。

| 环节 | 实际证据 | 能说明什么、还不能说明什么 | 最便宜区分实验 |
|---|---|---|---|
| 上游深度/相机时序 | public44左手前100ms，对齐前Z变化+4.67mm、对齐后−128.55mm；静态桌面预测高度跨度178.88mm，相邻最大跳70.02mm | 证明该样本逐帧深度对齐引入时序不一致；不是20集绝对重建误差的真值统计 | 固定手尺寸/共同camera，按静态背景和可见深度比较前后；不先长训 |
| 局部手重建 | 0.4s右拇指投影已有明显偏差；撤手附近局部关节点存在大单帧变化；8.4s右拇指MANO实际存在 | 手重建确有局部问题，不能把机器人拇指不参与全部归给它；投影合理也不证明深度接触正确 | 同帧RGB→MANO→机器人IK→actual四层分别测位置与接触 |
| 手物映射 | 011可完成任务但右DP持握接触0；03右手勺接触只9/9/15/11帧，双拇指0 | 几何映射与奖励可形成非自然接触；有接触pair不等于承载力 | 保持参考比较腕/指动作、shape接触与可校准力贡献 |
| reset/伺服/碰撞 | 37 reset腕等于参考，首步Z却偏112/82mm；仅过滤ground后左97mm、右仍82mm，物体不改善；原joint target Z未被clip改写 | 支持物理/控制执行链在首步就偏离；排除了单纯目标clip及“关ground即可修好”，完整分量仍未知 | 同步记录reset→命令→PD/碰撞→actual第一步，做短单因素物理比较 |
| 释放与桌面支撑 | multi-tip29/32在runtime404–409失败；406→407实际角增量中位.13057rad，参考.01933rad；support单项使完整物体0→14/32 | 支持真实释放滑动/旋转和支撑参考影响，不能归咎纯阈值或参考跳变；改法向不是充分修复 | 固定actor/动作比较释放姿态、桌面碰撞和支撑参考，并保留失败尾迹 |
| 失败后的长尾 | 三路全时序图有少数world离开有限支撑后持续落下，完整尾迹达数十米；Dex中间模型均误差46.52m | 解释为何均值很大、world0视频却未必极端；不是所有world都有同样漂移，也未按失败截断“美化” | 同时报均值/中位/P95、首次失败时刻和Z尾迹；继续保存全量 |
| 学习与迁移覆盖 | 正式20集仅两个actor；03后段平均episode1.74步，matched policy未改善 | 强烈提示任务学习及迁移不足；不能量化为某因素占官方分数的百分比 | 在03/37先解除首步可执行性问题，再做相同场景训练vs迁移actor有限预算比较 |
| 配置迁移 | public44 actor原半速、有限support；正式单体原速、ground开启、无有限support。正式双体物理100Hz、单体200Hz | 已存在训练/执行分布变化；不是严格复现旧成功条件，因果影响未独立量化 | 同一actor/样本一次只改speed、support或物理步频；不移植44专属release407 |
| 隐藏GT/尺度/mesh | CD-O较低，但它单独允许尺度配准；物体轨迹仍差 | CD-O不能证明绝对尺度、轨迹和所有物体局部几何正确；尚无法完整拆分隐藏GT重建误差与控制误差 | 不使用GT mesh；以可见背景/深度及共同坐标校验，分别报重建代理和执行误差 |

结论是：**当前不是单纯“再训练久一点”或“让拇指碰上”就能解决。已有参考时序、物理首步、释放支撑和跨任务策略不足的证据，但尚无唯一根因解释全部20集。** 当前投入优先控制/参考可执行性及任务训练覆盖；自然抓握单列，不作为推进官方指标的额外门禁。

[深度时序审计](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/experiments/nvidia_v2d/STATIC_DEPTH_AUDIT_20261027.md)；[手重建审计](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/experiments/nvidia_v2d/HAND_RECONSTRUCTION_AUDIT_20261029.md)；[首步与原chain诊断](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/experiments/nvidia_v2d/TRACK3_RESET_20261029.md)。

## 6. 官方评分标准详解

以下按本次使用的官方Submission Kit实际实现说明，而非按指标名字猜测。[官方赛道说明](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)；[Kaggle Track3说明](https://www.kaggle.com/competitions/v2d-challenge-track3-sp-sr)。

### 6.1 评分输入与先对齐什么

Track3交付由视频重建得到的物体mesh，以及学习策略在物理仿真中产生的**实际物体轨迹**。本次指定20集、36个episode-object、9,120行，轨迹米制、四元数xyzw；mesh与pose必须同body系。轨迹参考回放、运动学动画或物体扶持轨迹不能冒充无辅助actual rollout。主办邮件明确不得把公开GT mesh用于Track3的配准或mesh评分。

姿态评分先将每个candidate mesh与主办隐藏扫描做body注册：表面质心/RMS半径、3,000表面点及刚体配准，得到anchor/坐标轴对应。然后每个world只根据初始帧确定一个**全场景共同刚体变换**，作用到整条轨迹。非对称物体可用于确定旋转；双轴对称情形结合对称轴和物体间方向。不能逐帧对齐来消掉运动误差，也不能给双物体各自独立修轨迹。注册时的尺度用于body对应，不会把rollout平移重新缩放到正确尺度。

对称性按官方对象定义：beige/blue cup绕轴连续旋转等价；锅盖允许0°/180°。表面/关键点的连续轴旋转以360个1°采样搜索；角误差对连续轴使用轴夹角。不是所有物体都任意允许旋转。实现：[注册、初始场景对齐与对称](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/v2dlb/track3_pose.py:54)。

下面以对齐后的实际位姿为 `(R̂[t,b], p̂[t,b])`，参考为`(R[t,b], p[t,b])`；`b`为物体，`t`为帧。单个pose文件只有world0，官方代码仍支持world维度。

### 6.2 AUC：1–9厘米范围内的表面运动跟踪准确率积分

每帧每物体，把同一组参考mesh表面点分别放到实际与参考位姿下，取对应点距离均值ADD，并在该对象允许的对称旋转中取最小：

`ADD[t,b] = min(S in symmetry) mean_x || R̂[t,b] x + p̂[t,b] − (R[t,b] Sx + p[t,b]) ||`。

对每个物体，统计`ADD < τ`的时间/world比例，τ依次为1、2、…、9cm。将九个准确率放在`linspace(0,1,9)`上做梯形积分，再平均各物体。因此可以写成`(a1/2 + a2 + ... + a8 + a9/2)/8`。端点半权，**不是九点简单平均，也不是9cm以下帧数比例**。

范围0–1，高好。本队.15786意味着这个多阈值准确率曲线的面积很低，不能解释为15.786%的完整抓放任务成功率。[阈值和积分实现](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/v2dlb/object_pose.py:24)。

### 6.3 SP-SR：较宽松的整段均值成功判据

先算位置误差`e_p[t,b]=||p̂−p||`与对称处理后的角误差`e_R[t,b]`，分别在时间上平均，再跨物体平均；同一episode同时满足：

- 平均位置误差 **≤0.1m（10cm）**；
- 平均角误差 **≤0.5rad（约28.65°）**。

每集得到成功/失败，最终对20集等权平均。本队5%在当前单world、完整20集口径下对应1/20集达标，但官方总分不告诉我们是哪一集。它允许局部帧超过阈值，不检查拇指、放稳或撤手是否自然；也不能认为每一帧都成功。[SP/MP实现](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/v2dlb/track3_pose.py:184)。

### 6.4 MP-SR：每个物体都须满足更严格的位置误差

对每个物体先时间平均，要求**每个物体**同时满足：位置均误差严格 **<3cm**、角均误差严格 **<30°**；该集所有物体都满足才成功，再对20集平均。

与SP比较，MP的位置阈值更严，角阈值反而略松；MP不允许靠另一个物体误差小来平均掉坏物体。它同样不是每帧判据。本队0表示无episode满足这组条件，不等于从来没拿起过物体，也不与public44某次CHORD成功矛盾。[代码](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/v2dlb/track3_pose.py:190)。

### 6.5 RPE：多个物体相对关系的误差

对每对`i<j`，计算`d̂=R̂_i^T(p̂_j−p̂_i)`、`d=R_i^T(p_j−p_i)`，比较它们的距离，按第一物体允许对称性最小化；平均时间、world及物体对，再从米转厘米。连续轴对称时比较轴向高度与径向半径，忽略合法绕轴等价。

它评估例如“勺子相对锅在哪里”，**不是相邻时间帧的相机相对位姿误差**。单物体没有pair时记0，仍参与20集均值。本队25.01154cm；因为4集单体贡献0，若其余16集就是全部双体，则双体episode平均RPE约31.2644cm（由总分推算，不是额外官方分项）。[代码](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/v2dlb/track3_pose.py:151)。

### 6.6 MPPE：六个物体附着关键点的运动误差

每个物体取body系`±X、±Y、±Z`方向、离中心**5cm**的六点。将六点分别用实际/参考位姿变换，计算平均对应点距离，取合法对称旋转的最小值，再平均时间、world及物体，单位cm。

它同时反映平移和旋转；不是机器人指尖误差，也不是原mesh所有顶点误差。半径必须5cm，不能拿训练奖励里的单位向量重算，否则旋转杠杆臂会放大20倍。本队20.82605cm。[六点与半径](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/v2dlb/object_pose.py:30)；[MPPE实现](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/v2dlb/track3_pose.py:192)。

### 6.7 CD-O：独立评估物体形状

每个episode-object的mesh分别进行表面质心/RMS尺度归一化和刚体配准。3,000面采样点用于对齐，identity+PCA共25个初始化、最多40次ICP、保留最近90%匹配；10,000均匀面采样点用于评分。最终是：

`CD-O = 100 × 0.5 × [mean_x min_y ||x−y|| + mean_y min_x ||y−x||]`。

在参考物体尺度下用厘米报告，是双向平均最近邻**距离**，不是平方距离，也不是最大Hausdorff距离。各episode-object等权平均，共36项；双体集有两个贡献项。它允许单独尺度/姿态注册，因此本队.51105cm说明配准后平均表面差约5.11mm，但**不能证明物理轨迹、绝对尺度、细把手或所有局部碰撞几何都正确**。[CD-O实现](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/v2dlb/object_mesh.py:197)。

### 6.8 聚合方式与本地成功率的区别

五项pose指标先每episode算分，再对20集等权平均，不按帧数加权；长视频不会仅因帧多获得更大episode权重。CD-O按36个episode-object平均。[pose聚合](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/metric_code/track_3/AUC.py:847)；[CD-O聚合](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/submission-smoke-001/official-kit/v2d_submission_kit/metric_code/track_3/CD-O.py:289)。

本地CHORD含腕跟踪和逐帧条件，本地object-only要求完整序列每帧阈值；本地ADD-AUC通常对重建参考评估且不经过官方隐藏扫描注册/对称处理。这些都是代理；**本地0/20、官方SP5%、public44局部26/32可以同时成立**。手动作自然度、拇指参与、释放、末段放稳是附加质量，分别报告，不冒充六项官方评分。

## 7. 过程视频、结果视频和数据入口

视频均为现有文件，经ffprobe检查。public44原片12.45秒；半速物理控制导致部分实际视频24.8秒，源时间和仿真时间分开显示。模型估计与参考动画不当成actual policy。

| 内容 | 文件 | 时长与身份 |
|---|---|---|
| public44原RGB | [原始视频](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/inputs/public44-source/source-video/episode_000044.mp4) | 12.45秒，人类示范 |
| 原RGB与MANO手部重投影 | [重建审计视频](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/hand-reconstruction-audit-20261029-001/source-vs-mano-full.mp4) | 12.45秒，训练前估计，无机器人policy |
| 重建→两路重定向过程 | [历史过程视频](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/historical-demo/01_手物重建与两路重定向.mp4) | 12.45秒，参考运动，不是执行成绩 |
| 三路公平比较完整actual | [三路world0失败视频](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/threeway-world0-native-video-001/threeway-world0-full-actual-FAILED.mp4) | 24.8秒，world0预选，三路无辅助，全部完整任务失败 |
| public44局部010成功展示 | [010实际成功视频](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/reference-repair-20261029-001/trained010-visual-001/learned-policy-world0-SUCCESS.mp4) | 24.8秒，无辅助；成功候选展示，不是总体/跨任务成功率；右DP仍未修好 |
| 正式样本03双物体诊断 | [03实际失败视频](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/coverage-20261030-001/visual03-world0-002/episode03-learned-world0-FAILED.mp4) | 15.65秒，预选world0，4/4失败，无辅助；属于提交前独立诊断，不冒称与提交那次逐位相同 |

**正式提交20集actual状态、动作和接触均有保存，但尚未全部渲染为20个视频。** 上述03和历史视频均不替代正式提交数据。完整actual记录按`episode-XXXXXX/rollout-001/result/diagnostic.npz`组织，含actual bodies、achieved/reference pose、policy actions、joint targets/state、辅助量、全部shape contacts和source time。

![三路开始/持握/释放/结束](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/threeway-world0-native-video-001/start-hold-release-end-contactsheet.png)

- [完整20集actual与收据目录](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/full-submission-20261030-001/complete-evidence.json)
- [实际提交pose包](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/full-submission-20261030-001/candidate-complete-001/track3_submission.parquet)
- [CD-O派生提交包](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/full-submission-20261030-001/candidate-complete-001/track3_submission.cd-o.parquet)
- [三路全部训练CSV](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/report-mainline-001/all-training-metrics.csv)
- [三路全部评估CSV](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/threeway-rollout-20261029-001/report-mainline-001/evaluation-curves.csv)
- [03训练原始CSV](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/coverage-20261030-001/train03-official-001/train-metrics.csv) / [03评估CSV](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/coverage-20261030-001/train03-official-001/evaluation-metrics.csv)
- [03最终checkpoint](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/coverage-20261030-001/train03-official-001/train/policy_131072.safetensors)
- [手重建逐帧CSV](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/.e2e/zero_success_20261027/hand-reconstruction-audit-20261029-001/all-frames.csv)

SwanLab本地5092、TensorBoard6006本轮均实际返回HTTP200。后续训练/评估需继续完整保存loss、曲线和checkpoint；此前缺失历史loss明确保留缺口，不以事后曲线冒充。03及完整20集评估双端逐标量回读已验证。

## 8. 下一步决策

当前最值得投入的是**首步控制可执行性和正式样本的任务学习覆盖**，并行完成比赛最终交付材料。保留完整六榜基线，不以自然抓握完美作为额外门禁。

1. 在03/37先做短reset→命令→actual状态/接触审计，固定actor与参考，预算0训练；同时把速度、support、100/200Hz差异分开比较。首步偏移不改善就停止同方向重复修奖励。
2. 有可执行性区分信号后，选择代表单/双体做有限预算本场景训练vs冻结迁移actor，预登记同预算、全部checkpoint与失败；直接看实际物体运动，而不是仅优化离线指尖误差。
3. 把上游时序/接触修复作为独立因素，先验证别的片段，再决定是否批量重建。已有support单项有效而法向组合退化，不能默认把所有“修复”一起打开。

这三项是下一轮建议，本报告未启动它们。本轮新增的是对已有20集提交的透明统计、视频索引与官方评分解释。

## 补充：ACE支线（本报告初版遗漏）

ACE-Ego-Hand已在自采纸箱视频完成修正版重建输出，旧适配的MANO mean错误已修并使旧衍生结果失效；尚未完成Track3策略训练和实际rollout对照，不在本次正式提交内。它不是仅有接口，也不是已经验证的比赛替代方法。详见[ACE接入专项复核](/home/fmc3/Documents/workspace1/EgoWorkSpace/video_to_data/experiments/nvidia_v2d/ACE_INTEGRATION_AUDIT_20261030.md)。
