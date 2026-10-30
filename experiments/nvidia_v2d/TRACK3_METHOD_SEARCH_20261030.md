# Track3 定向检索与方法决策（2026-10-30）

当前优先交付共同双物体场景；新增方法不替代20集真实rollout覆盖。官方隐藏GT六指标未知。现有官方IK、原DexPilot-chain、Sharpa rebuilt-chain、multi-tip保持明确身份，旧比较见TRACK3_RESET_20261029.md。

## 瓶颈与已开源候选

|候选与原始来源|针对瓶颈/实质区别|代码、权重、依赖核验|最小对照与淘汰条件|本轮状态/决策|
|---|---|---|---|---|
|现有CHORD多刚体loader：[官方代码](https://github.com/nvidia-isaac/video_to_data/tree/main/robotic_grounding)|单对象桥接导致共同场景接口缺失；使用一次共同变换、一份手源和多body接触身份|本地loader和spawn_objects实际支持独立多URDF；已有Sharpa、重建束和冻结容器，无新权重|03全313帧相对SE3保持、4world完整物理，然后新的722维actor短训；缺body/非有限停止|已实现并完成物理smoke；作为主线|
|[SPIDER论文](https://arxiv.org/abs/2511.09484v3)、[作者仓库](https://github.com/facebookresearch/spider)|离线IK好却无动态收益；通过物理采样优化可执行参考，再供RL学习|有MJWP/Genesis/ManipTrans入口、Python3.12依赖；仓库展示Sharpa数据。优化核心无需预训练网络；本机尚未安装/跑通。发布数据的状态回放验证不等于控制回放|先原例，再同37输入/Sharpa/物理、固定15分钟优化预算；必须导出动作并重新物理执行。再以相同RL预算比较原参考与优化参考；不能保留对象辅助|优先外部对照候选，未验证；裁剪4秒/单物体示例不能直接当全长双体Track3实现|
|[DexMachina论文](https://arxiv.org/abs/2505.24853)、[作者仓库](https://github.com/MandiZhao/dexmachina)|跨形态与长时物体跟踪；接触/碰撞参考+衰减VOC课程|代码与手/ARCTIC资产已发布，需作者Genesis与rl_games分支、Torch2.5.1；README中RL eval仍列TODO。不是可直接复用的Track3权重|取接触目标/课程的单因素改动与当前CHORD比较；若只是重复当前VOC、引入新仿真器却不能导出全长实际轨迹则暂缓|方法级备选，当前架构已有类似VOC，不优先整体迁移|
|[ManipTrans论文/项目](https://maniptrans.github.io/)、[作者仓库](https://github.com/ManipTrans/ManipTrans)|残差学习补偿运动学重定向误差，支持双手和新URDF/数据集接入|有训练、环境、数据适配代码；Python3.8、IsaacGym Preview4、Torch1.13.1/CUDA11.7，与当前runtime隔离。未验证现成Sharpa可用权重|原例安装及一次物理重放先过；再同03全长双body比较。依赖不兼容或仅单物体输出则不扩展长训|可用替代实现，接入成本高于修当前loader|
|[C2Dex论文](https://arxiv.org/abs/2608.07045)、[作者仓库](https://github.com/K-Jie/C2Dex_code)|物体本地坐标中跨帧聚合稳定接触，再做交互几何重定向|仓库公开但核心reconstruction/retarget代码仍TODO；不能称算法代码已开源可运行。权重未核实可下载|仅在代码可运行后做固定参考/actor的接触目标单因素实验；若自行实现需标记“受启发”，不能称复现|本轮排除直接接入，保留思想候选|
|[CARD论文](https://arxiv.org/abs/2610.36539)、[项目](https://card-retargeting.github.io/)|显式接触力目标、视频接触预测，针对thumb接触/承载不足|论文2026-10-27发布；项目本次读取失败，未核实代码及Vid-HACO权重，状态未知|需先取得可运行代码/权重，验证视频接触估计，再评估任务实际轨迹；无代码不投入训练预算|仅候选，不宣传已开源或本地验证|

|[Open-CHOIR论文](https://arxiv.org/abs/2605.20992)、[作者仓库](https://github.com/hxwork/CHOIR)|重建手物深度失配；抓握深度修正+时序手物优化，直接改重建参考而非机器人奖励|实际有stage2/3源码、release权重下载说明；本轮未下载验证权重。Python3.10/Torch2.6/CUDA12.6，Dyn-HaMR/VIPE独立环境。主要支持单手单物体，双手双体是两次独立拟合，不保证共同坐标|冻结mesh/相机/原RGB；先44已审计的0.4/8.4/11.95秒及37同预算短片，对比独立标注投影、腕局部动态、物体局部接触距离；重定向/物理预算不变。改善投影却破坏共同frame或物体轨迹则淘汰|优先重建侧候选，仅已核实代码存在，未接入；不作为扩覆盖的前置条件|
|[Dyn-HaMR论文](https://arxiv.org/abs/2412.12861)、[仓库](https://github.com/ZhengdiYu/Dyn-HaMR)；[WiLoR论文](https://arxiv.org/abs/2409.12259)、[仓库](https://github.com/rolpotamias/WiLoR)|相机与手运动纠缠、遮挡/缺帧与局部拇指错误；全局时序优化与逐帧重建分别测试|两者代码已发布、README有权重路径；项目已有对应入口，本轮未运行替代重建。WiLoR本身不含时序模块，不能称时序优化。Dyn-HaMR含VIPE更新|先复核现用版本/权重，再同RGB/相机对照手源，独立标注2D+三维连续性；只修改手后端，不同时换物体mesh/相机/IK。无改善或耗时超预算不全量切换|优先复用已有实现，防止重复接入；重建误差与实际thumb承载分开判定|

## 具体发现与因果界限

- 双体原始相机逐值一致，独立PCA却破坏对象关系；共享变换是输入契约修复，不需要换大模型。
- 03新装配保存313帧与两对象，按对象分别估计1cm接触后，在每个手link slot选最近的有效对象，所有位置/法向/ID成组选择；这是几何启发式，非C2Dex、非接触力估计，也不能同时表示同一slot接触两个对象。
- 37旧记录显示零残差首步目标Z与参考Z相等（左-.186782、右-.129682m），实际q为-.074591/-.048171m。已排除“目标被位置clip改写”的假设；动态响应/碰撞/场景仍未知。03同构机器人打印的腕限位为±1e10，也不支持把偏移归因到腕Z下限0。尚未对37重新测全部动力学项，不宣称根因已解。
- 03旧单物体checkpoint在输入维度504→722处被拒绝；这是有效性阻断，不能裁掉第二对象观测强行包装迁移。零残差诊断明确跳过actor加载，不能算learned提交。

## 来源可复核性

2026-10-30读取原论文/作者项目/README；GitHub API tree查询返回403限流，保存失败，未把未知树当空仓库。随后git ls-remote确认HEAD：SPIDER44717007de41cbef7565dff7ff9f4453557a2d3d；C2Dex eae9248accaedd7a61dccad562f8048bb9e6c36f；ManipTrans a3d08cfe3c3a5868a7f057533bcaf759c5af4705；DexMachina adae5bf620c57723d185b2757ee3ce9656927c20。代码存在不等于本机可运行，未下载大模型或安装新仿真器。证据根`.e2e/zero_success_20261027/coverage-20261030-001/research/`。
