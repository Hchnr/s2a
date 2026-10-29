# 038：有限真实 clean 残差训练与开发任务验证 — 2026-10-23

本项是[036总目标](036_project_goal_20261023.md)的条件协议。**只有034新数据资格、[037完整模型证据](037_clean_bridge_continuation_20261023.md)及[039原生mask独立审计](039_native_mask_audit_addendum_20261023.md)均通过，且037模型worker实际正常退出/GPU释放、039 CPU owner实际正常退出，才可prepare并执行本项。** 037原owner因独立checker的无效槽mask假设错误为False，保持原记录，不能要求或伪造该历史封套变True；联合资格必须引用039完整补充审计。当前不能根据本计划的存在推断该资格已通过、真实训练已启动或任务已有收益。实现、配置、source closure和独立准备审查仍须完成。

本项仅研究冻结8400的世界模型输出适配，S2A及固定OSC保持同一控制合同。完整训练和开发协议不能代替最终fresh验收，最终20/24、各任务4/8、末40端点连续成功门槛不变。原030/025/026/028等False、035未完成运行和全部历史消耗保留，不由本项追认。

## 研究问题与固定候选

029显示旧027在第二noise seed的teacher误差增加51.42%，在所选自由生成上下文的actual-clean误差增加1.60%。新实验直接监督**自由采样上下文的clean残差**，扩展同步native来源和训练noise，并用一次固定的完整生成与任务实测检验，而不继续以teacher拟合改善作为终点。[029:25](../../reports/wam_free_trajectory_diagnostic_20261021.md#L25)、[029:39](../../reports/wam_free_trajectory_diagnostic_20261021.md#L39)、[030后续:45](030_synchronous_native_training_20261021.md#L45)

模型固定8400非EMA、SDPA、无compile、单rank、guidance1.0、shift5、35步、H40/20Hz、normalization31/projection0、仅前10槽有效。继承037实际通过的隔离Cosmos源码、mask合同、解释器和数值语义，不直接使用正在变化的外部工作树。权重163文件stat及小metadata SHA沿用037核验边界；不将stat写成此次完整大权重内容验证。

唯一流程为：

```text
全部150个训练anchor × 两训练noise
    → raw轨迹D0（300请求） → 固定一次ridge fit W1
    → 同一300输入/初噪声的W1轨迹D1 → D0/D1等权union → 固定一次fit W2
    → 唯一最终候选W2 → 完整生成state诊断＋有限成对开发任务
```

W1只是事前固定的中间采集策略，不与W2按validation或任务成绩择优。固定两轮指D0/D1两类采样上下文和W1/W2两次生产fit，**不新增W2训练轨迹或第三轮fit**。不得以teacher指标、某个task、seed、anchor或阶段的局部成绩决定截断、加轮、缩放、换ridge或换候选。

## 来源、训练监督与隔离

训练仅使用034固定的全部30条、每task10条实际同步native轨迹、每条5个当前anchor，共150张当前canvas；第五anchor为N+40，前四按原规则保留。采用034补充独立审计和final peer的真实引用，原034 owner False不能当作True封套。原source roster已经排除fresh、历史开发来源及六个validation整demo；prepare须再次按已封存身份及initial SHA核该排除关系，不读取fresh动作、观测或性能。[034:5](034_settled_native_training_20261022.md#L5)、[034:7](034_settled_native_training_20261022.md#L7)、[034结果:26](../../reports/settled_native_training_20261022.md#L26)、[030:11](030_synchronous_native_training_20261021.md#L11)

旧六validation EP到native demo的保守来源映射已知，但其native initial同步未证实且没有可用initial SHA；不能为满足接口伪造该身份。prepare必须完整排除task0的demo16/32、task1的demo4/27、task2的demo14/24，联合原historical禁入demo及034实际登记的forbidden initial SHA核别名；fresh仍用原12个真实metadata initial SHA。validation另记`initial_sync_certified=False`及其真实EP/anchor来源，不把demo级排除写成物理同步认证。此边界在训练/评分前固定，与030原保守排除合同一致；既有validation state诊断也不因此变成新的同步物理证据。

034提供实际source/TCP pose、opening、state128、action128、masks和物理时间；每个anchor应有41个实测端点及40个有效local delta标签。slot9监督为下一实际端点opening，与原native gripper command分开，不把command或HDF5旧obs当实测future state。局部delta按SE(3)右乘、column-rot6d、米和0.05秒定义。当前实现证据见[native_training.py:70](../../src/s2a/native_training.py#L70)、[collector:99](../../scripts/collect_settled_native_training.py#L99)。

模型采样仅接收对应anchor的**当前PNG和正常任务语言**，空task_plan及其余037接口字段固定；不传future state/action、专家command、物体/contact/predicate、实测state或future RGB。034没有future RGB，本项也不新增或读取future RGB、不启动训练dataloader。自由采样中video latent自然演化，不用真实future video替换。完整raw阶段的300个生成落盘、来源/预算审计和正常模型退出后，独立CPU才读取034 future标签生成监督；D1亦先完成全部采样与退出，再与同一标签绑定。future标签不能进入service.prepare/execute或bridge。

`x0`为034实际40步action128经冻结normalization31和原clipping规则得到的active10目标。记录全部标签的mask、变换身份及clip计数，不删去clip较多、难拟合、静止或terminal行，不因fit质量换demo。training loss属于normalized Action128空间，rotation6d误差不是SO(3)弧度，position槽误差不是米。

开发validation固定沿用028/029原六EP：1254、1126、868、1139、1209、1176，各4个anchor；每anchor评测seed42/43。它们已有结果被查看，只能称开发诊断，不能称无偏新验证、同步native新采集或预训练未见。原train六EP不加入新训练；30条034为全部fit来源。旧六EP当前PNG、41状态窗口、mask、来源SHA及安全迁移定位在prepare时逐项确认；**本计划未重新核完这些缓存，当前可用性待prepare核实，缺失即停，不换EP或重新采未登记数据**。validation不进入D0/D1、特征中心/尺度、ridge或任何fit；唯一W2冻结且全部96＋3完整生成退出后，CPU才打开validation future状态评分。[028:12](028_scaled_readout_state_20261021.md#L12)、[028:13](028_scaled_readout_state_20261021.md#L13)

## 训练noise与每行目标

训练noise根固定101/102。准备阶段在读取训练future标签和初始化模型之前，对稳定的`task/demo/anchor_frame/root_seed`身份采用SHA256确定性派生两个63位非负整数种子；序列化字段顺序和派生算法写入新配置并冻结。枚举全部300个实际训练seed，要求互不重复、与42/43无交集；碰撞或碰到42/43即prepare False，不按结果改种子。不能只登记两个根而让实际执行seed游离在清单外。D0和D1相同anchor/root的实际seed及完整初始FP32 noise必须一致。NoiseRoster登记这300个实际种子，purpose明确training；评测purpose仅允许42/43。[030:31](030_synchronous_native_training_20261021.md#L31)、[NoiseRoster:44](../../src/s2a/wam_clean_capture.py#L44)

对全部35个实际正sigma调用、每调用40个action token，以原模型当次masked velocity定义：

```text
C0 = actual_sample_FP32.astype(FP64) - actual_sigma * raw_velocity.astype(FP64)
features = [hidden1280, C0_active10]                         # 1290列
target_clean_residual = normalized_native_x0_active10 - C0_active10
phi = [(features - mu_train) / global_L2_RMS_train, 1]
W = argmin sum_i weight_i * ||phi_i W - target_i||² + .1 * ||W||²
v_deployed = raw_velocity - (phi W) / actual_sigma          # 最后按原dtype一次舍入
```

不用BF16 packed action代替实际FP32 sampler state，不从整数t推sigma、不加时间多项式/独立sigma头，不作1/sigma损失重权、residual clip、PCA、逐列白化或validation调尺度；terminal sigma0不调用head。C0是理想FP64定义，不冒称scheduler混合精度转换的实际x0。[clean定义:73](../../src/s2a/wam_clean_readout.py#L73)、[features:110](../../src/s2a/wam_clean_readout.py#L110)、[部署:299](../../src/s2a/wam_clean_capture.py#L299)

**D1仍用W1轨迹当次原模型raw velocity计算C0和target**，不用deployed velocity计算C0。W2学习从原base到目标的完整clean修正，不把W1修正扣两次、也不将W1/W2叠加。推理只用当前hidden、C0和冻结参数，不读训练目标。

每task、EP、anchor、noise、35节点及40endpoint等权。由于每层数量固定，D0有`30×5×2×35×40=420000`行，每行权重1/420000；W1只用D0。W2对D0/D1两轮各赋1/2质量，完整union840000行，每行1/840000，不另给terminal较低/较高权重，不抽样或重复行。W1和W2各自在其完整训练集合计算唯一mu和全局向量L2 RMS，验证不更新统计。每次系数1291×10，mu1290、scale1，合计14201个学习数；bias也按lambda=.1正则。

生产fit每轮只solve一次；复用[流式两遍原语:166](../../src/s2a/wam_clean_streaming.py#L166)，固定row order与两遍features/target/weights完整SHA一致。每轮独立CPU审计重新解析全部源、标签、权重和归一化，使用分块增广QR独立重建一次同一ridge解（不复用生产normal方程/solve），全量行均参与；小矩阵AugLS黄金测试验证该审计路径。数值比较固定FP64 `rtol=1e-9, atol=1e-11`，仅用于实现等价性，不放宽质量门槛。若真实范围使此合同不适用，保留False并另登记，不能运行后调容差。独立重建不能用抽样子集替代全量。

## 分阶段模型与资源预算

以下是**本项新增上限**，不包括037的7/245或旧035未知消耗。预定矩阵完整时生成数量和denoise必须精确相符；发生来源/协议/数值/进程错误则保存实际已消费和未执行分母，不补跑或冒称完整。

| 阶段 | 完整模型请求 | denoise | 新模型初始化 | CPU fit/独立重建 | 物理步 |
| --- | ---: | ---: | ---: | --- | ---: |
| D0 raw训练上下文 | 300 | 10500 | 1 | 随后W1各1 | 0 |
| D1 W1训练上下文 | 300 | 10500 | 1 | 随后W2各1 | 0 |
| 唯一W2完整生成诊断 | 3zero＋96raw/W2配对＝99 | 3465 | 1 | 0 | 0 |
| 训练/生成小计 | **699** | **24465** | **3** | **生产2＋独立2** | **0** |
| 新成对开发raw6＋W2六 | 900 | 31500 | 1 | 0 | 7200 |
| 完整条件预算合计 | **1599** | **55965** | **4** | **生产2＋独立2** | **7200** |

无新增teacher denoise、future RGB、HDF5/native重新采集、fresh性能或真机预算。CPU准备/fit/checker及OSMesa均空CUDA/LD；四个模型阶段逐个使用`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`，不并发持有多个实例，不沿用历史数字6。另一允许UUID不自动扩展本预算。端口、解释器和确切命令在prepare逐阶段冻结并预检，不能以端口失败自动启动副本。

各D0/D1模型worker最多86400秒（各24h），完整生成worker43200秒（12h），开发单模型service整体172800秒（48h）；每个CPU生产fit及独立重建各7200秒，其余阶段独立CPU来源/生成审计各1800秒；HTTP单请求300秒，无自动重试。此上限是在037带完整capture实际运行已观测约3.6秒/denoise、35步约2分钟之后固定；300请求据此约10.5h，900开发请求若同速约31.5h，不能沿用旧7200秒或8h限制。prepare绑定实际计时来源，这只是带capture路径的阶段观测，**不作为普通无capture推理速度或实时性能结论**；精简sink潜在提速不用于减掉任何请求预算。

prepare须核可用磁盘/RAM和线程配置，冻结CPU BLAS线程上限8、单fit分块最多1400行，完整D×D工作矩阵为1291维；独立QR也分块，不将840000×1290 FP64行全部保留在RAM。上述时限是资源上限，不是性能成绩或完成保证。

每阶段独立owner从真实Popen持有handle至actual wait/reap，记录PID/startticks/UUID/env/源码与参数SHA。只处理自有child；超时/异常保存False和原始退出状态，不补造退出收据、不kill未知进程。模型正常退出后逐次记录GPU释放观察，最长120秒，非空或超时保持资源合同False；独立CPU审计及下一次模型初始化前必须闭环。D0退出审计→W1 fit/独立核验→D1退出审计→W2 fit/独立核验→最终生成退出审计/评分→开发，所有依赖串行。

## 精简capture与独立证据

不直接复制035每节点完整video/flat-array落盘规模到600条训练请求。新的sink只能减少保存的冗余数组，**不修改037已通过的真实bridge计算、原模型/RNG或每节点一次head/denoise/velocity委托语义**。实际sink及服务包装源码进入本项新source closure，必要CPU非干扰/布局/拒错测试后冻结。

每个35节点保留：request/anchor/seed/轮次/系数SHA、step index、实际sigma及网络时间、action offset与原dtype、hidden40×1280、实际FP32 action sample40×128、masked raw和真实deployed action velocity40×128、C0、clean residual和velocity delta，以及inactive118为零/vision未改的独立逐值检查结果。全量packed action与原head输出可按合同保存；未来标签另存CPU侧来源索引，不混入在线point。

对每节点巨型packed vision、完整flat sample/raw/deployed等不重复存多份数组，而在实际委托边界读取其shape/dtype和完整原字节SHA；分别记录vision及inactive切片相等性检查。每请求另保存完整初始FP32 noise/condition、真实request/response、sigma trace和35节点ledger，支持D0/D1以及raw/W2配对初始条件精确核验。不得将“仅有SHA”描述为可从文件独立逐值重建全部video；完整flat数学证据仍由037代表性资格提供，本项全部action数值与拟合输入可独立重算。训练sink完整消费清单和参数/源码前后SHA均保留。

实际压缩后字节数、当前磁盘余量、单请求峰值内存和原bridge暂存35点的内存上界尚未实测，prepare只根据已知shape/dtype做保守上界并核资源，真实运行再记录实际值，不能预报为已通过。若无法证明完整600请求证据能保存，prepare False，不通过事后少存节点、删训练行或切到不登记的capture实现解决。

## 唯一最终候选的完整生成state诊断

W2及mu/scale SHA冻结后，先执行原三份开发输入的zero head完整输出精确回归，再执行六validation EP×四anchor×seed42/43×raw/W2，共96配对生成。每对当前PNG、任务、FP32初始noise、mask、首condition latent、token及采样协议精确一致；保留全40×128 raw和normalized输出，zero失败立即停止。3zero与历史身份来自037，旧缓存须在prepare确认存在及SHA相符；不因失败换zero样本。

独立核验全部99生成、3465 denoise、每次35实际sigma/单次委托、原模型参数不变及正常退出后，才只读六validation future状态并在H8/H16/H40完整40端点上用SO(3)/SE(3)重算。raw、W2、hold及constant-velocity共用同mask、锚点和物理时间，不从量化action构造GT。按EP、task、seed保留全部结果。[state评分:686](../../scripts/verify_wam_scaled_state.py#L686)

完整保留028原state gate，不改历史False：H8/H16 joint score≤.90raw且严格优于hold/CV；H8/H16/H40九组件各≤1.05raw并同时满足+2mm/+0.01rad/+0.01opening保护；至少5/6EP两个短分数≤1.05raw、任一EP短分数≤1.25raw；两个seed分别两个短分数≤1.05raw。joint score继续使用`mean((p/.01)²+(r/.05)²+(opening/.1)²)`，不是RMSE之和。[028:15](028_scaled_readout_state_20261021.md#L15)、[独立门:1036](../../scripts/verify_wam_readout.py#L1036)

state gate与下面任务gate**分别报告**。它衡量对固定示范轨迹的接近程度，不是多模态任务能力的替代；本项在注册时明确增加一次有限任务实测。全部来源、zero/数值/掩码、输出有限及执行命令约束合同通过后，即使state gate False也执行下面这一轮固定开发矩阵，不因state成绩换候选或改预算。这不放宽原028判定，也不自动获得fresh准入。

## 固定开发任务矩阵及任务门

仅用历史三个`task0_init0/task1_init0/task2_init0`开发初态，绑定原fixture SHA `f522ee333827d51ab66b5b8d1c2a7ebd8ceca3832403b3c54dfa56a821fb2a8e`和source demo/initial/XML身份，sampling及环境seed42/43。每初态/seed各跑新raw和W2，共12次、每次600个20Hz端点、末40端点连续原生success为稳定成功；推理等待时物理暂停，成功不提前停止。不是12个新初态，也不是最终fresh集合。fixture、原seed43正确参考资格及当前迁移后的执行源码/依赖的可复用性**待prepare核实**；不合格不得以历史24/24代替当前资格，须先停本项并另登记所缺正控预算，不能偷偷新增物理正控。

S2A完整沿用原raw8/alpha1：H40、每8物理步重观测、完整保留生成旋转和opening、固定position_hold夹爪、原rotation response/OSC inverse .5/rate5、无governor；低层不看task/object/contact/success。任务语言只进模型。每次75请求，12次精确900请求/31500denoise，最大7200物理步。旧raw8的seed42 1/3、seed43 0/3仅作背景，本项**新raw6/W2六**才构成同源成对比较。[旧raw8配置](../../configs/wam_short_replan_8_alpha1_20261021.json)、[旧seed43配置](../../configs/wam_raw8_seed43_task0_20261021.json)

一个串行私有模型service按已登记case绑定raw或冻结W2，不并行接受不同variant；每case独立初始化sim、控制/夹爪/参考/滤波状态。顺序固定task0/1/2→seed42/43→raw/W2，全部source在运行前冻结。每对首请求保存当前相机/初始物理绑定及模型初始条件核验；后续轨迹和图像自然可不同。新控制源码若不与已资格原合同相符，不能仅凭首响应相同推断全轨迹资格。

**开发稳定门固定：W2六条至少5/6稳定、每task至少1/2。** 同时报告ever、terminal、stable、success时长、TCP/SO(3)/opening跟踪、限幅、物体进度、两seed及逐task差异和全部失败日志；不按task/seed拼策略。适配增益另以W2稳定总数严格高于本次new raw六条判定，不额外改写上述稳定门。若raw与W2均6/6，可报告两者均达到开发稳定门，但本项不证明适配增益，`adaptation_benefit_gate=False`，不能省略对照或追改“严格高于”标准。

分别保留`state_gate`、`development_stability_gate`及`adaptation_benefit_gate`。开发稳定门达到5/6且各task1/2，并且来源、控制资格、因果、数值、预算和进程合同完整后，才有条件**另行冻结**最终fresh候选及专用接入协议；不额外要求`state_gate=True`。这一准入区别在任何本项训练/评分前固定：028 state gate衡量单条专家未来轨迹的点态接近程度，不等于候选命令经过实际前向控制系统后的参考跟踪误差，也不等于多模态任务成功，不能用该窄指标阻止已预注册的任务目标验证。[001主要指标:44](001_state_to_action.md#L44)、[001执行验证:336](001_state_to_action.md#L336) state gate仍完整按028原门槛评分和分列诊断，旧028 False及其原不推广结论保留；本项不把False改为True或事后放宽阈值。适配增益标志False时不得报告适配优于raw。所有失败原样记录，不能直接进入fresh或在六条结果上继续挑W1、缩放、lambda、task策略。有限开发成功不证明预训练未见、一般可靠性、实时性或真机可用；最终20/24、各task4/8验收门槛不变。

## 必须先补齐的实现与准备检查

当前可复用的是`wam_clean_capture.py`真实采样桥、`wam_clean_readout.py`clean语义与参数、`wam_clean_streaming.py`全量两遍ridge，以及034标签和旧state/任务计分原语。它们没有自动提供本项来源、训练或任务资格。

1. 新prepare/配置/独立prepare checker：绑定034实际审计、037模型正常退出/GPU释放及039完整补充资格/正常退出收据，30 train/6 validation/fresh身份排除、150窗口与当前PNG、300派生seed、完整source snapshot、四阶段命令和新输出根；只读RelocatedArtifacts解析历史path/SHA，不改旧根或旧JSON。
2. 同步标签reader与stream factory：实际41端点/40mask、normalization31/clip审计、D0/D1 row identity与等权、完整两遍重放，拒validation/fresh及无效尾行。
3. 两轮真实采样worker与精简sink、正常ownership编排、两个CPU fit和全量独立增广QR审计；D0/D1实际初始化一致而后续latent允许分歧，raw C0与deployed C0不能混淆。
4. 唯一W2的96＋3生成/独立state checker，以及支持CleanCaptureBridge和冻结参数的私有开发服务；不能直接把旧FlowReadoutHook或硬编码数字GPU6入口当新clean服务。
5. 新raw6/W2六执行与seed-aware来源/因果审计，明确完整600/75请求和已验证S2A合同，不向fresh root写文件。必要测试覆盖真实脚本入口、未来输入拒绝、noise/row泄漏、W1/W2混用、全量权重及预算拒错，不堆占位数量。

prepare-only阶段应输出所有待核实项的具体通过/False及证据，不将“计划已有字段”当作数据可用或执行已成功。实现与相关CPU QA完成、实际source与准备审查冻结后才开始条件执行；所有新配置/代码/报告/产物留在本S2A，旧结果、外部Cosmos/数据/权重只读，已有输出拒覆盖。
