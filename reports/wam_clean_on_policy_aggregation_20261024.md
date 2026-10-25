# 056：自身策略训练轨迹聚合准备

总目标和原24条最终验收门槛不变；055性能失败，fresh关闭。依据同步validation末节点的teacher误差反转，固定一次D2分布覆盖检验。设计见 `plans/v0.0.1/056_clean_on_policy_aggregation_20261024.md`，显式配置为 `configs/wam_clean_on_policy_aggregation_20261024.json`。

## 已完成

`scripts/prepare_wam_clean_d2.py`在真实数据上正常exit0，生成current-only清单。逐一核查原D0完整请求、实际31位seed计划、原current图像、initial记录及其数组文件SHA；没有解码任何未来trajectory。150个锚点、300个唯一训练种子、每锚点两种子，保留原请求顺序，排除评估seed42/43。模型输入和anchor清单不含trajectory、native commands或future state。

052策略的生产和独立完整QR由既有candidate_head合同读取核验，参数身份固定。清单 `outputs/clean_on_policy_aggregation_20261024/current_manifest.json` SHA256 `185bb7719f38f0b8e992913f70d67119f4f2df1d9fffc2d199f97b729124413c`。preparer自身检查收据 `proposal_verified_by_preparer.json` SHA256 `1c01bf669620bd3b3fa00368753fad728f83c7a49d3dc1ebe6e17dd2543b1b3f`。这是preparer检查，**不是独立metadata执行准入**；两文件execution_authorized均为False。

新增 `src/s2a/clean_training_d2.py`拒绝future/command anchor字段、评估噪声、越界31位seed、训练覆盖或顺序变化、其他采集策略以及将proposal改成execution许可。新增与既有当前输入/采集合同组合测试68项通过，Ruff及格式检查通过。测试不等于实际D2模型或独立数值审查。

## 尚待完成

独立D2 worker、逐节点数值审查及持久化owner现已实现，但metadata执行登记、来源冻结与实际准入dry尚未完成，未启动模型。不能把新请求伪装为旧D1来绕过原两轮限制。后续仅在完整准入后运行一次300请求/10500节点D2；正常退出与审查合格后，才在原150训练窗口上固定聚合D0/D1/D2的1260000行拟合一个统一头，并独立完整QR。lambda固定0.1/1290，不用validation标签拟合，不按任务拼接，不自动新增D3。

本轮模型初始化、采集请求、fit/QR、物理步、fresh均为0；候选收益及总目标完成尚未证明。

## D2执行组件实现与CPU检查

- `src/s2a/clean_training_d2_execution.py`分离proposal与执行登记，要求完整metadata、固定角色源码、输入/快照SHA、163个checkpoint文件stat、原D0审查及owner绑定，并逐请求复核原始配对来源。只接受原052参数，不借用旧D1权限。
- `scripts/collect_wam_clean_d2.py`具有独立D2身份。模型采样/捕获主体由038实现派生；AST逐语句比较证明仅改变末尾准入函数和initial策略标签，保留调度、原始速度、残差注入、舍入、mask、配对、清理及错误传播逻辑。dry路径在模型导入及目录创建前返回。
- `scripts/verify_wam_clean_d2.py`复用既有独立NumPy数值primitive，不调用producer预测函数；核验全部10500节点、D0/D2初始条件exact、请求/网格/归一化及真实wait/reap/GPU释放。原始D0数组按原来源目录定位，不误指向新目录。视频/flat省略量仍仅有在线hash witness，不能声明完整离线重构。
- `scripts/run_wam_clean_d2.py`为独立持久化owner，持有实际模型子进程，UUID/空闲状态和端口31356检查后才执行；真实释放后运行CPU审查，失败保留收据且不自动重试，不自动fit或打开fresh。

三个实际CLI的`--help`均在空CUDA/LD环境正常解析；owner生成的命令已通过真实worker的argparse联调。新组件、原采集与独立数值primitive的119项检查在增加owner联调测试前全部通过。最终加入持久化supervisor的156项组合检查首轮155通过、1项旧bundle入口`--help`超过其3秒限时；该单项原样独立复跑通过（6.75秒为整项测试耗时），未修改旧源码或放宽限时，首轮失败如实保留。Ruff及5个新文件格式检查通过。这些是CPU软件证据，不是实际D2准入、模型执行或性能资格。

下一步实现独立metadata核验和执行准备，冻结所有实际角色及依赖，再用真实模型解释器运行已准入CLI dry。启动前仍须实际fsync写探针、GPU与端口检查；当前清单的execution_authorized=False保持不变。

## 完整执行准入及D2启动

新增 `verify_wam_clean_d2_metadata.py` 从原D0实际请求、种子计划、原current图像和initial引用独立复核300行清单；不调用producer的清单校验函数，也不解码未来数组。完整捕获主体AST核验纳入实际metadata。新增 `prepare_wam_clean_d2_execution.py` 冻结所有角色并登记独立metadata后，使用真实Cosmos模型解释器执行实际worker `--dry`。157项组合测试完整通过，首轮旧bundle超时记录不删除；新增metadata核验会拒绝把实际35步scope改成34步的源码。

实际准备冻结1470份来源及1470份快照，所有源/输入SHA及163个checkpoint文件stat核查通过。另将model_config、prerequisite_refs、weight_files与合格055 prepared逐项比较，完全相同，记录在 `model_identity_crosscheck.json`。

|证据|SHA256|
|---|---|
|prepared.json|806e2bebbbf5185a0c6c224e1f98b0497e096b9e2b95b2cef3068293935e474a|
|metadata_verified.json|c8e370cf8ac1aafc00aaa806f338eae833f9feb97dc0c02a8251bdf25fb29c66|
|registration.json|797886063ffb05c3a45bb3d793c52b2e214952aedeb991c95d95e9bb5ee22bbc|
|successful_cli_dry.json|4d5775dff44eca975294bb025bea2d38f412bb46f7ec1539a642697702fbf943|
|supervisor/launcher_result.json|d0ab7bce5e40a8e0adfb58f538714ab69451344a383d0ae5ed6eaf9076bf387b|

实际小文件write/flush/fsync/unlink通过，文件系统约155T可用；并不据此声称此前ENOSPC原因已查明。允许UUID对应当前visible2/minor6，GPU memory0/apps空/util0，端口31356可绑定。owner自身在实际模型启动前再次做写探针、容量、GPU和端口检查。

短launcher正常返回；持久化owner PID2866962/birth579368025已由真实ps确认PPID1。实际模型worker PID2867063/birth579368604由owner持有，launch记录明确使用允许UUID及空LD。初次进程核查两者均存活，worker仍在启动前置阶段，尚无完整请求或模型初始化成功收据。这里只证明实际进程启动，不声称已完成采集或通过数值审查。

固定300请求/10500节点和一次模型初始化预算不变，无自动重试。应继续观察同一owner/worker，完整采集、退出/GPU释放与独立D2审查全部合格后才开放聚合fit。当前无新fit、物理或fresh；原proposal仍为execution_authorized=False，实际许可来自独立registration，两种身份不混用。

## D2采集中：聚合拟合入口准备

同一owner PID2866962、worker PID2867063已再次经真实ps确认存活，配对采集日志已实际闭合19/300请求、665个denoise节点；并非只有初始化或预期进度。1470份源与1470份快照全部重查无漂移，未重启或新增模型实例。这是中间快照，不代表全部采集或独立审查通过。

新增 `src/s2a/clean_training_aggregation.py` 只读JSON准入：要求完整D2 owner、模型与审查器各自的真实wait0/reap/no-signal及PID/birth绑定、实际GPU释放、10500节点数值审查和300初态配对、原052策略身份以及既有D0/D1资格。必须全部满足后才允许调用方解码训练未来标签；live/partial D2不能形成fit尝试。

新增 `scripts/fit_wam_clean_aggregation.py` 固定按D0/D1/D2、900请求、1260000行等权训练一个头，ridge=0.1/1290。双遍重新计算聚合训练中心/全局scale并核对全流SHA，同时独立保留旧D0/D1 feature与target前缀SHA比较。权重改为1/1260000，因此不能要求其权重前缀与旧1/840000相同。目标始终减raw C0，不减部署残差。

新增 `scripts/verify_wam_clean_aggregation.py` 从实际FP32 sample/raw velocity及sigma独立重建C0，独立FP32 multiply/add/clipping归一化原训练标签，以完整augmented QR重构全部1260000行，按原rtol1e-9/atol1e-11比较中心、scale与参数，并比对全量feature/target/weight流。此入口不调用producer的行构造或fit函数，也不使用validation/fresh数据。

聚合生命周期/标签隔离与既有streaming、独立QR primitive组合71项测试通过，包括实际producer在live owner下先拒绝、任何np.load或fit均不得发生，以及wait/reap/signal/PID/birth和跨轮身份错误拒绝。新增入口真实`--help`、Ruff与格式检查通过；首次单独测试的scripts导入路径问题已在新测试中显式设置并重跑通过。尚未生成实际fit/audit registration、未尝试fit/QR，测试不构成真实聚合数学资格。下一步继续原D2进程观察，并准备固定fit/QR的注册与持久化CPU owner；只有完整D2资格产生后才绑定并执行。

## 聚合注册/CPU owner及实际live门检查

新增 `scripts/prepare_wam_clean_aggregation.py`：仅在完整D2 owner及审查合格后，才创建 `aggregate_fit/`、冻结fit/QR/owner来源与快照、绑定原W2资格及D2实际收据，生成唯一fit registration，并执行实际producer `--dry`。该dry只读metadata，不产生fit尝试或解码标签。

新增 `scripts/run_wam_clean_aggregation.py`：持久化CPU owner实际持有producer和独立QR各自的Popen，生产正常wait/reap并取得完整1260000行结果后才生成独立audit registration。全部QR资格正常完成前，不授予新候选数学资格；没有模型、物理、validation或fresh自动调用。失败保留实际launch/exit/release收据，不自动重试。

在实际仍存活的D2 owner/worker上调用真实注册函数，并将该CPU进程的np.load替换为禁止调用：入口因尚无最终D2 owner收据按预期拒绝，未创建aggregate_fit目录、标签或fit尝试。证据 `aggregation_gate_live_probe.json` 保留真实PID/birth及实际缺失收据路径；这不是D2失败或进程停止。

新增真实CPU小子进程exit7的owner测试，实际wait/reap、无signal清理、失败收据保存通过，且QR没有启动。聚合、streaming、独立QR及durable supervisor组合108项通过，Ruff通过。此处的CPU测试子进程不计为真实训练fit。当前固定D2同一进程已实际闭合40/300请求、1400节点，两PID再次确认存活，完整独立审查尚待采集退出后执行。

另已实际对首个完整D2请求做label-free数值抽查：35个保存节点的原始C0、052残差、速度注入、舍入及源链由既有独立NumPy primitive重算通过，首个D0/D2 initial arrays exact配对通过，消费SHA前后不变。产物 `first_request_numeric_probe.json` 明确eligible_for_fit=False；它不是全300请求数值审查或正常退出资格，也不打开训练标签。

## 聚合候选接入和固定validation评分组件

同一D2 owner/worker继续经ps确认存活，最新采集快照为62/300请求、2170节点。未重启模型，仍无完整D2收尾或聚合参数。

新增 `src/s2a/clean_aggregation_candidate.py` 要求新聚合producer、独立完整QR和CPU owner同时正常完成，核验真实wait/reap、1260000行、相同三轮数据与D2 owner、参数身份、全量feature/target/weight流和原容差，再返回只读head。不允许052 schema、旧840000行、其他参数或未reap过程冒充新候选；数学资格不会授予validation/fresh运行许可。

新增 `src/s2a/aggregate_state_sampling.py` 保持原current-only schema和3zero＋48raw＋48新头的99请求矩阵，新variant明确为 `aggregate056`。新增 `src/s2a/aggregate_state_metrics.py` 用新候选身份计算原054全部指标与门槛；完整pool/state_gate/diagnosis函数AST仅替换候选名称，阈值、分组与数学运算不变，旧冻结源码不改。

22项候选身份、全量QR及评分等价性测试通过，Ruff/格式检查通过。另在实际053的24个当前锚点和原资格上构造99请求矩阵，逐对核查图像、任务、seed和原始请求内容（除name）一致；将旧055已失败的评分JSON仅换variant名重放，所有pooled/task/episode/seed指标和gate完全相同，仍为False。产物 `aggregate_state_components_probe.json` 明确new_candidate_performance_available=False；这是软件回归证据，不是对新头的新采样或性能评价。

新聚合head尚未产生，实际validation采样worker/准入仍待接入和冻结，未运行validation或fresh。当前推进不能替代原留出新初态上的机器人跟踪和稳定任务验收。

## 聚合validation采样/评分入口实现

新增 `src/s2a/aggregate_state_execution.py` 和 `scripts/sample_wam_aggregate_state.py`，独立输出根固定为 `aggregate_validation/`，明确使用新的aggregate owner/参数身份。准入要求完整新头资格、冻结角色/来源/快照/输入、163个checkpoint文件stat及独立metadata，99请求矩阵必须从原053 current-only清单重新构造并逐项相等；当前尚未授予这种执行登记。

采样actor沿用原完整99请求的已验证采样/捕获逻辑，`raw/zero/aggregate056`显式分派。实际worker表达式在全部99个变体上均正确选择readout，未知或旧probe052变体被拒绝；AST比较证明完整捕获主体仅改变候选名称与readout选择表达式，原调度、mask、残差注入、舍入、zero回归、初始条件配对及清理未变。实际CLI的dry路径不会导入torch/Cosmos或解码数组。

新增 `scripts/score_wam_aggregate_state.py`，仅接受新aggregate数值/生命周期完整合格收据，才可读取原053六个离线未来archive；保留原integrate/SO(3)与门槛计算，并在结果中绑定新candidate owner/params。已用实际合格055收据测试拒绝路径，在禁止np.load条件下按预期拒绝，未创建新score。证据为 `aggregate_old_receipt_rejection_probe.json`，不把旧055资格或结果改写成新候选证据。

新actor、候选资格、评分等价性及原current-only采样组合37项通过，Ruff/格式检查及实际worker/scorer CLI help通过。首次组合检查中旧测试缺scripts导入路径的两项失败，使用显式PYTHONPATH=src:scripts原样重跑通过；未修改旧冻结源码。当前D2两实际PID继续存活，未重启或额外初始化模型。完整聚合validation数值审查、metadata登记、owner及执行仍待完成，不能启动新模型；无新聚合参数或性能、fit、物理、fresh。

## 聚合validation全量数值审查和owner实现

新增 `scripts/verify_wam_aggregate_state.py`，要求新aggregate采样schema、候选owner/参数和完整99/3465预算、105个zero节点及48组初始条件配对，逐节点重算原始C0、残差与部署舍入并核验zero原始响应精确回归。完整 `verify_generations` AST与原独立审查仅候选名称不同，数学和源链检查保留；未调用producer预测函数。调用时再次通过新候选资格并从原current-only来源重算完整矩阵，拒绝旧候选命令或其他模型脚本的退出收据。

生命周期独立绑定实际new worker解释器/路径、prepared/registration/candidate-owner SHA参数、允许UUID和空LD、真实wait0/reap/no-signal、PID/birth及GPU释放，任何收据逃出本次输出根均拒绝。完整通过后才发布评分器所需的新 `s2a.aggregate-state-audit.v1`，不读取未来标签。省略video/flat值仍保留既有在线hash witness边界，不声称完整离线重构。

新增 `scripts/run_wam_aggregate_state.py` 持久化owner，使用独立端口31357和相同UUID锁；实际写/fsync、容量、GPU身份/空闲和端口检查后才启动模型。模型实际正常退出并释放GPU后运行独立全量审查，再运行离线评分。失败保留真实收据，不自动重试，不调用开发或fresh。新增owner命令已通过真实worker argparse/dry分支的CPU联调。

新numerical/lifecycle、采样、候选接入、原数值primitive及durable supervisor共130项组合测试通过，Ruff、格式及两个实际CLI help通过。D2仍为同一owner PID2866962和worker PID2867063，最新真实日志闭合97/300请求、3395节点，ps确认两者存活。当前无聚合参数、fit/QR或新validation模型；下一步补齐独立metadata与执行准备，但要待原D2、聚合fit及独立QR合格后才能冻结候选、实际登记新validation执行。原055失败、原最终门槛和fresh关闭不变。

## 独立metadata和执行准备入口

新增 `scripts/verify_wam_aggregate_state_metadata.py`：独立使用原冻结矩阵实现重建99请求，仅替换候选身份并逐项比较；将模型配置、163个权重文件stat、当前输入、zero案例、模板、噪声种子及原资格逐项绑定055基线。只接受JSON/当前PNG输入映射，新候选完整fit/QR/owner及参数身份必须通过。实际worker分派表达式在完整99行上执行marker核验，105个历史zero节点必须可用；完整run、数值审查函数与评分/门槛函数AST逐项证明，没有改变原数学或门槛。

新增 `scripts/prepare_wam_aggregate_state.py` 和显式配置 `configs/wam_aggregate_state_20261024.json`。只有完整聚合头资格产生后才创建validation根和冻结角色/依赖/输入；独立metadata合格后单独登记，再以实际模型解释器执行精确CLI dry。原proposal/config不因此变成执行许可。模型仍仅允许已核实UUID，固定端口31357及99/3465预算，不自动重试。

新增测试实际篡改采样步数、全量数值配对数和原评分阈值，完整AST证明均拒绝；真实准备函数在缺聚合head时先拒绝且不创建目录。相关30项组合测试、Ruff/格式及两个实际CLI help通过。另在真实live D2上调用准备函数、禁止该CPU进程np.load，因聚合owner尚未产生按预期拒绝，0validation目录/数组/模型。产物 `aggregate_validation_preparation_gate_probe.json` 明确full_metadata_executed=False；完整metadata执行资格仍须等待实际参数，不能以body proof或单测代替。

同一D2进程继续存活，最新核实123/300请求、4305节点，原模型不重启。已确认README、报告索引及当前状态文档不在1470冻结来源中，修正其仍显示055准备中的过时首页信息，保留历史失败与阶段快照。当前无新fit/QR、模型性能、物理或fresh，总目标未完成。

## D2完整采集与正常模型退出已完成

原worker PID2867063/birth579368604实际完成300/300请求、10500 denoise/head节点，`capture/D2/result.json` complete/passed=True。实际退出收据为returncode0、actual_wait=True、automatic_retry=False；release绑定同一PID/birth，reaped=True、signal_sent=False、original_error=None。OS进程已消失，与真实wait/reap一致，未重启模型。

`orchestration/gpu_after.json` released=True，允许UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f` 的memory_mib=0、compute_apps为空。原持久化owner PID2866962继续存活，已实际启动独立CPU审查PID2953038/birth579870815，PPID为原owner；最新进程核查审查仍在运行，尚无完整数值收据。

当前只成立完整采集和正常退出，不等于10500节点独立数学资格或任务收益。等待同一审查进程完成，禁止将观察超时视为失败并重试；仍无aggregate_fit目录、fit/QR、物理或fresh。

## D2完整独立资格通过，聚合CPU fit已启动

独立审查实际完成300请求/10500节点、300组原D0初始配对，`d2_verified.json` complete/passed=True，SHA256 `05af16c4bc16c097e4f017f33a1ad1bf76782f6b9201a3833d2f23b3c65c442a`。审查实际wait0/reap/no-signal通过，审查PID2953038已消失；原D2 owner也正常完成并消失，`orchestration/result.json` SHA256 `f27ac87d0d4fe29e9ce13a3dcd8c060800133bf9fdc9f7c0b1e42bdd1d935f1a`。采集、数值审查和两阶段真实生命周期联合资格成立，但不是任务效果证明。

固定1260000行聚合fit注册已在完整D2资格后生成，30份CPU来源/快照冻结，真实producer CLI dry正常exit0，0fit/标签解码。注册SHA256 `3331dee32fe0c8b663a42b134c64a75f7110414ff16320b1780a59ee3e162e32`，successful_cli_dry SHA256 `2c14a93fab55b32d1ef124fbb1a4d144605e16534cf25703ae45bd75758d3bf1`。输出独立位于 `aggregate_fit/`，旧D0/D1/D2和052/055不改写。

实际持久化CPU owner PID2956273/birth579893341已确认PPID1，producer PID2956281/birth579893414由其持有。唯一fit attempt1已记录，空CUDA/LD、线程8；第一遍实际进度400/900请求、560000行，两进程存活。固定三轮等权、ridge=0.1/1290及原容差不变。完整双遍/solve、原D0/D1前缀重放SHA及独立完整QR尚待验证，不能称新头合格。producer正常完成后owner才登记并启动一次QR；无新GPU模型、物理、validation或fresh。

## 聚合producer完整完成，独立QR运行中

同一producer实际完成双遍各900请求/1260000行，原D0/D1的feature/target前缀指纹及完整双遍重放检查通过，全部消费输入前后SHA不变。`aggregate_fit/result.json` complete/passed=True、producer_fits=1，SHA256 `6d88641948d9a336c05ea149ec380933bd4b9f0d23d1368980c8f1e07c038cf4`；参数SHA256 `ac96dac5cf2d545a6d29a5b7d6bc2ac91406a2b2c291e50ba7f7489a17fceaf1`。producer的independent_numerical_qualification=False如实保留，不能提前宣称新头合格。

producer正常exit0且真实reap/no-signal，PID2956281已消失；同一owner实际生成唯一audit registration（SHA256 `06ef1409851ca1085d0129f9b4aa4adde9ab3086c69318f649a7e9a525686dc3`），启动独立CPU QR PID2960122/birth579919256、PPID2956273。最新真实进度QR第一遍900请求完整，第二遍50/900请求；审查及owner均经实际PID/birth确认存活。原rtol1e-9/atol1e-11不变，完整QR参数比较及正常退出收尾尚待完成。当前无新validation模型、任务效果、物理或fresh。

## 聚合完整QR资格通过，固定validation已启动

独立QR实际完成双遍全部900请求/1260000行、150个训练标签窗口及raw C0重构，三条全流SHA与producer一致，消费输入前后不变。按原rtol1e-9/atol1e-11比较通过，coefficients/center/scale最大绝对差分别为5.653255641391297e-13/0.0/1.7763568394002505e-15。`aggregate_fit/independent_verified.json` SHA256 `d4d795690f52646bbe1330352c844398c09260e76bb1b3abb7300f6537da47cf`，CPU owner完整成功收据SHA256 `50664366bd9be6976421eb5959de3aed99b83b5a0dad4c866796d5fe00788a93`。QR实际exit0/reap/no-signal，QR及owner PID均已消失；producer的独立资格False保留，由两份完整收据联合认证数学资格。

新统一参数SHA256固定为 `ac96dac5cf2d545a6d29a5b7d6bc2ac91406a2b2c291e50ba7f7489a17fceaf1`，不是旧052头。数学资格不等于泛化或任务收益。随后在独立 `aggregate_validation/` 根完成真实准备、完整独立metadata核验及真实模型解释器CLI dry，1476份来源/快照冻结。

|validation证据|SHA256|
|---|---|
|prepared.json|df404a16102782280407b70fb606243c5aeb7071e205f6f2f156dc0931c7e015|
|metadata_verified.json|103f3d73087170dac9ec2699b6be3220ee5ad4502cbde72f9fcb6f2ed11aa203|
|registration.json|e46be22daa8133c1cc93ab3af2ccb7d7fa73f55be8b348fc8224d596042ac780|
|successful_cli_dry.json|ab416c6126d1c07c58be01601dd79e8f6bf9ec51c7abefc1c84d13a271399140|
|supervisor/launcher_result.json|944f7974adafd7bb5970dc11d14eb8c23b8d91d52315d250a6f859a4a7ae8f27|

实际启动前允许UUID memory0/apps空/util0，owner再次执行fsync写探针、容量/GPU身份/端口31357检查。短launcher正常返回；实际持久化owner PID2972453/birth579999208已确认PPID1，worker PID2972468/birth579999367由其持有。实际launch环境仅允许UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`、LD为空；两进程存活，模型正在加载。固定3zero＋48raw＋48aggregate056共99请求/3465节点，仅一次模型初始化，无自动重试。

尚无完整采样、独立数值/生命周期资格或新候选state gate；模型进程不读取validation未来标签。完整正常收尾后owner才进行独立审查，再离线评分。未启动开发/物理/fresh，原055失败、原24条验收门槛与总目标未完成状态不变。

## 完整validation终态：数值合格，性能门槛仍失败

全部99请求/3465节点完成，独立审查核验105个zero节点、48组完整初始条件配对、保存动作数学及来源，数值/生命周期均通过。模型、独立审查和评分三个实际子进程均exit0、wait/reap/no-signal，GPU释放；原owner正常完成。脚本complete/passed=True表示执行和核验成功，**state_gate_passed=False、eligible_for_development=False**仍然保留。没有启动开发或fresh。

|证据|SHA256|
|---|---|
|state_sampling/result.json|5dcadf062a28c8e8b158205a4765d6e924c0bf3eea83ab01b97c7c38add6d094|
|numerical_verified.json|927d8554a0f5dfe34a9064f6f317151b1b9fc6bfaf45db6f11df8150f2e9dcb2|
|state_score.json|25e377bc6e6892cddd33bd805941c95eedc83b66c1b0c12b7438d14949b7ebfa|
|orchestration/result.json|e3b08fedba0f0a7af5f886b2d770b4968575e30f3f1f274b9b64aa400d96b319|

实际同步参考误差如下，开度为物理0–1状态，rotation为SO(3)角误差；这些不是机器人跟踪或任务成功率。

|H|raw位置RMSE(m)|aggregate位置|raw旋转RMSE(rad)|aggregate旋转|raw开度MAE|aggregate开度|raw joint|aggregate joint|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8|0.022961|0.025142|0.098577|0.075565|0.063151|0.107001|10.094769|10.312038|
|16|0.037509|0.043606|0.179833|0.133437|0.046795|0.111865|27.581307|27.928950|
|40|0.067346|0.071820|0.457855|0.302036|0.097898|0.142701|133.080340|91.596230|

H16相对raw：位置+16.255%、旋转−25.799%、开度+139.056%、joint+1.260%。所有时域的旋转分量保护通过，位置及开度保护失败；短期joint改善、≥5个episode、无短期outlier及两seed保护均失败。仅“优于hold/constant_velocity”通过。3/6个episode满足短期joint不退化检查，不是3/6任务成功或稳定。

与055/052头比较，48组raw实际full128 action及normalized_action逐值完全相同，受控比较记录在 `comparison_to052.json`，消费SHA前后不变。聚合头H8/H16/H40的joint相对052分别降低12.276%/14.203%/18.993%；位置、旋转、开度也均改善，但未恢复相对raw的原保护门槛。D2改善了候选，却未完成注册假设的性能目标，不能据此继续未登记D3或把新头送入fresh。

新增 `scripts/analyze_wam_aggregate_validation_teacher.py` 对已合格节点离线诊断：96请求/3360节点、每variant67200行，六个053未来archive/24个H40窗口只用于诊断，不回流拟合。新头在raw节点的全步teacher xyz/rot6d/opening MSE降低17.724%/67.209%/12.660%，自身节点降低15.341%/66.422%/11.179%；但自身末节点xyz/opening仍反向增加7.243%/8.761%。相比052自身末节点增加23.60%/14.51%有所缓解，仍未消失。新头首节点xyz也仍增大2.207%。预测与实际保存残差比较通过，消费SHA前后不变，Ruff/格式检查通过；无新模型/fit/物理/fresh。

下一步应区分局部归一化clean MSE收益与积分后物理state误差，检查有符号增量偏差、旋转/平移耦合及开度阶段偏差，再提出新的单一、因果候选。不能将平均teacher改善等同于参考轨迹或任务成功，也不再将分布覆盖视为唯一解释。完整056性能失败、原055失败和原最终验收门槛均保留；总目标继续未完成。

随后已完成[物理积分误差分解](wam_reference_error_decomposition_20261025.md)：全部96请求的逐点位置/开度与原评分复现。朝向耦合项下降，位置退化来自更大的局部平移累积项；H40单步平移MSE虽降低，累积MSE却升高。H16闭合/打开参考的预测分别向0.205/0.892偏移，均比raw更远离真实0.052/0.977。三项实际几何测试通过，无新fit/模型/物理/fresh；下一步检查时间交叉项与训练中的开度损失差异，不据GT分组在线切换。
