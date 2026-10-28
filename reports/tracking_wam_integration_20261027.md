# Tracking候选WAM接入：实现与准入检查

当前目标仍是当前图像WAM轨迹驱动实际控制并在留出新初态完成任务，正确参考18/18本身不是该目标完成。

新增独立 `src/s2a/tracking_preview_wam_simulation.py`，保留原当前图像/HTTP/固定12条开发case、600步/75请求、无expert输入、固定夹爪、服务身份、空CUDA/LD检查；原执行器源码未修改。原solve_preview作为因果seed，经TrackingPreviewTransaction.prepare选择实际arm命令，并在实际派发前核对commit、真实周期结束后observe。原生控制目标逐周期核对；日志将原seed预测与实际selected command分开记录，不能把seed forecast误作最终命令预测准确度。

新入口要求独立 `s2a.tracking-preview-wam-development-registration.v1` 注册，绑定tracking_admission及新入口/gate/transaction源码，旧WAM注册不满足新入口要求。collector、模型service、执行注册及独立新日志检查器尚未适配，因此当前不能直接使用旧启动器运行或声称接入已通过物理全链验证。

新增 `src/s2a/tracking_control_gate.py` 验证完整18条原门槛、18条输入与图像审查的同一actual candidate绑定、独立回放相机、汇总stage实际wait0/reap、实际触发优化的提交组件证据及所有冻结来源SHA一致。使用当前真实产物执行准入检查通过，进程实际退出0，权威产物 `outputs/tracking_wam_admission_20261027/verified.json`，SHA256 `7a7535a599f6592da876e2f54499bd7946229c19a657bab231918d9af165ca5e`。这只验证前置证据读取与准入，不是WAM执行器物理验证。

两新模块ruff与格式检查通过；原提交组件3项CPU协议测试再次通过。新增物理周期0、WAM请求0、fresh0。下一步适配新collector/协议注册与独立检查，执行实际CLI dry和物理接入验证，再启动原固定12条WAM开发；仍保持原开发与最终24条新初态门槛，总目标未完成。

## 独立采集、服务与执行注册适配

新增 `tracking_preview_wam_execution.py` 与 `tracking_preview_wam_control.py`，独立准备/注册schema及root `outputs/tracking_preview_wam_development_20261027`，HTTP端口31375、分布式端口31376。保持原固定12条、每条600周期/75请求、合计7200物理周期/900请求、单模型初始化、raw/terminal058配对及最终fresh未授权约束，并额外要求完整tracking_admission绑定。保留允许GPU UUID的既有显式检查；没有启动任何模型。

新增三个入口：`collect_tracking_preview_wam_development.py`、`serve_tracking_preview_wam_development.py`、`run_tracking_preview_wam_development.py`。collector调用run_tracking_wam_case而非原规划器执行器；私有服务和owner使用新的注册/角色/source绑定，旧已完成执行授权不能复用。HTTP当前图像输入、请求keys、固定seed/图像目录、300s超时、进程身份和释放协议保持。三个入口均在原物理Python环境、空CUDA/LD下实际执行--help并退出0；这只是CLI导入/参数检查，不是注册准入--dry或物理验证。

`tests/test_tracking_wam_transport.py` 检查新端口请求不改写内容、超时封顶、未来状态/错seed/其他图像/外部地址/无超时在派发前拒绝，并使用真实本地HTTP验证三个非法服务请求不触发模型。独立旧schema拒绝测试验证不能用旧注册启动新服务。新文件ruff/格式检查通过。

尚缺：新准备与独立metadata验证器、独立物理/请求/候选日志检查器（不能复用旧优化器精确重放断言），以及完整新注册下的实际CLI dry与物理接入验证。新execution角色中列出的 `verify_tracking_preview_wam_development.py` 和metadata验证器目前未实现；新owner尚不能进入可执行准入状态。新增物理/WAM/fresh均0，不把这些入口实现或帮助命令当作WAM性能证据。总目标仍未完成。

上述新入口测试最终实际执行10 passed（1.08s），包含旧注册拒绝用例；无GPU模型加载或物理step。后续接入验证仍按前述未完成项推进。

## 准备与元数据验证进展

已实现独立候选arm日志检查器：保留原seed规划的独立数值验证，同时单独检查实际tracking命令、完整计划边界/变化率、原生目标、机器人端点输入及limited历史递推；不要求最终命令等于seed，也不将seed的KKT证书视为最终候选的最优性证明。共享HTTP、当前图像身份、夹爪与物理任务/指标检查保留。新程序为 `tracking_preview_arm_log_replay.py`、`tracking_preview_wam_acceptance_checker.py`、`tracking_preview_wam_audit_admission.py`、`verify_tracking_preview_wam_development.py`；仅完成实现/静态检查，尚未在新WAM日志上完整执行，所记录limited模型预测本身不构成独立模型最优性证书。

`check_tracking_preview_wam_inputs.py` 在原Python3.11环境实际完成12条配置输入预检，passed=true、数组解码/物理/WAM均0、execution_authorized=false。`prepare_tracking_preview_wam_development.py --qa` 实际97 passed（18.03s）。准备首次在.venv因缺h5py于新ROOT创建前失败，未改共享环境；改用现有native Python后成功冻结 `outputs/tracking_preview_wam_development_20261027/prepared.json`，SHA256 `e16f5409ef5689ab366ed257e72b2cb63533e90955de24a010f3e413c5597ebb`。来源及安全副本保留在该ROOT/source。

独立metadata v1拒绝该准备产物：QA实际检查清单新增了tracking测试，但verifier仍使用机械改名后的空测试glob构造期望命令，因而报“actual complete QA command differs”。没有改动已冻结源码、没有发布metadata_verified.json、没有注册或启动模型。新增 `verify_tracking_preview_wam_metadata_v2.py` 明确按实际应执行的旧共享回归＋tracking专用测试重建清单，并输出独立metadata-review.v2与registration_compatible=false；这仅用于核查修正，不能被现有v1执行gate接受。下一步必须通过版本化准备/验证/注册绑定修正才能运行完整CLI dry，不能把review当作执行授权。

v2独立metadata review实际完成并退出0，12条profile/来源/前置state资格及97项QA命令清单核对通过。产物 `outputs/tracking_preview_wam_development_20261027/metadata_review_v2.json`，SHA256 `730a2abb2a2cc76d9e751f436fde07f4b727dd057ab96d77ddafe72008dd576a`；保持registration_compatible=false。全流程新增物理/WAM/fresh均0，完整目标未完成。

## 修订版注册、完整dry与固定开发执行启动

保留全部v1源码/准备产物及独立review，新增成套v2入口/控制/执行/检查器版本和root `outputs/tracking_preview_wam_development_v2_20261027`。修复QA清单后，原Python3.11的12条输入预检通过；QA实际107 passed（23.83s）；独立metadata revision2通过。准备SHA256 `993efc55018ba4fc4f50617acd0ed37a2b0cece4419b0a11f2f014180abecb35`，metadata SHA256 `1cfcd5bc5e8110c05f4534022b7425c382da1a79f708229b1492a78d1d675fb7`，注册SHA256 `77680fa73b6f25a4d994961176cc0b4fecca6cfcf34cd3d5833515fdd7c26c5d`，均为新ROOT下对应JSON。1638份来源与1638份安全快照冻结，未热改v1或正确参考来源。

完整注册下owner、私有服务和12个collector共14条实际CLI dry全部退出0，权威 `successful_cli_dry.json`；无模型初始化/物理推进。新注册fresh_authorized=false、automatic_retry=false。

在允许UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f` 空闲（0MiB、0%且无compute进程）时，使用已通过dry的同一owner参数启动独立supervisor；launcher实际wait返回0，启动产物 `supervisor/launcher_result.json` SHA256 `6ffb82bd0de6e083a67e86b3ebb28fe9db2f5564ce96f1ce24c9e066412252e8`。执行owner PID1936469的内核start_ticks与launch记录一致且为活进程；其直接模型child PID1936529存在，实际进程环境CUDA_VISIBLE_DEVICES为上述允许UUID、LD_LIBRARY_PATH为空。没有在禁止GPU启动任务。

本次固定12条WAM开发仍是一次模型初始化、900请求/31500节点、7200物理周期，三任务×seed42/43×raw/terminal058配对。当前仅确认owner与模型子进程启动/存活，尚未确认模型初始化完成、完整请求数或任务成功；不将启动计为通过。须继续检查实际服务身份、请求/物理结果、独立日志审查、正常退出与GPU释放，并按原开发门槛决定后续；最终24条fresh保持关闭，总目标未完成。

## v2实际启动失败与端口合同修正

启动后的原owner和模型child均已终止，不能继续报告运行中。v2模型初始化实际完成1次；首条collector在物理循环入口报“Registered237 UUID and HTTP port required”，实际请求/denoise/物理周期均0。owner/service result均False；collector与service均实际wait1/reap，gpu_after.released=true，允许UUID显存回到0MiB。所有失败产物与v2源码/快照保留，没有重启原任务，也没有把该失败算成任务性能失败或成功。

根因定位到 `tracking_preview_wam_simulation_v2.py:320` 仍硬编码旧HTTP端口31365，而新准备/服务/collector使用31375；此前dry在进入物理函数前返回，未覆盖该检查。第三版使用独立源码/目录/schema，将同一个 `validate_checkpoint_endpoint` 同时用于配置生成（因而覆盖dry）和实际物理入口/HTTP配置，端口取自注册执行模块PORT常量，不再在物理执行器复制旧数字。实际入口在读取tracking_admission或任何输出/HTTP/环境前就会拒绝错端点。

新增针对v3的实际物理入口旧端口/禁止UUID/重复checkpoint/缺失checkpoint拒绝测试，以及注册正确端点和当前请求/真实HTTP拒绝测试；合计15 passed（10.82s），静态/格式检查通过。新12条输入预检和完整QA正在执行，尚未注册或重启模型。后续若启动v3，将是新登记的独立尝试；累计模型初始化需另计v2这次1次，不能伪称整个过程只初始化一次。fresh仍关闭，总目标未完成。

## v3完整准入与独立新尝试启动

v3实际12条输入预检通过，完整QA122 passed（17.33s），独立metadata revision3通过，14条实际CLI dry（owner/service/12collector）全部退出0。新准备SHA256 `8f86f4186b5c56354828b07c95e8d77683eec80fe512803067643b74c0947e71`，metadata SHA256 `85e9270e8d1654fbe2bfef49dbe79a276645225f9d5babd5fbc9ad6f847a4065`，注册SHA256 `9ad5385ca793f306f9e77cddc395c37fa42fb32aed68ae5d033794fd5dd3524a`，均位于 `outputs/tracking_preview_wam_development_v3_20261027/`。各旧版本不覆盖；fresh/自动重试均false。

确认允许UUID显存0MiB且利用率0后，用该新注册启动独立固定12条开发尝试；launcher实际wait0，启动凭据 `supervisor/launcher_result.json` SHA256 `da02caf5c31344604998ff5cc68dab39791d49576007cd27f99c2f7893e72ddd`。新owner PID1940388内核birth与记录一致、实际存活；直接服务child PID1940405存在，ppid=1940388，进程实际CUDA_VISIBLE_DEVICES为允许UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`、LD_LIBRARY_PATH为空。

当前服务子进程处于初始化阶段，尚无新的请求/物理/性能结果，不能计为已通过。v3仍限一次初始化、原12条/900请求/31500节点/7200物理周期；累计过程另保留v2已完成的1次初始化、0请求/物理失败。后续按同一新owner观察服务就绪、实际请求与物理进度，不因连接未就绪重启。最终24条fresh保持关闭，总目标未完成。

## v3服务就绪、真实WAM控制推进与独立请求核验

原owner PID1940388、模型child1940405和首collector1940796的kernel birth持续匹配且活跃；服务实际ready=true、model_initializations=1。首raw/task0/seed42案例已通过修正的物理入口，当前图像请求、生成参考和真实物理端点均已产生，不再停留于注册/dry阶段。

已对已完成的前4及前12个请求做只读独立核验：原request/response文件SHA、当前PNG像素SHA、任务/seed/当前case/请求索引、checkpoint/readout身份、raw与normalized action及mask/单位/时间戳、参考chunk与物理时钟保持字段均通过。结果为 `outputs/tracking_wam_query_review_20261027/case00_first04/verified.json`（SHA `f38362d41b713c025786911f411f1d468ac557a6de08c05b5f3b14bb170f7fa2`）与 `case00_first12/verified.json`（SHA `ce7a49d6b2b780e6c6de3752ef66e584e1cc2f7e180e393a6f3521b6a6527963`），各自保存独立prefix snapshot而不将仍增长的整份chunks文件假作冻结输入。检查进程均实际退出0，新增模型请求/物理周期0。此为已保存请求/图像/响应核验，不是完整native状态/任务或整个900请求资格。

在一次完整可解析的120行控制日志前缀观察到70次unchanged_free_feasible、49次interior_witness和1次adaptive_margin_witness；说明真实当前图像WAM反馈下已触发优化分支，不是仅正确参考bypass。但该前缀没有独立完整任务成功结论，不代表600周期稳定或12条门槛通过。

并发现冻结v3独立acceptance checker的verify_query仍有旧端口31365断言，预计会拒绝31375的新数据。运行中来源不热改，实际控制继续。新增只读 `tracking_preview_wam_acceptance_checker_review.py` 仅改端口到注册31375及模块说明，其余AST逐项相同，证明为 `outputs/tracking_wam_query_review_20261027/checker_correction.json`；全部其他检查保持。对应完整审查入口 `verify_tracking_preview_wam_development_review.py` 使用独立review schema，尚未执行整轮审查，不伪称原冻结verifier已修复或通过。原verifier的未来实际结果应保留；必要时以另行登记的只读复核处理已完成数据，不重采集或改变控制/性能门槛。

当前仍首条案例执行中，尚无完整600周期/12条任务结果；fresh继续关闭，总目标未完成。

## 单案例审核入口与独立复核登记

继续核实原PID/birth，首条raw/task0/seed42轨迹仍运行；最新观测有345条端点receipt（包含初态，约344物理周期），worker未关闭，没有600周期稳定成功结论。未因观察间隔或等待生成而重启。

新增 `audit_tracking_wam_completed_case.py`：仅接受实际600周期/75请求且正常wait0/reap的已关闭worker，执行独立seed/实际命令机械日志、75次请求、机器人输入、标签/图像与原生600/tail40谓词审查；模型全900请求及31500点数值资格仍须另行完整审查。当前只完成静态与实际CLI导入检查，尚未对未关闭的首条轨迹执行该审查。

发现采集器没有发出旧verifier要求的 `execution_snapshot_ref`，不能假装该产物存在。新增只读 `tracking_wam_review_source_binding.py`，核对真实owned launch中prepared/registration/config的路径与SHA、事前冻结源码及运行manifest中的实际签名；若生产者确有独立snapshot仍强制核对它。当前真实launch/config/manifest的绑定检查通过，实际退出0，记录 `outputs/tracking_wam_query_review_20261027/live_source_binding.json`（SHA `69c5ec37860f055881918d01f1b2771e7a20e645371d624f4f970ba262f2012b`）。这是实时已发出配置/源码绑定，不证明尚未完成worker的正常退出或任务成功；完整只读复核将明确标注独立snapshot是否存在，保留原verifier缺项/失败结果。

在首条案例关闭前，已单独冻结只读复核登记 `outputs/tracking_wam_independent_review_20261027/registration.json`（SHA `0a11f18e7424c1318db160f2af61fb77a60cfa8c182095a171b1647509468326`），包含原生产注册、复核来源SHA及端口改动AST证明。保持原terminal058稳定≥5/6、各任务≥1/2、stable_last40开发门槛，原case顺序不变；物理/新模型/新请求/fresh预算均0，不授予模型或fresh执行权。原冻结v3检查器、运行输入与物理结果均不修改；全12条完成前不宣布开发资格。总目标未完成。

## v3前三条已独立核验失败、第四条数值错误终止

此前等待中断后，原v3继续完成raw/task0/seed42、terminal058/task0/seed42、raw/task0/seed43三条，各600物理周期/75请求。三条独立case审核现均完整通过并actual_wait0/reap，任务原生日志600周期successful_steps=0、ever/terminal/stable均false；审查结果 `outputs/tracking_wam_independent_review_20261027/case00..02_execution/result.json` 和 `outputs/tracking_wam_case_review_20261027/case00..02/verified.json`。这是日志/机械/输入/标签/图像/请求核验，不是整个900请求和31500节点资格。

生产日志描述：raw42位置/SO(3)RMSE 0.00839736005226656m/0.035552005938084386rad、目标抓取0周期、q7未越XML边界；terminal42为0.005433148005090785m/0.020399859473554452rad、目标抓取0周期、q7未越边界。raw43为0.0124833135398095m/0.05502045744605921rad、目标抓取0周期，600个周期前状态中258次q7越界，最小margin -0.0031933104097650045rad。该观测不支持“跟踪误差小就任务成功”，也不证明关节问题已普遍解决。描述产物 `outputs/tracking_wam_interactions_20261027/first_three.json`。

生成参考时序描述显示：raw42和raw43均在15个query预测过opening<.35，但均位于被重查询丢弃的后32个节点；实际前8节点opening最低分别0.4055059254169464与0.9004740715026855，而全40节点最低0.005407243967056274与0.03672311455011368。terminal42则有8个延期query、40个含闭合query且执行过0开度，仍无抓取。阈值只作描述，不是任务判据；不能唯一归因闭合延期，也不能据此改变在线时钟。产物 `outputs/tracking_wam_interactions_20261027/reference_phase_first_three.json` SHA `c9184b435d970add9561bcd0b2f20818237fd5c8e0bda6819f0f98a3dcc60886`。另核对task0/seed42与原合格正确参考case15的初始sim-state SHA相同，作为已存在正控对照，不将不同输入轨迹的差异当作单因素因果干预。

第四条terminal/task0/seed43执行210周期后，在seed QP抛出“Dependent QP blocking constraint; original state retained”。v3固定矩阵已终止，不能再报告运行中。实际252请求/8820去噪节点、一次模型初始化，物理为三条1800＋第四条210=2010周期；后续八条未执行，第四条不能当完整600步失败样本。collector与service均actual_wait1/reap，owner消失且结果false，gpu_after.released=true，允许GPU显存0MiB；没有自动重试。所有原产物/source保留，fresh关闭。

## 精确捕获QP与守护数值修正（未部署）

CPU-only失败前缀回放已精确复现211个端点SHA（210周期），第210周期原seed第一次QP调用重现同一错误。首次捕获probe在MuJoCo早于OSMesa加载时初始化失败、物理0，记录保留；v2调整加载顺序，在原CPU线程/渲染环境完成捕获并actual_wait0/reap。追加210重放物理周期，WAM/fresh0，不混入原2010周期。证据 `outputs/tracking_wam_qp_failure_capture_v2_20261027/result.json`，SHA `f2902056a2b766737af62409367ec5a8569aba17e74b2f3e76460763d0265988`，包含failing_inputs.npz与failing_qp.npz指纹。

同一H/g/A/b/initial由现有null-space QP求解，在原不变门槛下独立检验：primal违约5.55e-17、stationarity1.78e-15、complementarity1.25e-18，较原可行initial目标下降0.6824883464981373；原solver仍可复现错误。该结果 `outputs/tracking_wam_captured_qp_check_20261027/verified.json` SHA `d00b6b76419a029db4c1c4a9f933f84f02d36d6a9be1c3e102c720ef5988eccc`。

新增可复用 `src/s2a/dependency_safe_qp.py` 只对精确匹配的dependent-blocker错误启用null-space解；原solver成功时直接原样返回，其他错误不重试，原约束/目标/容差不变。针对原成功结果逐值保持、真实dependent fixture、输入不修改及非指定错误不退化等合计8项回归通过（2.60s），ruff/格式检查通过。

完整seed规划在捕获状态使用此入口后也通过原独立四步preview检查：stationarity1.11e-11、complementarity0、预测差2.41e-13，实际触发一次守护分支。结果 `outputs/tracking_wam_recovered_seed_20261027/verified.json` SHA `8dc8e0d8b704e36b9618330454a072d40d2ba530371027e9c0bb2380e74bd146`。仅为离线同状态规划验证，没有派发新动作，没有新控制/任务资格；原冻结生产源码未热改。下一步需把修正接入独立版本，完成数值与控制资格及明确登记的接续/回归，同时继续检验参考时序和真实抓取失败，不能把QP修复等同于目标达成。

## 独立守护规划入口与故障点真实派发

新增 `dependency_safe_pose_preview.py`、`dependency_safe_preview_executable.py`、`dependency_safe_preview_runtime.py`，与冻结原入口分离；原成功路径结果逐值保持，仅实际恢复时记录 `qp_recovery`。包含旧 runtime/executable 和 null-space QP 的相关回归15项通过（本次复验2.31s）。独立入口在捕获状态通过原preview独立检查，命令与先前离线诊断逐位相同：`outputs/dependency_safe_preview_seed_20261027/verified.json`，SHA `9be2a9391b25cf9d7b6519b22f8267e21d30f28fd2b251a7dd75dd7eda9f774d`。

独立登记的CPU故障点派发探针已完成：精确重放原210周期，全部211端点SHA一致；同一失败状态确实触发守护分支，原四步preview独立检查通过（stationarity1.11e-11、complementarity0、预测差2.41e-13）。将修复后的seed arm命令与上一实际gripper命令组成action，验证幅值和变化率后，经固定原生控制器执行一步，仿真时间推进且状态有限。追加物理211=210重放+1新派发，WAM/fresh均0。此探针没有tracking refinement，gripper明确保持上一命令，不作为完整线上策略或任务成功证据。结果 `outputs/dependency_safe_seed_dispatch_20261027/result.json`，SHA `f37d40885727cab0f2d9c2a6fbf729a2719c67f79bc030ab1f6d714e57c1dfce`；owner `outputs/dependency_safe_seed_dispatch_execution_20261027/result.json` 已actual wait0/reap，外层进程exit0。

完整原18条守护入口回归已独立登记并执行：`outputs/dependency_safe_correct_reference_execution_20261027/registration.json`，固定4907候选+4907独立native replay周期、原40稳定步/.01m/.05rad门槛、要求18/18。记录本段时owner PID2031555仍存活，case06/00/01已关闭且性能达标，其余尚未全量关闭；不能宣称18/18完成。完整输入和图像审查仍单列待办。原WAM矩阵失败与未执行记录保持，未启动新模型或fresh验收；QP恢复不等于抓取或留出任务目标达成。

## 守护入口输入审查与完整控制事务探针

新增独立审查脚本 `audit_dependency_safe_robot_inputs.py` / `audit_dependency_safe_labels_images.py`，保持原审查逻辑，绑定新候选和新native replay路径。首批case06/00/01/02共8阶段全部完成并actual wait0/reap，owner exit0：`outputs/dependency_safe_audits_first_four_20261027/result.json`，SHA `4af9d2f722c047884811f38f18476008d3c182019ada0860a3c14f69047c8857`。范围包含保存端点重建M/J、关节/位姿/速度与因果历史、计划界限/变化率、SE(3)标签/mask、当前图像绑定以及独立native replay像素一致性；仅这四条，非18条全量审查。

新增完整tracking事务故障点探针 `probe_dependency_safe_tracking_dispatch.py`：从原210步实际命令重放，每步使用已记录的实际计划及nominal prediction，经stage/commit/真实endpoint observe重建因果历史，并逐步核对history_after；全部211前缀端点SHA与原记录一致。第210点独立守护seed检查通过，再用当前机器人M/J/q/dq/控制器参数与重建history调用原TrackingPreviewTransaction.prepare，实际commit/dispatch/observe完成第211周期，控制器实际位置/朝向goal符合预测。所选tracking route为unchanged_free_feasible，故此点证明恢复命令可穿过完整事务，不证明活动优化带来收益。gripper仍明确保持上一实际命令，未接完整夹爪在线策略。

探针结果 `outputs/dependency_safe_tracking_dispatch_20261027/result.json`，SHA `83bb0cf225a7b9b9c482d1d94e7dc2a8aaa9a3d0e8127e45eb89758c1d553f63`；owner `outputs/dependency_safe_tracking_dispatch_execution_20261027/result.json` actual wait0/reap、外层exit0。追加211=210重放+1新事务周期；不与上一seed-only探针211混淆，两者分别登记，均0世界模型/0fresh。新增5脚本ruff/格式检查通过。原WAM失败、未执行矩阵和最终门槛不变，总目标未完成。

同轮追加case03两项审查已actual wait0/reap、owner exit0：`outputs/dependency_safe_audits_case03_20261027/result.json`。累计五条/10审查阶段完成。最后进程检查PID2031555仍存活，原18回归case06/00/01/02/03/04六条已关闭，其余未全量完成，保持待核验状态。

## 修复版完整资格汇总与WAM准入门

新增 `close_dependency_safe_matrix.py` / `execute_dependency_safe_closure.py`，沿用原完整18条、36项审查、4907候选+4907replay、稳定40步/.01m/.05rad和actual wait0/reap条件，明确绑定修复版产物。持续监督入口 `supervise_dependency_safe_qualification.py` 已登记并启动，PID2038059，绑定当前producer PID2031555及其kernel start_ticks/command/cwd/CPU环境。它等待同一producer完整成功后才执行其余13条审查（先前5条已关闭）与完整汇总；不重新启动producer、不自动重试、不使用部分结果代替18/18。其registration位于 `outputs/dependency_safe_qualification_supervisor_20261027/registration.json`。本段记录时两进程均经ps确认存活；case00..06七条已关闭，case07 native replay执行中，不能宣称全量资格完成。

新增 `src/s2a/dependency_safe_control_gate.py`：先执行原完整控制准入，再核对真实恢复事务owned CLI、actual wait0/reap、210前缀+1派发与211历史周期、独立preview容差、实际恢复分支及源版本一致性，尤其要求四个守护入口源码确实包含于完整18资格中。已用真实旧18资格+新恢复事务做负向验证，准确拒绝“Guarded implementation actually qualified”，避免旧实现成功记录被当作新实现资格。证据 `outputs/dependency_safe_admission_negative_20261027/verified.json` SHA `b269c956cd4310df44bb5fe47dd3e08f0a25e5670e228740a9a27438ed3318a9`。这仅证明拒绝不充分证据，正向准入需等待新18全量关闭。新增4脚本/模块ruff检查与格式检查通过；0新增物理/WAM/fresh。原WAM开发失败与最终留出门槛继续保留，尚未启动修复版WAM模型。

## 预测闭合意图的独立候选（仅固定输入触发覆盖）

新增 `src/s2a/pose_gated_gripper_reference.py`，独立实验入口，未接入冻结生产策略。输入仅当前TCP、已生成TCP轨迹和opening序列；仅当第一个预测闭合在执行8步之后、且其位姿已接近当前TCP时，将该闭合opening提前用于当前8步，保持arm参考和之后opening不变。显式参数execute_steps8、opening阈值.35、距离.02m、SO(3)角.15rad；不读取object/接触/任务成功或未来观测，也不把这些经验门槛称作安全或成功保证。三项实际几何/时序测试通过：符合门槛的未来闭合、位置/朝向分别拒绝、已在执行段闭合或没有闭合时保持原序列；输入不修改，ruff与格式检查通过。

`audit_pose_gated_closure_candidate.py` 在计算前登记上述固定参数，再检查原三个已结束案例的全部225生成块。结果 `outputs/pose_gated_closure_candidate_20261027/verified.json` SHA `07ebc0a0d044bb7d6343e6879b3ab035168364d6c434d9b8de2bdd55787505da`：raw/task0/seed42触发8/75，terminal/task0/seed42触发2/75，raw/task0/seed43触发0/75。这是当时可用输入上的触发覆盖，并非真实反事实控制；尚无新物理/WAM/fresh，不据此宣称解决抓取。raw43不触发，说明此候选不能被当作共同故障解释。后续应以明确登记的成对物理探针检查闭合是否改善真实抓取、是否扰乱任务，固定录制参考实验与响应式WAM实验分别记证据等级。

同轮kernel检查producer PID2031555与supervisor PID2038059均存活，原18回归case00..08共九条已关闭；全量控制资格、正向WAM准入仍等待剩余案例和36项审查完整汇总。未启动新模型。

## 闭合时序的成对原生物理干预已启动

新增 `probe_pose_gated_closure_pair.py` 与owned入口 `execute_pose_gated_closure_pair.py`，已在原WAM CPU/OSMesa环境登记并启动（owner PID2041951，worker PID2042223，kernel检查均存活）。登记预算1200=600基线+600候选周期，WAM/fresh0，无自动重试。只取原raw/task0/init0/seed42：两组在原查询时刻接收相同录制参考、执行相同录制arm命令；仅候选按当前实测TCP和固定.02m/.15rad门槛改变opening，夹爪控制状态每组独立初始化并使用实际开度/速度/motor cache。基线重算夹爪命令必须匹配原记录，全部601端点必须匹配原SHA；候选首次改变命令前也要求同样前缀一致。

计划说明 `plans/v0.0.1/pose_gated_closure_pair_20261027.md`，实际执行前的权威预算/参数/源登记 `outputs/pose_gated_closure_pair_execution_20261027/registration.json` 与 `outputs/pose_gated_closure_pair_20261027/registration.json`。记录时worker仍运行，未获得完整成对结果；不宣称抓取改善。目标输出包括两组真实抓取接触步、目标高度增益、成功步、末尾和40步稳定结果。此实验使用固定录制参考和arm命令，不能替代响应式WAM或最终留出验收。两脚本ruff及格式检查通过。

同轮原18回归case00..10共11条已关闭；producer PID2031555与持续监督PID2038059均存活。尚无完整18/36项资格和新模型准入。

## 成对闭合干预的实际负结果与汇总失败

两组各600物理周期和完整steps.json已写出，但最终summary写入遇到NumPy bool_序列化错误，owner actual_wait1/reap、外层exit1，原result.json为截断JSON；不得把其中开头complete/passed=true片段当作成功资格。原源码、partial summary、失败owner与全部日志保持原样，没有重跑这1200周期。

新增只读 `review_pose_gated_closure_pair.py` 验证两份独立完整600步日志、原基线命令逐值一致及opening匹配、两组arm命令完全相同、有限动作与真实bool谓词，并核对实际wait1/reap和具体TypeError日志。审查结果 `outputs/pose_gated_closure_pair_review_20261027/verified.json` SHA `3df6d4a46ea87bfb69d6db79463c299f4758daccea6f2dac74d8bdb696299449`。它只恢复闭合生产日志能证明的结果，原owner仍failed，不认证独立physics replay或WAM/fresh。

基线：600步、夹爪命令改变0、开度范围[.4838520987,.9971162367]、抓取接触0、任务成功0、terminal/stable均false。候选：600步、首次改变step136，共77条夹爪命令改变，开度范围[.2951794919,.9969485155]，抓取接触仍0，任务成功仍0，terminal/stable均false。实验证明该时序规则在本固定参考/arm轨迹下确实改变闭合，但没有产生目标抓取或任务收益；不采用为已验证修复，不在同条数据上反复调门槛制造成功。后续优先检查实际夹爪与物体相对位姿、闭合深度和接触条件。未保存为完整有效summary的高度指标不从partial文件补造。新增只读脚本ruff检查通过，新增物理0、WAM/fresh0。

最后kernel核对原18 producer PID2031555与supervisor PID2038059均存活，case00..11共12条已关闭，尚无完整18控制资格。

## 保存端点的原生指垫接触几何审查

新增并执行 `scripts/audit_tracking_grasp_geometry.py`：以各自保存XML与全部601个原生state端点逐个restore/MuJoCo mj_forward，直接检查碰撞geom对，并记录物体中心在control site轴及指垫中点下的坐标、开度、指垫中心间距。三个完整WAM案例加正确参考case15共2404端点；无时间推进、无新物理/WAM/fresh。原生版本无mj_geomDistance API，因此不捏造碰撞距离；物体中心距离仅几何描述，不是抓取目标/安全间隙。contact.dist<=0的端点接触不代表控制周期内全部子步或接触力。

结果 `outputs/tracking_grasp_geometry_20261027/verified.json` SHA `e15a136e3c21dcd85ea5779adc236fee857cc604a2fc4eecc6330e4cd61d1256`。raw42/terminal42/raw43的any finger碰物体端点分别103/22/0、两finger分别19/17/0，但双侧pad端点均0；正确参考case15分别37/29/21。低开度<.35端点分别0/265/0/41；terminal42低开度时物体中心到pad中点最近.074498m，对照为.046922m，但不能把中心差直接等同于必要位置修正。raw42距离最近端点step136，opening.856624，pad只单侧接触；此处也是前述干预首次改变夹爪命令的位置。

原生只读源码确认Panda实际grasp分组正是finger1_pad_collision与finger2_pad_collision：`outputs/libero_env311/lib/python3.11/site-packages/robosuite/models/grippers/panda_gripper.py:29`、`.../robosuite/environments/manipulation/manipulation_env.py:201`（实际group选择229）。因此普通finger碰撞不能作为成功抓取证据。正确参考对照已另核对其独立native验证 `outputs/tracking_correct_reference_ordered_check_20261027/case15/verified.json`，performance_qualified=true、529成功步/末尾连续523步稳定；其原初态与raw42逐值一致，审查已断言。两轨迹仍非单因素因果比较，不能把性能差唯一归因于pose或时钟。

结论：单纯闭合提前没有支持收益；接下来需检查预测抓取位姿/闭合深度与实际接触几何，保持世界模型期望与实测状态区分。审查脚本ruff/格式通过。最后producer2031555与supervisor2038059仍经kernel确认存活，case00..13共14条已关闭，尚未宣称18/18或WAM新资格。

## 预测闭合位姿的静态反事实检查

新增 `probe_generated_grasp_geometry.py` 并完成检查：三个完整WAM案例全部75查询、使用当时保存的原生state，物体冻结在查询端点，仅将robot固定基座整体刚体变换以把control site精确摆放到当前已生成轨迹的第一个闭合目标。分别使用预测opening与opening0（实际指关节映射gap=.001+.079*opening）。每次验证site位置/朝向误差<=1e-9、物体位置完全不变；所有225查询先做identity relocation，并断言pad contact与原生端点一致。只用mj_forward，不执行时间步、不使用该摆放作为可执行action。

结果 `outputs/generated_grasp_geometry_20261027/verified.json` SHA `fcd591721df19662d6898fb5ea9050467f773a0a7c6c4489fc3ec4e1cc531447`：raw42有15个预测闭合查询，其中预测opening时双pad接触6、opening0时13；terminal42有40个，分别0/1；raw43有15个，分别0/7。这里的静态接触包含可能不合法的穿透，物体在未来本应运动、机器人关节可达性/碰撞/力闭合均未认证，因此不能把这些数值当作实际可抓取数量或成功率。

这削弱“所有预测闭合位姿都没有接触几何”的统一解释：raw42部分已生成未来位姿具有静态接触，值得以单个已生成块的连续跟踪/闭合物理探针检验执行段是否丢弃关键阶段；raw43则显示闭合深度可能影响几何，terminal42更多表现为位姿/任务阶段偏差。下一步应分别登记真实物理干预，不能从此静态结果直接改生产策略或声称唯一因果。新增脚本ruff及格式检查通过，新增时间推进/WAM/fresh均0。

同轮原18回归case00..15共16条已关闭，producer2031555及supervisor2038059仍经kernel确认存活；最后两条和完整审查尚未关闭。

## 单块已生成抓取阶段的真实控制续跑已启动

新增 `probe_committed_generated_chunk.py` 与owned入口 `execute_committed_generated_chunk.py`，登记预算352=两组各136前缀+40续跑周期；原raw/task0/init0/seed42，无新WAM/fresh、无自动重试。两组重建LoadedVelocity夹爪状态、tracking计划/预测历史及coupled创新，前缀137端点SHA必须与原一致；基线全部177端点复现原记录。候选只依序执行原step136已经生成的第17块40个target/opening，不使用之后查询生成块；每步读取真实当前机器人状态，经守护seed+原tracking prepare/commit/env.step/observe、实际selected command coupled模型更新，核对controller goal、动作幅值和变化率。保存全部177原生state以及40步完整控制日志。这个实验改变参考消费时序，不包含前一pose-gated opening规则。

计划 `plans/v0.0.1/committed_generated_chunk_20261027.md`；执行前源/预算登记 `outputs/committed_generated_chunk_execution_20261027/registration.json` 和 `outputs/committed_generated_chunk_20261027/registration.json`。本段记录时owner PID2049115与worker PID2049386已启动，worker经kernel确认仍存活，尚无完整结果。单块40步即使有抓取，也不能代替原600稳定任务、响应式WAM和fresh验收。

新产物全部使用既有preview_result_publication.publish_result，先序列化NumPy再独占创建，避免上次bool_导致partial summary。相关真实回归 `test_preview_result_publication.py`、`test_tracking_preview_transaction.py`、`test_dependency_safe_preview.py`、`test_dependency_safe_qp.py` 共16项通过（11.82s），新两脚本ruff检查/格式通过。

同轮原18 producer已关闭17条、17条performance_qualified=true，完整result尚未产生。producer2031555和supervisor2038059均仍存活，最后case17及剩余审查/完整汇总待完成，未宣称全量资格。

## 守护QP版本完整控制资格已关闭

完整原18回归、4907候选+4907独立native replay周期、全部36机器人输入/因果历史/标签/图像审查已完成，保持稳定40步/.01m/.05rad且18/18达标。最终证据 `outputs/dependency_safe_matrix_closed_20261027/verified.json` SHA `27f81add4bba87f65e65180e88bba29e6d21c44618705062716562a8aa633fc7`；closure owner SHA `2c34f507c761448027499d799c1ddc618e0be2f109da0fe1347dbeb898615797`；监督owner `outputs/dependency_safe_qualification_supervisor_20261027/result.json` SHA `1a37d20064c648e5439c4bee95c6fb48d57464d2955c578e9d96e7fb81e18eb7`，实际owned stage wait0/reap并外层exit0。producer、监督进程均已正常结束，不再报告运行中。

逐行汇总新18全部4907周期均unchanged_free_feasible、seed_qp_recovery0；因此这证明正常参考上的行为保持，不是活动优化收益。已捕获QP故障的恢复来自另行登记的完整真实事务探针。使用新完整18资格、closure owner、既有54活动周期事务证明与新恢复事务，`admit_dependency_safe_control` 正向准入通过：`outputs/dependency_safe_wam_admission_20261027/verified.json` SHA `9dfb95d80f5bcec16b750260fe993c0bc469d42e901f59de40fee293c2e8486d`。这只是修复版下一轮WAM准入，未执行新WAM模型，不认证任务或fresh成功。

## 单块连续跟踪完成，局部未抓取

`outputs/committed_generated_chunk_20261027/result.json` complete/passed=true，原登记352周期全部执行，owner actual_wait0/reap、外层exit0。基线与候选40续跑均0抓取接触、0任务成功。候选最小opening .00562183，确实接近全闭；但其自身40点TCP参考下posRMSE .0136126m、SO(3)RMSE .154499rad，超过原.01/.05门槛；全部tracking route interior_witness。基线40段posRMSE .0120102m、SO(3)RMSE .0207738rad，但基线误差针对原重规划参考，不能把两者当相同目标下的性能对比。此结果不支持将连续单块执行作为已验证任务修复。

新增 `check_committed_generated_chunk.py` 和owned入口独立回放候选176周期，登记追加176物理预算、0WAM/fresh；核对全部177原生state、真实task/grasp谓词、实际controller goal、40点TCP误差和全部关节界限。记录时worker PID2051890仍经kernel确认存活，尚未获得其完整结果，不提前宣称独立验证完成。新两脚本ruff/格式检查通过。

同轮独立176周期回放现已完成：`outputs/committed_generated_chunk_check_20261027/verified.json` SHA `a551d0cf011bfc0cd9ff5ef767d293654e8fcbf1d6cdebe6d77b78aa7b0ad1e5`；全部177状态与候选逐值相同（最大差0），40步真实grasp/task均0，独立误差与生产一致。40步及两端点各关节均无越界，q7最小实际margin .05008385rad，其他关节最小margin均>.95rad；这说明本段较大朝向跟踪误差伴随q7贴近既定.05缓冲，但不单凭此断言唯一因果。owner actual_wait0/reap、外层exit0，worker已正常退出。追加176独立回放物理与原352生产分开计数；不重定义为任务通过。

## 朝向误差与显式正缓冲候选

将单块续跑40周期的实际TCP误差与同周期保存的joint-model first预测误差分解：实际posRMSE .0136126m/rotRMSE .154499rad，所选plan模型预测.0123442m/.154108rad，seed经同joint-model预测.0122609m/.144503rad。所有周期interior_witness，内层最小margin约.05。由此该段误差主要已经出现在规划预期中，并非仅执行时出现未建模朝向漂移；但仍不能据此把唯一因果归给某一个约束。

新增独立候选 `src/s2a/configurable_tracking_terminal_plan.py`，从冻结tracking_terminal_joint_plan派生，只将请求的正关节buffer显式参数化，默认.05；保持实际关节界限、动作幅值/变化率、内层2ms约束、终端非外向条件和原tracking目标/TCP指标不变。未改生产策略或验收门槛。新增两项真实约束测试及原一项共3项通过（4.42s）：默认命令/目标值精确一致；.01正buffer在构造的受限状态改善跟踪模型目标，同时保持真实界限、正buffer、终端条件和输入targets不修改。ruff/格式通过。

`probe_tracking_buffer_tradeoff.py` 首次离线检查在恢复状态重算的.05命令与历史记录不逐值相等时停止，最大差.00045786，先前objective差检查通过；未产生完整verified，新增物理0。原登记/源码和独立failure_review.json保留，不放宽该历史匹配测试伪称通过。v2独立入口 `probe_tracking_buffer_tradeoff_v2.py` 改为对同一恢复输入调用原solver及可配置默认solver，严格逐值比较；再比较.05与.01。登记固定原cursor0/8/20/30，0新增物理/WAM/fresh，不恢复或执行新动作。记录时kernel PID2054889仍在运行，尚无完整结果。正buffer缩小只属于候选模型比较，不能据此推断真实安全性或任务收益。

v2现已exit0完成：`outputs/tracking_buffer_tradeoff_v2_20261027/verified.json` SHA `4d36b88e3b9158b4bfd40c07d47c76372e5965674e78f1bc6100fd71305793a7`。四个完全相同恢复输入上，可配置.05与原solver命令逐值/目标值精确一致。buffer .05→.01使cursor0/8/20/30 whole-plan模型目标分别 .7449996→.1457161、4.2255862→2.3708087、40.6453485→29.1250114、113.2975345→93.1531092；first朝向误差分别 .0075565→.0052130、.0419809→.0364723、.1066761→.0944932、.2625368→.2520739rad。各自实际模型内层最小margin达到请求正buffer，原物理范围未变。较小buffer有局部模型收益，但晚段误差仍明显超.05门槛，不能把它作为已解决跟踪/抓取的方案；尚未执行该候选真实控制或取得新资格。

## .01正缓冲候选的真实物理对照已启动

新增独立 `buffer01_tracking_hybrid.py`、`buffer01_tracking_native.py`，主要hard interior分支调用可配置solver的.01rad正buffer，保留原adaptive/soft后备；未改已登记生产源码。`.05 requested_margin_satisfied`仍按原.05目标单列报告，所选实际joint_margin_rad记录.01，不能混淆两者。新增 `probe_committed_chunk_buffer01.py` / `execute_committed_chunk_buffer01.py`，沿用原单块40步真实控制闭环与完全共同初态/前缀，实验登记明确joint_buffer_rad=.01；预算352=两组各136+40、0WAM/fresh，无自动重试。owner PID2056634与worker PID2056723经kernel确认存活，尚无完整结果，不宣称改善任务。

计划 `plans/v0.0.1/committed_chunk_buffer01_20261027.md`。执行前权威源/预算登记 `outputs/committed_chunk_buffer01_execution_20261027/registration.json` 和 `outputs/committed_generated_chunk_buffer01_20261027/registration.json`。候选若完整结束，已准备独立回放 `check_committed_chunk_buffer01.py` / `execute_committed_chunk_buffer01_check.py`，另计176物理预算，核对全部177状态、controller goal、真实任务/接触、TCP误差与关节margin，当前尚未启动该回放。

相关可配置solver/原solver/真实tracking事务共6项测试通过（6.13s）；新增六文件ruff及格式检查通过。原18完整资格仅适用于.05既有修复版，不转移为新.01候选资格；新候选仍须实际验证及原门槛。

## .01正缓冲物理对照完成，仍无抓取成功

`outputs/committed_generated_chunk_buffer01_20261027/result.json` complete/passed=true，原登记352周期全部执行，owner正常wait0/reap、外层exit0。原基线steps/states指纹与前一.05实验基线完全相同；branch元数据joint_buffer_rad=.01仅说明实验配置，baseline本身只回放原动作、不运行该缓冲规划。候选40步均interior_witness，grasp/task仍0、terminal/all40_success均false。相比同一单块.05候选，实际posRMSE .0136126→.0125402m、rotRMSE .154499→.133751rad，最小opening .00555458，仍超原.01/.05误差门槛，无任务收益证据。不能据局部误差下降宣布完成或转移原18资格。

已按计划启动独立176周期回放 `execute_committed_chunk_buffer01_check.py`（追加物理176、WAM/fresh0），实际进程仍在运行，尚未获得完整独立结果。它核对全部177状态、actual controller goal、40步真实grasp/task/TCP误差与实际关节margin。原.05已资格版本和新.01实验源码/登记均保持独立，未修改世界模型或启动新GPU任务。

同轮独立回放现已exit0完成：`outputs/committed_chunk_buffer01_check_20261027/verified.json` SHA `f2c45d46468ad678a93e21a64bce47fa6b1c4c77ca22fb1f05c2fb3f2c23d221`，177原生state最大差0，40步grasp/task均0，误差与生产完全相同，全部关节端点无越界、q7最小margin .0101122163rad。owner actual wait0/reap。此处只证实所保存控制端点，不扩大为所有2ms内环无越界或普遍安全保证。该对照已关闭，不再报告运行中；未达到任务/误差门槛，不采用为已验证修复。

## v4完整WAM开发版本准备与实际启动

新独立v4包仅接入已通过完整18资格的dependency-safe seed；保留原.05 tracking缓冲、夹爪与12案例矩阵，未采用未成功的闭合提前/单块/.01候选。新增 `dependency_safe_wam_admission.py` 在所有实际入口绑定新18/36资格和恢复事务，固定admission证明与真实launch/exit/release/dispatch指纹共同冻结。实际运行时的seed函数经import核验来自 `s2a.dependency_safe_preview_runtime`。独立checker端口31365改为实际31375；完整verifier采用已审查的实际argv/配置SHA+prepared immutable source+runtime manifest绑定，明确不捏造不存在的execution_snapshot。

入口相关22测试通过4.11s；完整组件122测试通过17.42s；实际Python3.11下全部12输入预检通过，模型/物理调用0；16新增文件ruff/格式通过。首次prepare使用.venv因缺h5py在新运行root创建前退出，物理/模型0；未安装或修改共享环境，改用已有仓库内native Python完成冻结。prepared `outputs/tracking_preview_wam_development_v4_20261027/prepared.json` SHA `316e1f498170bb81403b11910462a86216a977d5322499f84128bab89c3c77eb`；独立metadata SHA `8ed3b5c9ebf7e601e93a4c432fe0c7d3033fbcdec23f8e5da2c3eebf5e7b2c5f`。14实际CLI dry均return0（owner/service/全部12collector），独立execution registration SHA `6da21b2c7ee678f723ece06235944017aaca5b066e5843fe234ff4746caef997`。

计划 `plans/v0.0.1/tracking_wam_v4_20261027.md`。启动前允许UUID显存0MiB/利用率0，31375/31376端口空闲。按同一已通过dry的owner argv改--launch启动，launcher实际exit0；`supervisor/launcher_result.json` SHA `3bfaafc0717cbd7fd3e6276abb232a2b2d70390c28aae9a319bca273e59192be`。owner PID2061685和model service PID2061750已由kernel确认存活，service实际argv只指定允许UUID GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f，当前处于初始化阶段，尚无新案例性能结果。不能把launcher或metadata通过当作WAM任务通过。

固定完整预算一次初始化、900请求/31500 denoise/head、7200物理周期、12×600，fresh0且无自动重试；terminal至少5/6并各任务至少1/2稳定的原门槛保持。v3失败及其实际252请求/2010周期保留，v4另记，不合并伪造900/7200结果；未授权新fresh任务。后续以同一live handle及实际产生的case/service结果推进，不因观察超时重启。

## v4初始化期间的实际进程检查与逐案例审查准备

继续绑定同一owner PID2061685/service PID2061750，kernel确认均存活；service仍初始化、GPU显存3MiB/利用率0，尚无600/75完成案例。实际wchan=wait_on_page_bit_common，读取计数rchar2052971780/read_bytes351690752，属于当前I/O等待观察，不以日志未输出或初始化慢判断进程终止，不重启服务。服务CUDA实际环境仍绑定允许UUID、cwd为当前S2A。

新增 `audit_tracking_wam_completed_case_v4.py` 与 `execute_tracking_wam_case_review_v4.py`，从已使用的独立闭合案例审查入口派生，仅绑定v4路径、v4正确端口checker/完整verifier及新审查目录。已事先登记全部12案例，严格要求完整600步/75请求、actual collector wait0/reap后才允许单条审查，保留来源/当前输入/独立数值与标签图像检查；不把单条审查当完整900 denoise/head账本或任务门槛通过。登记 `outputs/tracking_wam_independent_review_v4_20261027/registration.json` SHA `7351f3d364b11bbbbe8ad4301893f1b7770436e13f1b55a1b9420ccf457bc9c0`。新增审查物理/模型/fresh预算均0，两新脚本ruff/格式通过；未修改任何冻结生产文件。

## v4同一模型完成初始化并进入首条案例

持续对同一kernel identity观察，未重启。service PID2061750/start_ticks604992954现已输出server_ready=true、实际host127.0.0.1/port31375，CUDA初始化完成；GPU仅允许UUID，显存33717MiB。owner PID2061685仍存活，首条task0_init0_s42_raw collector PID2064980已实际启动并产生cv2_readiness/http/simulation目录，真实CPU CUDA/LD均空。模型日志记录/status与/bind_case成功；尚无完整600/75案例或任务性能结果。

独立观察快照 `outputs/tracking_wam_v4_initialization_observations_20261027/server_ready.json` 绑定实际ready JSON和service/collector的kernel身份，不将就绪等同于任务成功。下一步继续检查当前真实模型调用与物理控制进度，只有完整案例结束且actual wait0/reap后才执行已登记的独立案例审查。原完整矩阵/门槛保持。

## 2026-10-28：v4完整生产结束，任务门槛失败，独立验收未通过

中断后以实际文件/kernel重新核验：原owner/service/首collector均已终止，全部12 worker complete/passed=true，各600周期/75请求；生产orchestration complete/passed=true。服务一次初始化、900请求、31500 denoise/head全部完成，物理合计7200；这只表示预算及执行完成，不是任务成功。原独立验收进程actual_wait1/reap，最终owner complete/passed=false，未自动重试。GPU after记录released=true，当前允许UUID再次查询0MiB/0%，无残留任务。

独立验收失败位于 `tracking_preview_arm_log_replay.py:157` 的响应矩阵重算：1/72元素差1.12329417e-8，原atol1e-8/rtol0未通过。冻结的v4生产与验收源码、失败记录保持原样，不放宽门槛。新增独立 `current_coupled_precision_independent.py` 使用五点SO(3)数值差分与更严格DOP853积分，保留不同于生产expm的验证方法；`audit_tracking_wam_response_precision.py` 在全12保存输入上检查同一原容差。首次三条600均通过、最大差约1.1e-13，但第四条task0/seed43/terminal058 step512 goal_forcing_response仍差1.09811253e-8而停止；因此不能把差异简单归因为积分/差分容差，也不宣称完整独立验收通过。该追加检查物理/WAM/fresh均0，源与登记/前三条闭合证据保留，exit1。

新增 `describe_tracking_wam_v4_outcome.py` 读取全部7200生产日志，固定目标对象、原task谓词和最后40步。结果 `outputs/tracking_wam_interactions_v4_20261028/all_twelve.json` SHA `fef0a2f4925d35aeeb0d703219d36299832e7a8234b2ee7b06b21c52619ddd3c`：12/12均0任务成功步，terminal/stable均false，故原terminal>=5/6且每任务>=1/2开发门槛明确未满足；fresh继续关闭。该摘要是已完成真实生产日志描述，不替代失败的完整独立审查。

守护QP确实在完整运行中启用3周期：task0/seed43/terminal058两次、task2/seed43/raw一次，均继续完成600，原故障已不再阻止矩阵执行。task0/seed43/terminal058有50个目标grasp_contact生产步，但任务成功仍0；其他11条目标grasp_contact均0。良好TCP tracking不等于任务成功：例如task1/seed42/terminal058 posRMSE.00135m、rotRMSE.00448rad，却无抓取/任务成功。q7仍有实际越界日志（raw task0/seed43=258、task1/seed43=136、task2/seed42=202、task2/seed43=257；terminal task2/seed43=275周期），不能声称已普遍解决关节限制。完整数值矩阵差异与真实任务/安全失败分开记录；后续需定位独立重算差异，并针对抓取/阶段参考和实际关节约束继续实现，总目标未完成。

## 2026-10-28：独立响应重算的稳定线性代数修正

精确定位原失败样本task0/seed43/terminal058 step512：独立方法显式inverse mass形成J M^-1 J^T，mass条件数11.49、Jacobian条件数3198.17，逆与乘积舍入差在耦合矩阵放大。五点差分/更严ODE本身不能消除此差异。另行新增 `current_coupled_stable_independent.py` 改用求解M X=J^T后左乘J，保持独立五点SO(3)导数与DOP853积分，不调用生产expm，也不改任何验收容差。同一样本coupling差2.0773e-9→0、goal_forcing_response差1.0981e-8→4.6612e-12。

`audit_tracking_wam_response_stable.py` 独立全12/7200周期保存输入检查已exit0完成：`outputs/tracking_wam_response_stable_20261028/verified.json` SHA `29e5d026e2904c7d8148414d2be302ee5b88ea999f9516e91fce90f16d0f92b0`，全部原atol1e-8/rtol0通过，最大矩阵差9.60654e-12。范围仅同保存输入的响应矩阵，不等于完整source/原生状态/seed KKT/900请求账本通过。原失败检查与partial闭合记录保持原样。

新增带原steps.jsonl路径/SHA/step512的fixture `tests/fixtures/coupled_near_singular_response.json`，以及独立临界阻尼闭式解和真实近奇异样本回归。首次在.venv Python3.12/NumPy2环境直接比较历史矩阵不通过（compiled数值环境不同）；真实历史回归现明确在记录使用的原生Python3.11/NumPy1.26.4中运行，不放宽历史容差。默认pytest调用会实际执行该原生子进程；两项测试均通过4.66s，闭式解仍在当前pytest环境执行。

新增只读独立修订链 `coupled_pose_preview_stable_independent.py` / `tracking_preview_arm_log_replay_stable.py` / `tracking_preview_wam_acceptance_checker_stable_review.py` / `verify_tracking_preview_wam_development_stable_review.py`。仅依赖稳定独立响应方法，原数值/物理/来源/任务门槛保持，原生产与失败verifier不热改。`execute_tracking_wam_stable_review.py` 已独立登记并启动全12/900请求/31500头节点审查，另计物理0/WAM0/fresh0；新schema为verification-stable-review，不冒充原失败验收。登记与实际owned stage位于 `outputs/tracking_wam_stable_review_execution_20261028`，当前仍运行，不能提前宣称完整通过。

此修正不改变真实任务负结果：12条均无任务成功，原开发门槛未满足，fresh继续关闭。总目标未完成，审查差异澄清后继续针对真实抓取/阶段参考和关节限制实现。

## 2026-10-28：抓取后释放的因果候选与物理对照

v4 task0/init0/seed43/terminal058的50个grasp_contact步分布在58–72、81–98、102–110、116–123，最大目标抬升.099866m，但task始终false。step73真实gripper route从native_close_load转model_tracking，reference .10722556、历史minimum .04112872、release_level .0891908，原短时上升/clearance条件确认后command=-1；实际开度随之上升、接触丢失。此为时序线索，其他contact丢失也发生在closed hold，不据此宣称唯一因果。

新增独立 `gripper_release_floor.py` / `loaded_velocity_release_floor.py`，只在原release latch中增加reference>=.35的明确张开条件，其他原excursion/clearance/time条件保持。当前opening参考原值仍记录，不输入object/contact/task或未来观测。它有意改变意图解释，不能称作连续开度重建精确或普遍安全。小回升不解除闭合、明确张开可释放及独立reset两项测试通过.64s，ruff/格式通过。

首版物理探针误绑定v3未完成210步的worker，启动前Closed original600检查拦截，owner actual_wait1/reap，物理/模型0；原失败代码和产物保持。独立v2 `probe_gripper_release_floor_pair_v2.py` / `execute_gripper_release_floor_pair_v2.py` 已预检查v4实际600/75 worker并启动，源/预算登记位于 `outputs/gripper_release_floor_pair_execution_v2_20261028` 与 `outputs/gripper_release_floor_pair_v2_20261028`。原登记1200=完整600基线+600候选，seed43，原固定75参考块及600arm命令，两组独立初始化夹爪；baseline601端点及重算夹爪命令必须复现原轨迹，候选动作改变前也需前缀一致。保存全部状态/动作/抓取与任务谓词，0世界模型/fresh，无自动重试。当前v2仍运行，未有完整结果；不将固定参考探针当成响应式WAM成功。

计划 `plans/v0.0.1/gripper_release_floor_20261028.md`；其中首版路径保留历史声明，实际正在运行的安全版本按上述v2路径定位。完整数值修订审查owner2313788/worker2314073同轮kernel确认仍运行，尚未宣称独立全量通过。

## 释放floor物理结果与原完整控制回归

v2成对1200周期已完成并owner actual_wait0/reap、外层exit0。基线重现原600命令和全部601端点，grasp50步、最大抬升.0998661m、success0；候选只改变39条gripper命令，第一次step73，grasp56步、最大抬升.1034976m、success0。基线接触段58–72/81–98/102–110/116–123；候选58–98/102–103/106–110/116–123，第一次连续接触15→41步。该对照支持原step73早释放对接触中断有贡献，但不能证明它解释全部失败；候选仍不完成任务，不采用为已验证修复。

已启动 `check_gripper_release_floor_pair.py` / `execute_gripper_release_floor_pair_check.py` 独立重放候选600命令，追加物理600、0模型/fresh，要求全部601state和实际task/grasp/开度/高度一致。当前仍运行，尚无独立结果。启动命令的lint检查报独立checker导入排序I001，但shell随后仍执行了已登记版本；该格式问题不宣称已通过，冻结checker源码保持不热改。后续版本需在冻结前修正，不能以事后改源码破坏证据指纹。

同时新增独立完整18资格入口 `run_release_floor_correct_reference.py` / `check_release_floor_correct_reference.py` / `execute_release_floor_correct_reference.py`，明确使用loaded_velocity_release_floor，与旧.05 arm/guarded seed组合，原18/4907候选+4907native replay、40稳定/.01m/.05rad门槛及数据划分保持。source和release_reference_minimum=.35已登记 `outputs/release_floor_correct_reference_execution_20261028/registration.json`，实际执行已启动，case06已关闭且达标，其余待完成。原runner复制保留method_contract.gripper的旧描述字符串，已用单独metadata_clarification.json明确实际新模块/参数；实际代码和登记不改，不把文字标签当资格依据。三个资格脚本ruff/格式通过。

完整稳定数值修订审查owner2313788/worker2314073同轮仍经kernel确认存活，未取得完整结果。新释放策略无响应式WAM或fresh结果；总目标保持原范围，未完成。

同轮600周期独立命令回放现已完成：`outputs/gripper_release_floor_pair_check_20261028/verified.json` complete/passed=true，全部601state最大差0，原生grasp56、success0及接触段/高度与生产相同；owner actual_wait0/reap、外层exit0。该记录只认证固定参考对照的局部接触保持收益与任务负结果，不认证新WAM或留出成功。

## 2026-10-28：放置判据与固定参考的对齐缺口

只读核对原BDDL goal为On(akita_black_bowl_1,plate_1)。原生object-state check_ontop要求plate中心z<=bowl中心z、两物接触且水平中心距离<.03m；原源码路径和SHA已在独立审查登记，未作为在线策略输入。`audit_release_floor_placement.py` 从各自XML和601个保存state逐端点mj_forward，按原接触geom及判据独立重算，所有600 task booleans与生产一致。`outputs/gripper_release_floor_placement_20261028/verified.json` SHA `cacaae20e1b62eca10dcdaf0459127c8be04d8f80e74f55adee9e0a0fe2a0aff`：baseline/candidate均0个水平对齐端点；bowl–plate接触端点23/127，但成功均0。最近水平距离分别.03735185/.03745005m，均在endpoint91且仍grasped，超过原.03门槛。

`audit_release_floor_placement_tracking.py` 另行登记理想TCP的刚性握持投影：在每个实际grasped端点取当前TCP→bowl变换，用原当时desired_tcp代替实际TCP，保持物体相对握持假设不变。baseline/candidate投影最小水平距离.03588136/.03594286m，仍无<.03端点。证据 `outputs/gripper_release_floor_placement_tracking_20261028/verified.json` SHA `ea563cd48281a2b27813de52d708a75cb2a00d052a447a4316e35b4f907b2b96`。这是局部几何分解，不是独立动态可达性/抓取力/任务不可能性证明；物体滑移会改变握持变换，且固定参考没有依据候选新图像重新生成。因此不能把它扩大为响应式WAM结论，也不使用特权object/plate位置生成在线补偿。

这表明本固定参考实验中，减少step73早释放确实延长连续接触，但仍存在放置阶段水平对齐缺口；第99步丢接触时native_close_load command0，没有主动释放，不能把所有失败都归因于release。新两份审查均0时间推进/模型/fresh，ruff/格式通过。

同轮释放候选原完整18回归已关闭7条、7条性能达标，owner2322188仍存活；全量稳定独立审查owner2313788/worker2314073仍存活且CPU时间推进，未有最终结果。原门槛与总目标保持，不提前宣称候选资格或新任务成功。

## 释放候选完整审查与汇总监督已准备

新增 `audit_release_floor_robot_inputs.py` / `audit_release_floor_labels_images.py` / `execute_release_floor_audits.py`，沿用原全部机器人当前输入、完成历史、计划界限/变化率、SE(3)标签/mask与独立native replay像素审查，仅绑定新候选路径。新增 `close_release_floor_matrix.py` / `execute_release_floor_closure.py` 要求原18完整performance、4907候选+4907回放、全部36唯一审查、actual wait0/reap，并明确release_reference_minimum=.35和实际source/policy说明的登记绑定。未缩减案例或阈值。

`supervise_release_floor_qualification.py --producer-pid2322188` 已登记并实际启动，supervisor PID2332127，绑定现有producer的kernel start_ticks/command/cwd/CPU环境；只在同一producer原完整18性能通过后，执行全18×2审查和最终汇总，不自动重试或启动替代producer。登记根 `outputs/release_floor_qualification_supervisor_20261028`。六新增脚本ruff/格式通过，新增物理/模型/fresh均0（已有生产预算仍分别登记）。

最后kernel检查producer2322188、supervisor2332127和完整稳定审查owner2313788/worker2314073均存活；释放候选已关闭11条、11条性能达标，余下7条和全36审查/汇总尚未完成。原v4 full review仍在计算、CPU时间推进，尚无最终独立通过记录。不据部分结果启动新WAM/fresh，目标保持完整原范围。

## 释放策略的新WAM准入绑定

新增 `src/s2a/release_floor_control_gate.py`：先要求guarded seed原完整控制/实际事务证据，再要求qualification明确release_reference_minimum=.35、metadata/source实际策略绑定、新gripper/latch模块确实包含于该18资格；另要求独立600回放的实际CLI、wait0/reap、全部原生state重现及真实动作干预，绑定1200成对生产和候选完整states/steps指纹。它只作未来控制准入，不改变原WAM或fresh任务门槛，不以局部41步抓取宣称任务成功。

已用真实旧guarded18资格+新release物理回放做负向检查，准确拒绝旧策略代替新策略资格，证据 `outputs/release_floor_admission_negative_20261028/verified.json` SHA `b142d731205b944f6339ac1b87afbdb567bb12abbe24312167f8bbf673711361`。新gate ruff/格式通过，追加物理/模型/fresh0；正向准入仍需等新18和36审查全部关闭后实际核验，不能提前授权新模型。

最后kernel检查释放producer2322188与supervisor2332127均存活，已关闭12条、12条达标；完整稳定独立审查owner2313788/worker2314073仍存活，CPU时间推进，尚无完整结果。当前不启动新WAM/fresh，不把部分控制通过替代原总目标。

## 2026-10-28：释放策略控制资格关闭、审查运行时修正

释放候选完整18性能回归及全部36项当前输入/历史/标签/图像审查、actual wait0/reap汇总已全部关闭：`outputs/release_floor_matrix_closed_20261028/verified.json` SHA `f75baab6041ef74c0819d0eb46358914c7409c1b0fd4dd160f2ebc7316d44e1c`，18/18、4907候选+4907native replay、原40稳定/.01m/.05rad门槛保持。新gate正向实际调用通过，`outputs/release_floor_wam_admission_20261028/verified.json` SHA `2a635dcbbe2256bbd3e8cf59a03981e8d82b5af8b6c9f969b076450bef7ab341`。这是该策略的新WAM控制准入，不是WAM或fresh任务成功。

前一稳定全量审查实际达到原audit_timeout_s=1800而TimeoutExpired，owner失败、worker被SIGTERM且reaped、returncode-15；没有exit.json/actual_wait0成功证据，不再报告该进程仍运行或全审查通过。原失败/timeout登记保持，未自动重试。另一个释放decision审查在dataclass tuple与JSON list的reference_endpoints编码比较处停止，非物理/模型新增失败；原脚本登记保持，独立新版本在进入独立checker前严格按实际JSON编码规范转换。

为保持原完整枚举而消除Python递归瓶颈，新增独立C ABI标量枚举 `scripts/velocity_scalar_audit.cpp` 与 `velocity_plan_compiled_audit.py`。所有3^h(h<=8)路径、25内层motor增量、稳定补偿残差求和、非有限/速度拒绝计数和原最优性/预测容差保持。只编译审核计算，实际producer控制算法未改。编译禁止fast-math及FMA收缩，g++版本/flags/source/binary SHA登记 `outputs/velocity_scalar_audit_native_20261028/build.json`。wrapper对所选sequence仍用原独立Python scalar_step逐点复核，完整枚举返回计数和全局最小cost。冻结真实参数fixture下h1/4/8与原Python标量审查结果对照、拒绝错误cost，三项测试通过.73s；未减少候选或放宽1e-10容差。

新独立释放检查 `audit_release_floor_decisions_fast.py` 对真实600个归档当前state和实际命令完成检查：重建opening/velocity/current TCP/25×.01 motor cache，原参考按实际8步到达，重算producer决定必须等于已实际执行动作；独立latch/MPC随后完整审查。JSON编码边界正确处理，600/600通过，model_tracking515/native_close_load85，`outputs/gripper_release_floor_decisions_fast_20261028/verified.json` SHA `b7ed87342e964b8bbf8cd66c6be39de8c4b13978018c3f6f29de69e4825e2efd`，物理/模型/fresh0。该检查确认新策略的独立审核可用，不把既有任务0成功改判为成功。

新全量 `execute_tracking_wam_fast_review.py` 已独立登记并实际启动，调用原同范围12/900/31500审查、原1800秒时限，数值稳定response方法及上述完整标量native枚举，原source/图像/物理/任务容差和score不变；新source、C++ source及binary/build指纹均冻结在 `outputs/tracking_wam_fast_review_execution_20261028/registration.json`，不是覆盖前次失败。当前该审查仍运行，尚无完整结果。新增各Python文件ruff/格式通过；原生产GPU仍释放，未启动新WAM/fresh。

## 2026-10-28 v5完整开发矩阵准备

新v5只改变已经过完整控制资格的释放参考最小阈值.35，保持守护QP、.05关节buffer和原12条/900请求/31500头调用/7200周期预算与任务门槛。详见plans/v0.0.1/tracking_wam_v5_20261028.md。独立释放决策审查和native完整标量枚举绑定纳入新准入，不用旧夹爪策略资格替代新策略证据。

组件QA129通过、实际12条CPU输入预检通过、独立metadata complete/passed且cases_verified12。prepared SHA 64ff281922755538537ec87e70e744788f04f82c4bd32a982d5033ace42525f7。此阶段模型/物理新增0，实际14 CLI dry正在登记执行，尚未启动模型。v4 fast review kernel owner2347456/worker2347553仍存活，尚无完整审查结果。原总目标未完成。

实际14 CLI dry全部returncode0，successful_cli_dry.json complete/passed；registration SHA 36e99f5aacede043c7212eb9f373f64f480931d1e45d2b79596b8c68abb1b3f9。已从该记录取owner原argv，仅将--dry-run替换--launch，执行持久入口；启动过程尚待kernel身份核实，不提前声称模型已初始化。

持久launcher实际退出0，launcher_result complete/passed，SHA 1fc727514cf058a7f504ab51edac3436bb7d1da9c9f1d6172f728d6172e82c00；kernel owner PID2358795存活，cwd为当前仓库，CPU CUDA/LD均为空。ready/launch绑定已建立，stage_completion_claim=false；这只证明持久owner启动，不证明模型初始化或生产完成。v4独立fast复核owner2347456/worker2347553同次检查仍存活。

后续kernel核实v5服务PID2359094为owner2358795子进程，实际CUDA_VISIBLE_DEVICES仅GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f、LD为空，cwd/来源沿登记。进程仍在导入外部只读模型环境，实际fd指向transformers/models/gemma3/image_processing_gemma3_fast.py，I/O字节持续推进；GPU尚无该模型占用，未声明模型初始化完成。观测保存于outputs/tracking_preview_wam_development_v5_20261028/runtime_observation_20261028T0328.json（内部observed_utc为实际时间）。同次v4 full review仍活跃，已读到请求428附近，原1800秒deadline保持，没有观察超时重启。此轮为核实运行中的等待，未新增物理/模型成功结论或开放fresh。

## 完整节点审查的有界并行读取准备

新增scripts/verify_wam_terminal_points_prefetch.py，以8线程、每线程私有Evidence读取单请求全部35 JSON/NPZ，在主线程合并相同SHA映射，然后调用原ordered audit_points逐点执行全部数学/摘要链/顺序检查。原Evidence.finish全量重读指纹保持；最多暂存一请求，不改变控制或数学容差。未接入正在运行的v4审查或v5冻结源码。

首次check_terminal_points_prefetch.py误为raw传入非空head，原顺序审查首先以independent W1 clean residual拒绝，未生成通过记录；保留原脚本。新check_terminal_points_prefetch_v2.py按生产raw=None、terminal=显式head pair修正。实际CPU执行退出0；真实请求0(raw)和75(terminal)各完整35节点、70文件结果/首样本/全部来源映射一致，错误point SHA及最终stream SHA均拒绝。证据outputs/terminal_points_prefetch_check_v2_20261028/verified.json SHA 69a2c8acd7cbfd563fdeef15456149f58b81eebc428ad9c5cc3cbd3ae8927b94。原/新时间分别.617/.183s与.281/.224s；顺序及缓存影响未控制，不能据此声称全量加速倍数或900请求资格。新脚本ruff/格式通过，新增物理/模型/fresh0。

最后检查v5服务2359094仍活跃且读取量推进，当前导入torchvision；无模型初始化完成记录。v4审查2347553仍活跃，原1800秒时限不变；没有重启或覆盖任一运行记录。

## v5模型服务实际就绪

服务PID2359094的ready.json已产出，SHA 690a84ecb185a286462df36391b0944a3838e6cba365698508e06ee06cf886e0；model_initializations=1、model_initialization_attempts=1，process_identity明确获准GPU UUID及owner2358795。第一个task0_init0_s42_raw已bind，采集子进程2365001实际存在，cwd当前仓库，CUDA/LD均空。这里只确认模型初始化和首案例开始；尚无完整600周期或任务成功证据。

另已准备verify_tracking_preview_wam_development_prefetch_review.py / execute_tracking_wam_prefetch_review.py，接入前述单请求35节点并行预读，全部原全量校验/1800秒时限保持。新owner要求前次fast审查result已失败且ownership/release明确reaped，再核对旧PID/start_ticks不存活，并绑定真实raw/terminal对照证据；尚未执行或登记新审查。两脚本ruff/格式通过。当前fast审查2347553仍存活，已读到请求605附近，不能启动该后继。

## v4 fast复核实际超时关闭；v5进入原生控制

原fast复核实际达1800秒，owner unified session47123退出1，worker2347553被SIGTERM/returncode-15且release.reaped=true，kernel owner/worker均已消失。不存在actual_wait0或全量通过证据，失败产物保留。其结果及release SHA分别为{'outputs/tracking_wam_fast_review_execution_20261028/result.json': '8f0fcd6e9b8c9fe28e4c071995c18c1dbcc63bd6951b5ad53f0903d04fd54fe0', 'outputs/tracking_wam_fast_review_execution_20261028/independent/ownership/release.json': 'f4a8359b46c532dd88692012b1bcd5c8e8c0eb216db1fb0983f81435bce42459'}。

实际核对上述终态后，启动独立新登记execute_tracking_wam_prefetch_review.py，root outputs/tracking_wam_prefetch_review_execution_20261028，owner2367267/worker2367543均kernel存活，unified session26209持有实际wait句柄。要求前次失败/回收及真实raw/terminal对照记录的入口检查已通过；保持完整12/900/31500、原1800秒和所有物理/数学/任务门槛，新增物理/模型/fresh0，无自动重试。新四脚本ruff和format检查通过，当前只是启动，未有全量结果。

v5首条task0_init0_s42_raw实际生成响应成功，已保存至少40个完整原生控制周期，当前success periods0。只读取完整JSONL行，缓冲中未写出的周期不作统计；局部0成功不提前代替600周期/全12评分。首条请求35步采样和后续连续推理已实际发生，模型仍为同一服务2359094，未重初始化。

## v5首案例实际前缀对照

新增compare_tracking_wam_v5_initial_prefix.py，保存v4/v5同task0_init0_s42_raw各前56个完整JSONL周期的不可覆盖独立快照，不对仍写入的源日志作固定全文件SHA声明。实际CPU退出0，脚本ruff/格式通过，comparison.json SHA f8980bb1811d324fa89f7152c4c524db85204232f26545dac7b703d42844b825，根outputs/tracking_wam_v5_initial_prefix_20261028。完整command、measured_source、after_source、desired_tcp、opening_before/after全部最大绝对差0、变化周期0，两者成功周期均0。这只是已发生同初态轨迹前缀一致性，不是独立原生回放或新策略收益/任务资格。

v5同一模型服务已完成11请求/385 denoise，实际控制仍推进。预读复核owner2367267/worker2367543同次kernel核实存活，尚无全量结果。原开发/最终留出门槛保持，新增诊断物理/模型/fresh0。

## v4完整独立复核已关闭：全部任务评分仍失败

预读版复核unified session26209实际退出0，owner2367267/worker2367543 kernel已消失；independent/exit.json actual_wait=true/returncode0，ownership/release.json reaped=true/returncode0。完整12案例、900请求、31500 denoise/head/独立节点、六组初始当前条件配对、生产7200控制周期的日志/机械/图像/数值与全量来源前后指纹检查完成；新增时间推进/模型/fresh均0。原1800秒期限保持。新verifier schema为s2a.tracking-preview-wam-v4-development-verification-prefetch-stable-review.v1，原v4失败与两次超时记录均保留，没有覆盖或改判。

verified.json SHA c5e74dd75394dd8ec46b328a8cd1f55f20aa3893e7d08f0becb67bd23eb6cccb；owner result.json SHA df9c9225e05406eb29e1578c6a0299dc520eac97d1c25aaefb4bff8a3cf7e767；registration SHA 1c8f2a8512a07a0770088b0c2ff6cc13ec5c75280005167821bbebe9e12994fc，均在outputs/tracking_wam_prefetch_review_execution_20261028。exit/release SHA分别c7d849a7cc03fdcc943e035408c4c94d0f1a866278c5227571199a0bac93d3ab、06f19041fbf4a0bad18b96432ca5c34381c3521752c6064e0ab2f3636ab69eed。

complete/passed仅指本独立审核完整通过，任务评分明确raw ever/terminal/stable全0/6、terminal058亦全0/6，三个任务stable_by_task均0。development_stability_gate_passed=false、strict_adaptation_gain_passed=false、eligible_for_later_fresh_candidate_freeze=false、fresh_performance_authorized=false；原最终24条、至少20/24且各任务至少4/8门槛保持。总目标未达成。

v5并行生产仍为同一owner2358795/服务2359094/采集2365001，最后kernel核实存活；首task0_init0_s42_raw已保存500/600完整周期，任务成功0、抓取0，尚无完整案例评分。上述v4审核不能代替v5自身全量审核。

## v5首条完整生产关闭，第二条已启动

case00 task0_init0_s42_raw worker complete/passed，600周期、75请求；orchestration该案例exit实际wait=true/returncode0，release reaped=true/returncode0、signal_sent=false，原采集PID2365001已消失。模型仍为同一2359094，owner2358795随后bind task0_init0_s42_terminal058，新的CPU采集子进程2390116 kernel存活，没有重初始化模型或自动重试。

新增describe_tracking_wam_v5_outcome.py，复用原描述口径，并要求caseXX标签准确对应唯一案例、all_twelve准确对应完整12条；脚本ruff/format通过。实际native311执行退出0，outputs/tracking_wam_interactions_v5_20261028/case00.json SHA c09406ade1c13302cf3f8637bb1a420fe6e703349fa8ed3f812b31c660dcad71。600完整周期任务成功0、terminal/stable_last40=false、目标grasp_contact0，位置RMSE .00839736005226656m、SO3 RMSE .035552005938084386rad；actual q7 before-period最小margin .049976378386232145rad，越界周期0。这些数值与旧同案例描述一致，但这里不扩大为全state字节一致或新策略全矩阵结果。

实际opening target范围.4055059254169464至1，opening actual范围.48385209874322715至.9971162367393493；该案例不证明新低释放参考策略在低参考阶段的收益。描述器只读取已发生生产日志，物理/模型/fresh新增0，不代替v5后续独立全量审核。其余11条仍待完成，原terminal至少5/6及各任务至少1/2稳定、最终24条门槛和总目标保持。

## v5第二条出现实际释放动作干预

新增compare_tracking_wam_v5_terminal_prefix.py，冻结task0_init0_s42_terminal058各前100完整周期的v4/v5快照，明确不对持续追加源日志宣称固定全文件SHA。实际native311执行退出0、脚本ruff/格式通过，outputs/tracking_wam_v5_terminal_prefix_20261028/comparison.json SHA 5e986ae54d1e39bde4230eb47f3d69cb395fec71ec4a486b0fe3bbc79edebfdc。两份prefix及其SHA均已独立保存。

第58/59周期新release_level已为.35，而旧值分别.3254088870802089/.3146082534904684，但当时实际命令仍相同。首次command差异在0-based step82：同一desired_tcp、同一measured_source，前6个arm command逐值相同；旧gripper=-1（position_tracking_release，reference=.30320414900779724，release_level=.05312277337664404，evidence=.05s），新gripper=0（closed_motor_hold，release_level=.35，evidence=0）。这确认新阈值已经实际阻止该次开放命令，而非只改元数据。新0表示保留已闭motor target，不宣称额外力控制。

执行后pose/opening首次差异在82，执行前pose/opening首次差异在83；已生成desired_tcp首次差异在88，符合后续当前反馈重规划的时序，尚未作独立逐请求全量因果审核。前100周期两者任务成功与目标grasp_contact仍0，不能把减少释放扩大为任务收益。v5同一服务/采集进程继续运行，原600×12完整范围、门槛与fresh关闭保持。新增此描述诊断物理/模型/fresh0。

## v5第二条完整生产关闭：任务仍失败且出现实际q7越界

case01 task0_init0_s42_terminal058完整600周期/75请求，worker complete/passed；对应exit actual_wait=true/returncode0，release reaped=true/returncode0、signal_sent=false，采集2390116已消失。same owner2358795/模型2359094随后bind第三条task0_init0_s43_raw，采集2402406 kernel存活。总模型初始化仍沿用一次；尚无全12或v5独立全量验收。

实际执行describe_tracking_wam_v5_outcome.py --case-indices1 --run-idcase01退出0，outputs/tracking_wam_interactions_v5_20261028/case01.json SHA 957babae655485ae25298c821466b92baa6ef4ccd3f8611ba71de45b2f898921。600周期任务成功0、terminal/stable_last40=false、目标grasp_contact0；position RMSE .007055320146371085m，SO3 RMSE .032100187309319884rad。旧同案例为.005433148005090785m/.020399859473554452rad，任务/抓取亦0；新完整反馈轨迹并非旧固定参考，不把这些差值解释为单步控制器因果收益。

新案例before-period q7越界197周期，首个0-based step389，最小margin -.0030810727922534475rad；旧同案例越界0、最小margin .34371143099013013rad。routes unchanged_free_feasible368/interior_witness15/adaptive_margin_witness18/preventive_fallback197/preventive_fallback_retained_seed2。arm跟踪误差较小不能代替关节可靠性或任务成功，模型计划的.05 buffer没有保证这些实际端点不越界。

追加audit_tracking_v5_case01_endpoints.py实际CPU核对全部601保存state、600当前输入/标签/速度/motor target/pose与三个图像绑定，退出0，ruff/format通过。由XML独立按joint类型建立qpos地址，直接读取601 sim_states的q7，并要求600 log的joint position逐值在1e-12内相符；原生模型范围[-2.8973,2.8973]rad，实际198个端点越界（含最终端点）、before-period197，首次端点389，最小margin同为-.0030810727922534475。所有来源前后重读一致；证据outputs/tracking_v5_case01_endpoints_20261028/verified.json SHA c88afaec29eaefb80eeed85849e463cdce52aa60836ac4f6b4c0bee9ab826fcf。complete/passed指数据核对完成，明确不是安全边界或任务通过。新增时间推进/模型/fresh0，v5其余10条生产继续原完整范围，最终留出仍关闭，总目标未达成。

## v5 case01边界退化的最早观测位置

对已关闭且601保存状态核实过的case01逐周期读取XML七关节范围及实际q，首次任意关节实测margin<.05在step383。step382 interior_witness记录的模型最小inner margin .04999999999999227，下一真实端点margin .04999853998412451（q7），约1.46e-6rad低于请求buffer。之后step383进入adaptive_margin_witness，step384实际q7 margin降至.03439180695743005，step385为.021017781210610043；step387转preventive_fallback。step388 q7余量.0005193057360548536、实际速度+.7575126062876212rad/s，step389 q7=2.9003810727922534、越界.0030810727922534475rad。

interior/tracking terminal求解器对initial margin<.05直接返回initial_outside_requested_interior，adaptive margin允许按当前余量保持较低模型margin，fallback的accepted并不要求requested_margin_satisfied或terminal_constraint_enforced。实际step389 accepted=true而optimizer_success=false、requested_margin_satisfied=false，不能把选择记录当作硬边界通过。后半段长计算亦与fallback反复搜索相符，例如step387/389 tracking_refinement.compute_seconds分别12.498/16.764秒；这是运行诊断，不是实时性证明。

这定位了微小模型偏差跨过buffer后策略切换、再到实际XML越界的观测链，尚未证明单独取消该切换即可修复。下一独立候选应在相同已捕获状态/历史/生成参考与命令box/rate下检查可执行的buffer回入/制动，并实际原生回放验证，不能把初始.0499985直接当作满足.05或放宽原验收；不热改正在运行的v5。任务抓取失败早于关节退化，不能声称解决该边界问题就会完成任务。原矩阵与最终门槛保持。

## 捕获step383的buffer回入数值探针

新增future_interior_reentry.py，复制冻结tracking_terminal_joint_plan的原目标、命令box/rate、全部25×h未来inner .05rad约束与terminal方向约束；仅将初始状态buffer外的直接拒绝改为要求仍在XML范围内后尝试求解。明确initial_inside_requested_buffer=false时不追认初始状态满足.05，不热改v5，不放宽未来约束或数值容差。

首次probe_tracking_v5_buffer_reentry.py错误地对生产硬约束分支传入warm start，step382原结果复现差4.67792875e-5而触发1e-7断言；失败脚本/registration/result均保持，未进入候选资格。检查实际tracking_margin_hybrid硬分支为solve_interior_plan(*args)不使用warm start。新probe_tracking_v5_buffer_reentry_v2.py按实际seed起点和CPU OMP2/OPENBLAS1/LP2执行，实际session92559退出0，无时间轴action/step或模型调用（仅native环境初始化以取实际controller.initial_joint）。两新增源及脚本ruff/格式通过。

v2结果outputs/tracking_v5_buffer_reentry_v2_20261028/result.json SHA 0483f80154e2ec9d78d425e651722be6d56a8e7bf363a8c98cb97f7f05409480。step382 baseline原计划在1e-7内复现，baseline/candidate均61响应、未来最小margin .0499999999999603；step383实际初始margin .04999853998412451，原分支按initial_outside_requested_interior拒绝，新分支58响应找到model_witness，未来最小inner margin .04999999999999449，command变化率保持<=.25，terminal约束保持。对所选计划另行重算全部未来inner点；该重算共享冻结前向模型，不是独立物理资格或任务成功。

此结果表明原始初始buffer早退会排除至少一个后续全部inner满足原buffer的局部模型解；并未证明该模型解在非线性原生机器人中也能回入或避免后续越界。下一步是相同捕获状态/已有参考下的原生实际命令对照及独立回放，然后才考虑完整原18控制资格。v5第三条仍由原模型2359094/采集2402406执行，最后实际端点80，总矩阵和最终留出门槛保持。

## step383回入解的原生命令对照与独立回放关闭

新增probe_tracking_v5_reentry_native.py及持久owner，登记两条各383前缀+1动作、总768物理周期/模型0/fresh0。首次基线endpoint1状态SHA失败，只执行1周期、候选未执行；owner session92538失败并回收，原失败记录保留。其入口漏掉生产初始化时的current_observation读取；成功历史回放入口也明确有该调用。新v2仅补齐初始化读取，不放宽状态SHA条件，独立root tracking_v5_reentry_native_v2_20261028与execution_v2登记。

v2实际session61409退出0，owner/capture exit actual_wait=true/returncode0、release reaped=true/returncode0、无signal。两条分支各完整384动作/385保存state，分支在endpoint0至383均与原生产SHA精确相同，baseline endpoint384亦精确相同；仅候选step383的前6轴改为已登记局部回入解，gripper及所有前缀命令保持原值。result.json SHA ccd62c54d67bae506fa5933de18bd878652e8b81ebff286ad23a1d6b09c03951。原动作后最小实际joint margin .03439180695743005rad，候选后 .050027415687298404rad（q7 .053670406314610286、q5 .050027415687298404）。这是一次实际端点回入，并不证明全部inner2ms或多周期保持。

同一已到达参考下，原动作后TCP position/rotation误差范数 .011115766625308193m/.09837441847379753rad，候选 .01095134250366563m/.11811624650844432rad，旋转误差更大；不把关节端点回入当作任务收益或整体胜出，两者原生任务仍失败。

追加check_tracking_v5_reentry_native.py及独立持久owner，另行登记768物理周期/模型0/fresh0，仅消费两个保存命令序列，不导入候选求解器；两次全384命令/385状态原生重放及末端joint margin/任务判据核对完成，两个分支maximum_state_difference均0。实际session4956退出0，exit actual_wait=true/returncode0、release reaped=true/returncode0，verified.json SHA f43b12e863c8127218be2b2447d4371828a7aab9d983b1fa770265545318ec88，根outputs/tracking_v5_reentry_native_check_20261028；两个owner和两版本producer/checker脚本ruff/format通过。此组总物理周期1失败+768生产+768独立回放=1537，自动重试0。

回入候选尚未接入v5生产、没有多周期/原18资格或新WAM/fresh任务资格；下一步需在同一已生成参考下验证持续反馈回入及跟踪代价，再满足原18完整控制保持门槛。v5第三条仍由原owner/服务继续，原总目标、完整开发矩阵与最终留出门槛保持。

## 33周期同参考反馈回入对照已登记；中断后的持久恢复

新增reentry_tracking_margin_hybrid.py/native.py，仅把原hard分支替换为future_interior_reentry的未来inner .05回入搜索，正常分支/自适应/fallback原顺序保持，并记录initial_inside_requested_buffer；未改冻结v5。新增probe_tracking_v5_reentry_continuation.py：两分支同一完整383周期前缀、相同已在step376生成的chunk47参考，cursor7至39共33周期反馈执行，终点416，832物理周期/模型0/fresh0。基线用原跟踪控制，候选用回入分支；两个分支保持同一release-floor夹爪逻辑，并在step383先复现已核验单步动作。只读观察continuation内每次native sim.step之后q/time，要求每周期25条、相邻.002s，不增加forward或step；记录全部inner数据与实际末端状态。四新源ruff/format通过。

首执行owner2416405/worker2416792在对话中断后kernel均消失，unified handle10661亦不存在；旧根仅有registration/launch/pending及prefix anchors，没有result、exit或release，不能声明实际wait0、回收状态或完整物理预算。执行了多少周期未知，只能绑定原832上界。已以独立interruption_observation.json记录实际kernel/句柄缺失及缺失退出证据，原目录不覆盖。

恢复版本probe/execute_tracking_v5_reentry_continuation_v2.py使用独立root tracking_v5_reentry_continuation_v2_20261028与execution_v2；原33/832范围及算法保持。新owner通过durable_stage_supervisor detach/ready握手启动，SIGINT/TERM转受控异常以便owned worker实际清理；来源包含当前owner/probe/回入三模块并绑定原中断观测。实际launcher session94549退出0，launcher_result SHA 06aa41aa17e3cd034659f4ff037629353195b078efdb22075878a50f1ec9275a，stage完成不在其声明范围。新owner2421275已PPID1，worker2421295 kernel存活/CPU时间推进，尚无完整结果。v2两入口ruff/format通过，未修改旧冻结文件。

单步原生产物的tcp_error调用为pose_error(actual,desired)，多周期入口为标准pose_error(desired,actual)；此前单步报告只使用位置/旋转范数，对参数交换不变，没有据该向量作方向补偿。后续跨产物比较需以范数或显式统一方向，不能直接混用向量。

v5第三条同一生产owner/模型/采集继续存活。总目标、原18控制保持/完整12开发/最终24留出门槛不变，当前回入策略尚无多周期或任务资格。

## 多周期回入的全端点独立检查入口已准备

新增check_tracking_v5_reentry_continuation.py及持久execute入口：只在原生producer owner complete/passed、actual_wait=true/returncode0、reaped=true/returncode0后，消费两分支实际保存命令，独立重放各416周期/417状态；检查全部825个continuation内环q/time、原命令box/rate、cursor7至39的同一已生成参考SITE绑定、TCP误差、原生任务与grasp判据，汇总实际XML越界/.05buffer违例及TCP RMSE。检查器不导入生产或回入求解器，拟登记独立832物理周期/模型0/fresh0。尚未执行该检查，不存在其通过记录。两个新入口ruff/format通过，参考变换已用旧捕获step383四目标对照，逐值差0。

同次kernel核实持久多周期owner2421275/worker2421295仍存活，CPU时间推进，首分支尚无完整输出。v5生产模型2359094与第三条采集2402406仍存活。原目标和全部门槛保持，等待实际生产关闭后再启动独立回放；不因观察等待而重复启动。

## 33周期回入对照与全内环独立回放关闭：候选不采用

持久v2 producer完整832周期完成，capture实际wait=true/returncode0、release reaped=true/returncode0，owner2421275/worker2421295均已退出。两条383周期原SHA前缀成立，step383原seed与单步动作复现检查通过；之后同一step376已生成chunk47的cursor7至39，各反馈33周期并记录825个native2ms端点。producer result SHA e637dd855aea296d9f451bc2133deb7df50e201692c9704e89d5dccb6ea358e6，根outputs/tracking_v5_reentry_continuation_v2_20261028。

全量独立checker由持久owner2429604/worker2429622执行，另832物理周期，全部两分支417状态和825内环q/time重现，maximum_state_difference及maximum_inner_joint_difference均0；同一参考SITE/cursor、box/rate、实际TCP误差、task/grasp判据亦核对完成。checker实际wait=true/returncode0、release reaped=true/returncode0，owner/worker已退出。verified.json SHA 5c97a0b12f14534cbcd2cd433992cf70e08fb4492585944bd44afcf266e50f01，根outputs/tracking_v5_reentry_continuation_check_20261028；checker owner result SHA 8a20e532054dd66f374eddf9c77ccc05b66dbcd763d2448dac1997e5dee1708b。所有新入口ruff/format检查通过；complete/passed指完整实验及重现核验，不是性能达标。

| 原同参考33周期指标 | original | reentry |
| --- | --- | --- |
| 最小实际inner joint margin(rad) | -.0051040876089527565 | -.007112439843481422 |
| XML越界inner样本/825 | 775 | 599 |
| 低于请求.05 buffer样本/825 | 825 | 669 |
| 首次XML越界(step,0-based inner) | (385,0) | (392,1) |
| TCP位置RMSE(m) | .03028219286671087 | .02963888904646734 |
| SO3 RMSE(rad) | .3525074004731379 | .38869922633929244 |
| 任务成功/抓取周期 | 0/0 | 0/0 |

候选延后首次越界、减少越界内环样本，但最深越界与旋转RMSE更差，未保持原buffer、更未保持XML边界或产生任务成功；不采用、不进入原18资格或新WAM/fresh。单步回入并未转化为持续可靠控制。候选step383至386实际每周期inner最小margin均>.05；step387在hard模型最小margin .04999999999935145时，真实最小已为.049990478670450145；step388仍hard witness而实际.0499928109206893。说明放开初始buffer早退不足以处理后续冻结M/J模型误差与不可行时fallback，不将该数值改成通过容差。

有效多周期生产+独立回放共1664时间推进、模型0/fresh0；首中断执行实际推进未知，仍仅有原832上界，不能把其计为已完成或0。旧单步阶段的1537预算记录保持，不重复计入本组。下一步若继续改进，应以这些真实joint预测残差/不可行状态为依据，保持跟踪/任务和原完整门槛，不凭局部单步正结果扩大部署。冻结v5第三条生产仍继续，最终留出保持关闭，总目标未达成。

## 全33周期关节响应残差与因果经验包络核对

新增audit/execute_tracking_v5_joint_response_residual.py，消费已完整独立重现的两个33周期分支，在原XML保存状态上mj_forward重算实际M/J/q/qd，逐周期对照源日志（M/J 1e-9、q/qd 1e-12），用实际固定controller.initial_joint和已执行命令重算自由/理想硬限位两种冻结模型的全部25点预测。无时间轴step、模型调用或fresh；不是新模型/控制资格。

首次OSMesa导入失败，未进入物理推进，原registration/owner失败保持；新v2入口在MuJoCo导入前取消MUJOCO_GL/PYOPENGL_PLATFORM强制值，只加载XML并计算动力学，不创建renderer/context。持久v2 owner实际wait=true/returncode0、release reaped=true/returncode0关闭，所有新增脚本ruff/格式通过。outputs/tracking_v5_joint_response_residual_v2_20261028/verified.json SHA da01404b2bfbd994d8c950e80b8ed46abb2fc7f1678f0c8ddc01b772d4640f5b，逐周期全部预测/实际内环/历史误差包络已保存。

回入分支step387自由模型预测最小margin .05000000000000604，实际.049990478670450145，显示约9.52e-6rad的物理模型偏差，不通过更改数值容差掩盖。原分支step385自由预测margin -.03658365775950401，理想硬限位预测0，而实际-.0051040876089527565；之后fallback同样使用理想限制掩盖真实soft-limit穿透。例如原step387自由预测-.05512389360020098、理想预测0、真实-.0008336370082426825。因此既不能把自由模型所有误差说成乐观，也不能把理想投影位置解释成真实关节约束成立。

按每关节此前已完成周期的自由模型绝对残差最大值构造包络，首周期不设可用bound；严格先评价当前误差、再更新running max，无当前/未来误差进入当前bound。原/回入分支在随后32×25×7=5600坐标中覆盖5421(96.8036%)/5200(92.8571%)，均未完全覆盖。该总体坐标比例还可能被非临界关节主导，不能当作关节边界保证或新控制收益。最终q7最大自由残差分别.07651193309770754/.07358884242954788rad，包含进入soft-limit后的失配，不能拿事后最大值冒充越界前可用先验。

下一控制改进需在接近边界前同时处理位置余量、向外速度和真实模型误差/不可行状态，而非仅放开初始buffer早退或信任理想硬投影；任何候选仍需原生多周期及原18/完整WAM/最终留出资格。当前v5生产不改动，总目标保持未完成。

## 位置—速度制动候选的数值检查与原生对照已启动

新增velocity_braking_joint_plan.py，保留原全部inner位置.05rad、命令box/rate与终端方向约束，追加所有控制周期端点的q+tau*v位于相同.05内区约束，tau固定.1s（两个20Hz周期，执行前确定），不是事后调参或动力学制动距离保证。q/v及其对命令导数来自同一冻结M/J模型，最终解再次核对全部叠加线性约束与rollout；没有放宽原位置容差或将模型保证说成物理保证。

probe/execute_tracking_v5_velocity_braking.py在已捕获step368/375/382执行零时间推进数值搜索（仅native环境初始化取实际initial_joint），持久owner实际wait=true/returncode0、release reaped=true/returncode0。step382原计划在1e-7内复现通过；三个新增制动模型witness分别71/52/58响应，最小inner位置margin .10200429322426352/.05084646744928367/.050000000000000266，最小位置+速度margin均约.05（原数值容差内）。数值结果SHA b989a9fe2c98a984058caceaade8c203f5eda5dd191c2ff910ecfb665c0f76ea，根outputs/tracking_v5_velocity_braking_numeric_20261028；数值模块及两入口ruff/格式通过。该阶段模型/fresh/时间轴推进0，不是候选物理资格。

新增velocity_braking_tracking_hybrid.py/native.py，将新增制动风险也作为需要优化的触发条件，hard失败后原adaptive/fallback顺序保持。新增probe/execute_tracking_v5_velocity_braking_continuation.py：两分支同一368周期精确原SHA前缀、同一step368已生成chunk46全部40参考（cursor0..39），分别原控制与固定.1s制动控制反馈执行40周期，终点408，各记录1000个实际2ms样本；总816物理周期、世界模型/fresh0。入口先要求当前step368原seed与数值候选第一动作复现，保持原rate/box，第一动作之后才继续多周期。不是把分叉后的旧WAM新图像参考当作可用输入。新增四文件ruff/format通过，尚无物理结果。

持久launcher实际退出0，launcher_result SHA b03466495ad8cd1f182fa0d69839d9d1f9d2c7765f54bcd6f80c8ab89f4f2f9d；owner2437611已PPID1、worker2437631 kernel存活且CPU时间推进，根outputs/tracking_v5_velocity_braking_continuation_execution_20261028。原v5模型/第三条采集继续运行，冻结生产源码不改动。后续必须独立全40周期/全内环回放，再判断是否值得原18资格；完整WAM与最终新初态门槛保持，不提前采用。

## 40周期制动对照的独立回放入口已准备

新增check/execute_tracking_v5_velocity_braking_continuation.py，保持独立检查不导入候选或生产求解器，只消费保存命令。入口要求producer完整816周期且持久owner实际等待并正常回收子进程（退出码0）后，再独立重放两分支各408周期/409状态、40×25=1000实际inner q/time，核对同一step368已生成chunk46参考的cursor0..39、command box/rate、TCP误差、任务/grasp及实际关节余量。拟追加独立816物理周期/模型0/fresh0，当前尚未启动或产生其通过记录。两个新文件ruff/format通过，所有范围/索引/数组形状已按40周期重新检查。

本轮kernel核实制动对照owner2437611/worker2437631仍存活且CPU时间增加，首分支尚未完整输出；v5原模型2359094和第三条采集2402406亦存活。不因局部数值witness或等待状态把候选提前接入生产，原总目标及全部资格门槛保持。

## 40周期制动对照与独立回放完整关闭：不采用

producer和独立checker均actual_wait=true/returncode0，release reaped=true/returncode0；各816实际物理周期，合计1632，世界模型调用及fresh均0。每分支409状态和1000内环关节样本独立重放最大差均0；参考固定为step368已生成chunk46，不使用分叉后旧日志的新参考。producer result SHA 365db158a805a78b8e89ab12eb97dd9cd1941aa2486aad772ae0326e6620733d；独立verified SHA 0f81a364b1d2808ba83e054ffa9223f28c2313699409a15831917cb2a4afa3c4；checker owner result SHA 049a872abc12cc0a6b6d96de35c002812b5f9af9eae0a12acf08b0c416329e26。根分别outputs/tracking_v5_velocity_braking_continuation_20261028、tracking_v5_velocity_braking_continuation_check_20261028及对应_execution_20261028。

|40周期指标|原分支|固定tau=.1s制动|
|---|---:|---:|
|最小实际inner关节margin(rad)|-0.007026965727407575|-0.009706095861528485|
|XML越界inner样本/1000|428|404|
|低于.05rad buffer样本/1000|486|457|
|TCP位置RMSE(m)|0.027628557791383374|0.02427186216518727|
|SO(3) RMSE(rad)|0.6990562026899628|0.7239108107993163|
|任务成功/抓取周期|0/0|0/0|

首个低于buffer内环(step,零基inner)原分支(384,2)、候选(389,18)；首XML越界分别(390,22)/(391,7)。原分支step389由hard转adaptive，390进入fallback；候选分别390/391。候选延后失守但最深越界与旋转误差更差，不采用，不进入原18或新WAM/fresh资格。complete/passed只表示执行和独立回放一致，不表示控制资格通过。下一步应核对首个hard保持期间的模型残差及之后不可行/fallback机制，不能仅凭位置RMSE下降继续调tau。v5冻结源码和生产保持，总目标及最终24新初态门槛未达成。

## 制动分支首次buffer失守发生于hard模型仍可行期间

新增audit/execute_tracking_v5_braking_response_residual.py，复用已验证的XML-only CPU动力学重建，对两个完整40周期、每周期25实际inner位置核对自由/理想限位模型；全部M/J/q/qd核对通过，未推进时间轴、模型/fresh0。持久owner actual_wait=true/returncode0、release reaped=true/returncode0。outputs/tracking_v5_braking_response_residual_20261028/verified.json SHA 32e5223362b9d230ce8ec0becca415871cb2af83cc13ba6628060ab96899abc0；两新增入口ruff/format通过。

原分支step384仍interior_witness/optimizer_success/accepted，首周期预测最小margin .05，实际.04999783981337513；其最小处(inner4,q5)残差4.690040901422776e-6rad，此前完成周期同关节最大残差.002123565566618524。制动分支step389同样仍hard可行，首周期预测最小margin .050062669015038974，实际.04737276932522727；最小处(inner24,q7)实际2.8499272306747727、预测2.84712434061962，残差.0028028900551526625rad，此前最大.004158746379582023，能够覆盖该次误差。这里的失守先于adaptive/fallback，不应全部归因于后续fallback。原/制动全局过去最大残差包络覆盖6488/6825和6505/6825坐标，仍非完全覆盖，也非未来保证。

下一候选应显式将因果历史的逐关节模型偏差纳入预测约束，并在收紧后不可行时明确报告；不能用事后全轨迹最大误差、放宽容差或仅增加固定制动tau。该方向尚未实现/执行/取得资格，冻结v5不改动，总目标未完成。

## 因果误差收紧候选实现，尚未进入物理对照

新增causal_joint_error.py及past_error_joint_plan.py，严格完成后更新逐关节位置残差；额外位置约束为.05+过去最大误差，未以位置误差冒充速度误差。保留原XML、命令box/rate、终端及固定.1s制动限制，不可行显式返回。native311 CPU两项实际测试通过(3.73s)，ruff/format通过；未执行捕获状态数值对照/物理反馈/18/WAM/fresh，不能称控制改进。后续合同见plans/v0.0.1/causal_joint_error_20261028.md。

## 因果误差约束真实捕获状态数值检查完成：边界前已不可行

新增probe/execute_tracking_v5_causal_error_numeric.py；在已独立精确回放的两个分支各40状态上重建M/J/q/qd，分别在step368/375/382/384/389搜索名义制动和历史误差收紧计划。每分支误差状态独立重建，逐周期先读取snapshot用于搜索，再stage当前预测并用已完成实际内环更新；step368无历史，不从另一分支或未来最大误差初始化。所有参考只用step368已生成chunk46。本实验是旧分支状态上的反事实数值搜索，不能当作候选自身闭环轨迹。

两分支step368零历史精确保留名义计划，step375各7个过去周期后收紧仍有witness，逐一核对全部inner约束slack在1e-10原容差内。两分支step382各14个过去周期时，名义制动仍model_witness，收紧后HiGHS status2（当前冻结线性约束无可行解）；step384同样。step389原分支初始已在.05外，制动分支收紧LP仍status2。没有放宽约束或把no_witness替换成成功。

owner actual_wait=true/returncode0、release reaped=true/returncode0；时间轴推进/世界模型/fresh均0，两个新脚本ruff/format通过。outputs/tracking_v5_causal_error_numeric_20261028/verified.json SHA 9b2c198826b4129a954666ad76a8484c257cf81ccdfd3b85154e58b86dbd4197，原始逐状态计划/历史/无解状态保存在各branch/searches.json。不能直接接入原adaptive/fallback并宣称稳健：它会在已发现风险时丢弃收紧。下一步需要区分位置约束、制动、终端和命令速率对不可行的贡献，设计提前恢复策略后再进行候选自身反馈物理对照；尚未进入18/WAM/fresh，总目标未达成。

## 收紧无解的约束归因：位置本身不可行，必须提前干预

新增joint_constraint_diagnostic.py及probe/execute_tracking_v5_causal_constraint_diagnostic.py，在相同两分支5个捕获状态分离position/braking/rate/terminal，box始终保留。全部LP状态与前一阶段独立执行的求解器一致（仅比较有linear_status者，不把初始guard返回当作LP结果）。step382/384/389两分支的position_only均status2；单独移除braking、rate或terminal仍status2，移除position后可行。不能把无解归因于制动tau或终端方向过严，也不能通过仅放宽命令变化率解决当前冻结模型的位置限制。

另外移除全部未来周期约束、制动和终端方向，保留box/rate，最大化第一周期25个inner点减逐关节历史误差后的最小XML余量。原/制动分支step382最佳余量分别.04693911138773308/.04935012423220481rad，均低于.05；step375分别.11286810446278328/.1120948969224504。说明在step382才开始该收紧已迟于当前模型的第一周期可恢复范围；不能把这种冻结模型结论扩大为真实机器人不可恢复证明。诊断LP命令明确标记executable_plan=false，不执行任何移除约束的计划。

native311 CPU解析不可控关节fixture测试通过(3.04s)：第7关节无法由六维接口移动时，误差调整后的最大余量精确.0475rad，position无解且移除后可行。新增四文件ruff/format通过。owner actual_wait=true/returncode0、release reaped=true/returncode0；physics/model/fresh0。outputs/tracking_v5_causal_constraint_diagnostic_20261028/verified.json SHA c07d15eace9ed2718b3cee241f2785a5096ba5093cc466af9a477a4c4764705b。下一步从已可行的更早状态开始候选自身反馈对照，用同一分支真实过去误差逐周期更新；收紧无解必须记录并停止候选，不能静默切回丢弃误差的旧fallback。最终判定仍需真实跟踪和任务成功，全部18/WAM/fresh门槛保持。

## 因果误差提前介入原生反馈对照已启动

新增causal_error_tracking_hybrid.py/native.py与probe/execute_tracking_v5_causal_error_continuation.py。候选从原step368可行状态、空误差历史开始，每周期强制检查past_error_joint_plan完整收紧约束；stage实际选定命令的自由模型25个inner预测，完成真实25步后才更新自身残差。原命令box/rate、物理范围、固定.1s制动和终端方向保持。hard无witness抛NoCausalWitness，由实验明确保存当前状态、历史和诊断，停止前不发送该周期action，不进入adaptive/fallback；这是候选未完成的记录，不是可运行系统的安全恢复保证。未支持的未来rotation latch直接作为执行错误报告。

原/候选各精确复现368原SHA前缀，同一已生成chunk46最多40反馈周期；首候选命令要求重现零误差制动数值witness。正常完成40或显式no-witness提前结束均保存实际states/inner/动作/终点和计数，all40_success要求确实执行40；总物理预算上限816、model/fresh0。候选独立评估重新建立误差历史，绝不读另一分支未来误差。

5项native311 CPU测试通过(6.31s)，含原有因果误差/约束归因及新增执行适配器真实可行/无解测试，确保无解不返回放宽命令；新增五文件ruff/format通过。durable launcher退出0，SHA 514dbfaccbb027571bb607daeb05ce9a6e3662947d24933253717aa4d227e729；kernel owner2453713已PPID1、worker2453721存活。根outputs/tracking_v5_causal_error_continuation_execution_20261028，尚无完整物理结果；后续独立回放必须按实际执行长度检查并区分正常40与候选no-witness停止，不能将owner成功误写成控制资格。原v5 owner2358795/service2359094/第三条2402406仍存活且CPU推进，冻结生产不改动。原总目标与18/WAM/fresh全部验收门槛保持。

## 实际执行长度回放与因果历史独立核对入口已准备

新增check/execute_tracking_v5_causal_error_continuation_check.py：等待producer实际wait0/reaped0后，按实际continuation_steps重放两分支全部命令与状态/inner，核对精确参考cursor、原box/rate、TCP/任务/grasp。原分支必须40，候选允许1..40，若不足40必须有step=368+n的no-witness且action_dispatched=false，终点包含在states中；不能把提前停止算完成40。独立物理预算按producer实际数核对，上限816。

另新增audit/execute_tracking_v5_causal_history.py，在物理独立复核正常关闭后，XML mj_forward重建每个实际周期的M/J/q/qd；不导入候选求解器或其误差状态类，以独立running max逐周期核对candidate before/after及停止history、实际发送第一命令与计划绑定、完整计划全部inner位置的.05+当时历史误差限制及原.1s制动约束。首周期history必须不可用，使用完before才更新当前残差。该检查不独立证明停止状态无解，也不证明未来误差包络或任务成功；时间推进/model/fresh0。四新增文件ruff/format通过，尚未启动或产生通过结果。

本轮kernel核实producer owner2453713/worker2453721持续运行且CPU时间增加（约2分钟），尚未关闭原分支；原v5第三条2402406亦存活并推进。不因观测等待重启或改写已登记源码。总目标与所有资格门槛保持。

## 因果误差提前介入完整关闭：保持22周期后无解停止，未完成控制资格

producer完整退出0并actual_wait=true，release reaped=true/returncode0。原分支368+40，候选368+22，共798真实周期；候选step390 hard返回linear_solver_no_witness，保存当前状态及历史且action_dispatched=false，未执行该周期命令。producer result SHA b93ae56230e26c530ac9091fcb0cd76001ec8e552b87c28e57308a3777c1fb24。并非完整40周期候选，也没有可继续运行的恢复控制。

独立物理回放追加798周期，两分支全部实际状态及内环最大差均0，参考/命令box/rate/TCP/任务/grasp与停止位置通过。verified SHA 7e1ffcfaf0c0fa59569dc344be92bef395d7d6528e42291f707de8df6541d93c，根outputs/tracking_v5_causal_error_continuation_check_20261028；owner实际wait0/reaped=true/returncode0。两次合计1596物理周期，model/fresh0。

|相同step368..389共22周期|原分支|因果误差候选|
|---|---:|---:|
|最小实际inner margin(rad)|.035358878255276505|.05066845294205802|
|XML越界inner样本/550|0|0|
|低于.05buffer样本/550|36|0|
|位置RMSE(m)|.018840732434290373|.01913878621074185|
|SO(3) RMSE(rad)|.2546682080910637|.25880203732956497|
|任务/抓取成功周期|0/0|0/0|

原分支完整40最小margin -.007026965727407575、XML越界428/1000，不能与候选短22周期直接比较来宣称跟踪改善。相同22周期上候选余量更好，但位置/旋转误差略差且后续停止，不能采用为完整控制器，不进入18/WAM/fresh。

独立因果历史审计也actual_wait=true/returncode0、release reaped=true/returncode0：outputs/tracking_v5_causal_history_audit_20261028/verified.json SHA ed658265b323aea4088cb6de847b5daad1af18e928a467cf0dd958c9e4bd535d。XML重建每周期M/J/q/qd，独立running max逐一确认before仅含此前完成同分支周期、after按当前真实误差更新、首次不可用、停止历史匹配；核对实际首命令和完整已执行计划全部收紧inner位置及原制动限制。该审计未独立重证停止状态LP无解。候选过去误差包络覆盖3547/3675后续坐标，仍非全覆盖/未来保证。总目标仍未完成，冻结v5继续运行。

## 因果inner时间误差分布：数值可行性恢复，原生反馈对照启动

已保存22周期中，q7历史2ms误差最大.000029230222667209205，50ms/peak.004158755838823858rad；旧peak法将后者从第一个2ms点施加。新增causal_inner_error.py/past_inner_error_joint_plan.py：独立已完成周期累积25×7误差并沿inner时间取前缀最大，仅第一规划周期使用时间profile，后续周期保持原peak收紧。原物理范围/box/rate/制动/终端/初始buffer全部保持，无未来误差保证。

四项native311 CPU测试通过（核心2项2.47s，执行适配器2项3.68s），含常量profile精确复现旧计划、未来周期仍用peak、防观测泄漏和无解不放宽；10新增文件ruff/format通过。数值probe/execute_tracking_v5_causal_inner_numeric.py owner实际wait0/reaped=true/returncode0，verified SHA 108d4659c9cc05de889b7cdbe56984f4254db6c748434501b84b220dcd64c231：两分支step382/384及制动389均恢复model_witness，原389仍初始buffer外拒绝。physics/model/fresh0，不宣称物理收益。

新增causal_inner_tracking_hybrid/native及原生probe/owner，保持同368前缀+chunk46、最多40反馈周期、上限816物理周期；候选只用自身完成周期更新profile，no-witness保存现场且当前动作不发送，实际长度明确记录。持久launcher退出0 SHA deaae0cec2cae3869a97fbce9d49441560a243139378ffcdc9d1fc1ed290b68c，kernel owner2464234 PPID1/worker2464311存活。尚无生产或独立检查结果，后续合同见plans/v0.0.1/causal_inner_error_20261028.md。冻结v5与原18/WAM/fresh门槛保持，总目标未完成。

## inner时间分布原生对照与两独立复核完整关闭：仍22周期后停止

新增实际长度回放check/execute_tracking_v5_causal_inner_continuation_check.py及独立audit/execute_tracking_v5_causal_inner_history.py，四文件ruff/format通过。producer及两个独立owner均actual_wait=true/returncode0、release reaped=true/returncode0。producer真实798周期，独立回放追加798，合计1596；世界模型/fresh0。

候选同样只执行step368..389的22反馈周期，step390返回linear_solver_no_witness并在派发前停止，completed_all40=false；原分支完整40。producer result SHA 5eceefac1c31f19d4bddb1d1c6a736b5711253d7847b9756c6e3c366c1de6697。候选550内环样本最小实际margin .050376941208823744rad，XML越界及低于.05buffer均0；任务/抓取0。相同22周期原分支最小margin .035358878255276505rad、低于buffer36/550；位置RMSE原.018840732434290373/候选.019117297664263063m，SO(3)原.2546682080910637/候选.2588029353522851rad。候选跟踪误差仍略差；不能与原完整40周期较大误差直接比较。相较旧peak候选未延长实际可执行段，不能用旧捕获状态数值可行性恢复替代真实反馈结论，不作为完整控制器采用。

物理独立verified SHA 7d1eff16798a25608d44a8ecc98824ed16ff3617a7f6d79b08e3e32695f47d89（outputs/tracking_v5_causal_inner_continuation_check_20261028），两分支全部实际状态及inner关节最大差0，停止位置/长度、任务计数、full40成功条件与命令/参考核对通过。

history独立verified SHA 028d218effd9e1436dcbcbbc69637c237e8546dfc01d12c219e5d5cf7b2204ba（outputs/tracking_v5_causal_inner_history_audit_20261028）：XML独立重建M/J/q/qd，逐周期不使用候选history类重建25×7过去残差profile及peak，核对before/after/停止历史；完整计划第一周期用profile、之后周期用peak的所有inner位置约束及原制动约束保持。没有重证停止状态LP无解或证明未来包络保证。下一步应直接诊断候选真实停止状态第一周期与未来周期的可恢复性，不仅在旧分支捕获状态搜索新witness。原18/WAM/fresh资格均未进入，总目标保持未完成，冻结v5继续。

## 自身停止状态诊断：第一周期已不可恢复，速度模型在此前发生突变

新增inner_constraint_diagnostic.py与audit/execute_tracking_v5_stopped_state.py，以两版候选各自已独立复核的最终step390保存状态重建当前M/J/q/qd；1/8周期分别检查位置/制动/rate/terminal，并最大化第一周期误差调整余量。使用重复上一实际命令作为仿射展开seed，不冒充生产原seed；position-only/第一周期边界结论只在冻结模型内成立，诊断命令不执行。两版在1与8周期中position_only均status2，分别去除制动、rate、terminal仍无解；第一周期box/rate下最大误差调整余量peak .04507565611985814、inner-profile .048266584020483255rad，均低于.05。并非仅远期约束造成停止，step390已晚于当前模型的可恢复区。inner候选q7=2.8469230587911762rad、向外qd7=.2765894764808469rad/s。

解析不可控关节fixture native311 CPU测试通过(3.63s)，四新增文件ruff/format通过。owner actual_wait=true/returncode0、release reaped=true/returncode0，physics/model/fresh0。outputs/tracking_v5_stopped_state_diagnostic_20261028/verified.json SHA ef5d59b74cc202c05047331709905cc12a0dcaa6bce7baef02cd282d9013e619。

新增audit/execute_tracking_v5_endpoint_velocity.py，从独立重放的端点qvel重建每周期真实速度，核对自由模型第一周期速度与q+.1v响应；每周期先记录过去最大位置/速度残差，后更新当前。owner亦actual_wait0/reaped=true/returncode0，ruff/format通过，physics/model/fresh0。verified SHA 43cce7ff027d106a9a99b1b95e723ba91e5ca53d711b11808e0aefa1e1da083c，根outputs/tracking_v5_endpoint_velocity_audit_20261028。

inner候选step389是首个真实q+.1v margin低于.05的周期：模型最小.05228749452768078、实际.022717993560739114rad。q7端点预测速度-.0016603930522735924、真实+.2765894764808469rad/s，误差.27824986953312053，超过此前同关节最大.1408884368818284。q7的q+.1v预测误差.031686211994539804rad，也超过过去位置peak+.1*速度peak经验界.0182475995270067；因此连新加入过去速度peak也不能宣称覆盖本次或未来突变。step387/388实际向外速度约-.002552/-.00008044，不能把389偏差描述成早已测得同等大小的当前速度。位置经验收紧未处理速度响应失配，冻结名义制动可行不代表真实制动成立。

下一步需定位该速度响应突变与实际控制/接触/动力学变化的关系，并评估同时考虑速度预测残差的提前恢复；未经核实不把任何原因写成结论，不事后扩大包络冒充因果先验，不改变固定控制器或验收门槛。上述均非新控制资格，18/WAM/fresh未执行，总目标保持未完成。

## 速度突变原生观测回放：新发生的夹爪—炉灶接触

新增replay/execute_tracking_v5_velocity_jump_replay.py，仅回放已执行的同一候选390命令（368前缀+22反馈），不调用控制求解器；最后step384..389逐2ms读取控制器原力矩、robot截断力矩、ctrl、qfrc_actuator/constraint/bias、接触geom名称与速度，native sim.step只调用一次，不增加forward。原始MuJoCo力字段按before/after原位记录，其计算相位不能描述成额外瞬时力重算。保存实际robosuite single_arm.py/osc.py/control_utils.py源指纹。

owner actual_wait=true/returncode0、release reaped=true/returncode0，390物理周期，model/fresh0；所有391状态与550 inner关节最大差均0，说明新增观测保持原轨迹。两个入口ruff/format通过。outputs/tracking_v5_velocity_jump_replay_20261028/verified.json SHA bd50996827c470c0e8bfe2c6396cd26bc0ae5ed92ca97b79bc0407981edafcb5，150条inner观测保存在inner_dynamics.json（6×25）；原source数据与固定控制器未修改。

step384..389的原始当前J最小奇异值约.19038→.20389、M条件数约8.873→8.992，未见明显病态。此前接触均为物体—桌面；最后6周期内首次涉及robot/gripper的接触出现在step389 inner14(零基)：flat_stove_1_collision_burner与gripper0_hand_collision，distance=-7.172583918690911e-6m，after qd7=.04108191230830347rad/s。inner15 qd7=.07794614、inner20 .21528434、inner24 .27658948；同周期after qfrc_constraint7最大绝对值15.134657492793188，而此前周期最大约.1。q7控制器力矩始终未达到±12限制；step389最后inner22/23/24是第6关节出现截断，其他先前记录均无截断。不能把整次事件描述成q7自身力矩饱和，也不能无依据归因为J/M奇异。

这是接触事件与速度响应突变同步的直接原生日志证据，尚非移除接触后的反事实因果证明。当前自由/理想关节限位模型未建模该夹爪—炉灶碰撞，过去自由运动残差不能自动覆盖新接触模式，位置/速度经验包络均不提供未来保证。后续应核对已生成目标与机器人实际扫掠几何，分清参考可行性、固定场景约束与当前可测接触反馈；对象ground truth仅允许离线诊断，不能借此把炉灶/对象标签偷偷加入在线S2A。任何接触干预仿真仅用于机制诊断，不作为原固定环境控制或任务资格。原18/WAM/fresh验收保持，总目标未完成。

## 离线刚性手部参考几何：接触时参考端点分离，较晚参考端点重叠

新增convex_mesh_box_diagnostic.py及audit/execute_tracking_v5_hand_reference_geometry.py：用实际MuJoCo hand碰撞mesh顶点的凸包、burner box尺寸及姿态执行完整分离轴测试（凸包面法向、box轴、边叉积）。输出是单位轴投影最大gap，不称欧氏距离；检查仅覆盖该一对凸形几何及离散端点，不是全机器人连续扫掠/IK可行性证明。解析box已知分离/接触/重叠及共同刚体变换fixture native311 CPU通过(3.24s)，四文件ruff/format通过。

从保存状态368/388/389/390独立mj_forward验证TCP—hand刚性变换在1e-9内一致、stove固定pose在1e-12内一致；将相同hand刚性几何放到已生成chunk46的全部40个目标TCP。实际endpoint gap分别.030165949479785903/.009312061321561038/.003422712206540224/-.0008848808254341423m；endpoint390 SAT负gap与原生接触dist精确相同，前面三个原生均无该pair接触。

候选step389命令对应cursor21/desired endpoint390，该参考的selected-pair最大轴gap+.06896992571880545m（分离），邻近cursor20/22/23也分离。真实step389完成后TCP位置误差.033561050249736296m、SO(3)误差.5209217757245937rad；step387/388也已约.03228/.03379m和.45775/.48375rad。其hard-witness记录original_first_pose_caps_satisfied=false。不能将这次接触简单解释为同一参考端点本身重叠；当前控制为保持关节模型内区牺牲了跟踪精度，手部偏离参考后发生接触。这里仍未做单独接触移除干预，不把几何同步事实冒充完整因果证明。

该同一chunk的cursor36..39（局部held-chunk实验desired endpoint405..408）刚性手部与burner确实重叠，轴gap分别-.0047686215637609/-.011645878604503834/-.01435867376105815/-.017552248184917074m。这些较晚点不一定会在原v5每8周期重新生成参考的生产中执行，不能外推为全量WAM任务不可行；但局部固定40参考对照中不能默认所有目标几何可行。两项问题需分别处理：当前跟踪误差导致的早期接触、较晚参考自身的selected-pair不可行。

owner actual_wait=true/returncode0、release reaped=true/returncode0，时间推进/model/fresh0。outputs/tracking_v5_hand_reference_geometry_20261028/verified.json SHA c4e6d9932be57cee397d8ef8a0cbd179b2c4f1571de609ac4e5a799f0103fdeb。所有对象几何仅离线诊断，不输入在线S2A；未改变固定环境控制器、原跟踪/任务或18/WAM/fresh验收门槛，总目标未完成。

## 跟踪偏差在接触前已由模型预测；增加首周期绝对精度约束的数值检查

同一inner-profile候选step376模型/实际旋转误差.043993/.044537rad，step377已.078392/.078859，之后同步增大至step388 .483480/.483747，step389 .520581/.520922。位置同样从约.00587/.00580m逐渐增大，step388 .033811/.033793m。因而不能把接触前全部跟踪偏差归因于未建模的碰撞冲击：求解器已预计并接受这些误差，hard joint witness及有限horizon总目标不代表首周期跟踪合格。

新增quality_capped_inner_joint_plan.py，在原inner-profile位置、固定.1s制动、终端、box/rate全部保留的同时，显式约束首周期TCP位置≤.01m、SO(3)≤.05rad；解析残差Jacobian、同一160响应预算，最终缓存逐点重查。未找到解返回nonlinear_search_no_witness，不声称全局不可行，不转放宽动作。两项native311 CPU测试通过(1.11s)：已知合规目标/计划同时保持关节及精度约束、仅joint-feasible但首目标相距1m的计划不误接受。四首版文件ruff/format通过。

probe/execute_tracking_v5_quality_capped_numeric首版在原分支完成后因沿用len(rows)==40断言而拒绝22周期候选，actual_wait=true/returncode1、release reaped=true/returncode1；原失败registration/部分输出保持，不计为完整结果。另建不可覆盖v2入口按每分支实际长度执行，两个入口ruff/format通过，owner actual_wait=true/returncode0、release reaped=true/returncode0，physics/model/fresh0。outputs/tracking_v5_quality_capped_numeric_v2_20261028/verified.json SHA b9c7b08a9354e6e7d0d0b16b75e970b146492a494dfdca2662a6b9d1f48817f7。

两分支step368/375/376均有quality model_witness；当前候选376第一误差约.005869805m/.043993254rad，原分支376约.005027079m/.040504471rad。step377/378/382两分支名义profile仍witness，quality搜索未找到解；原分支377耗尽160响应预算，其他失解不等于数学不可行。候选389同样未找到quality解，原389仍初始buffer外。该阶段只是原轨迹捕获状态反事实数值搜索，未执行新增控制，不进入18/WAM/fresh资格。

下一步可从偏差尚合格的早期状态比较原时序继续推进与暂缓参考相位；必须保留原参考路径/次序，明确属于重新时间参数化、记录实际相位及全部未完成点，不把重定时效果当作原时序跟踪通过。固定20Hz控制器与原始开发/fresh任务门槛保持，冻结v5不改动。未经验证不声称暂停相位能恢复可行性或带来任务成功，总目标未完成。

## 显式保持上一参考点：数值对照与原生单周期/独立回放关闭

新增probe/execute_tracking_v5_reference_hold_numeric.py，使用同一实测状态、同一分支过去inner误差profile、同一box/rate/关节/制动/终端及首周期.01m/.05rad要求，比较原时序未来目标与保持上一已生成目标、期望twist=0。两分支step376两种模式均有witness；step377原时序搜索未找到解，但保持有witness（原分支首误差.004982983m/.024118708rad，inner候选.004649424m/.036885538rad）；step378及382两模式搜索均未找到解，部分耗尽160预算，非数学不可行证明。owner actual_wait=true/returncode0、release reaped=true/returncode0，physics/model/fresh0。verified SHA 1f6cea9f8752903f771dca3b28bd811f060ededad16408a11f46137b69d3cd41。

新增probe/execute_tracking_v5_reference_hold_native.py：两分支精确重放同一377周期实际前缀（368原前缀+9个inner候选实际动作），在step377分别发送原记录timed动作和保持参考cursor8的数值witness动作。只改变这一周期arm动作，gripper均使用相同原记录命令；不是完整姿态/夹爪协调重定时控制器。正常原生756周期，实际wait0/reaped=true/returncode0。producer result SHA 608a74c52a9b19efa288c607e16d963e1d4959de72c6a4fb1467a46e58c59a78。

|单周期真实结果|原timed动作|保持cursor8动作|
|---|---:|---:|
|实际inner最小margin(rad)|.07078274298304699|.06995242884771091|
|对保持目标的位置误差(m)|.008729770574169817|.004912610100601898|
|对保持目标的旋转误差(rad)|.03712100248931948|.03719512145884468|
|对原时序下一目标的位置误差(m)|.007453504189239875|.004536202804683285|
|对原时序下一目标的旋转误差(rad)|.07885880608484654|.08779647201033672|
|任务/抓取|false/false|false/false|

保持动作符合其显式保持目标精度，仍保持.05关节余量；对原时序旋转误差更大，不能说原时序跟踪改善或任务成功。原timed动作对上一目标也在精度管内，故不能把保持模式的合格本身当作胜过基线的充分证据。尚无恢复后继续推进或完整参考消费证据。

新增check/execute_tracking_v5_reference_hold_native_check.py，独立只重放保存命令，追加756物理周期；两分支全部379状态以及最后25个inner q/qvel逐元素精确一致，时间、最小margin、两参考误差、任务/grasp全部核对通过。owner actual_wait=true/returncode0、release reaped=true/returncode0。verified SHA 82b3ba41a9bdca987a1c077cba2c262b951c92a5e3542d3856d81597651d9a50，根outputs/tracking_v5_reference_hold_native_check_20261028；合计1512物理周期/model0/fresh0。六新增入口ruff/format通过，无新增完整控制资格。

下一步应检查保持后的真实状态是否可恢复推进下一原参考点，并更新候选自身这一个周期的因果误差；不得继续借用旧分支后续状态或误差。任何连续相位策略必须明确标记重定时，保留参考顺序与未完成点，不冒充原时序验收。原固定控制器、18/WAM/fresh门槛及总目标保持。

## 提交前状态核实：v5第三案例生产关闭，第四案例继续

第三案例task0_init0_s43_raw原worker2402406已终止，orchestration对应exit actual_wait=true/returncode0、release reaped=true/returncode0；600周期/75请求，worker SHA 3335150cda78522a02f98a45d9d8f96f26a2539b11cf06d9ac185c42b11ac54a。保存日志描述器case02 SHA 9bf38011cc4dae5aeabd6ecf28f05f2c19bebb1de8ff87f4e9b28cdd4f18d5da：任务/抓取周期均0，TCP位置RMSE .0124833135398095m、SO(3) .05502045744605921rad；q7期前越界258周期、首step121、最小margin -.0031933104097650045rad。此为已关闭生产的描述统计，不是独立全量资格。原owner2358795和同一模型service2359094继续运行；kernel已核实第四条task0_init0_s43_terminal058 worker2494361存活。前三条均无任务成功，不提前判定完整12矩阵或fresh资格。

## 用户要求停止GPU任务（2026-10-28）

用户明确要求在另行通知前不使用GPU。向已核实的v5 owner2358795发送SIGTERM，由其清理拥有的当前采集2494361及模型service2359094；随后kernel确认三进程均不再存在，相关run/serve/collect进程列表为空，nvidia-smi compute-apps查询亦为空。不自动恢复或排队GPU任务；已完成前三案例和第四案例部分输出保留，第四条中止不能记作完整或性能通过。CPU全量测试进程保留，后续仅CPU工作。临时禁用约束已追加AGENTS.md，恢复须等待用户明确通知。

## 保持后真实状态的恢复数值检查已关闭（CPU）

新增probe/execute_tracking_v5_reference_resume_numeric.py。输入为已经独立逐状态/inner q/qvel复现的held分支实际endpoint378。误差profile仅取原候选368..376九个已完成周期，再用实际held周期377的25个q与该实际命令的预测更新，最后完成步377、共10个周期；不读旧分支step378后状态或误差。XML mj_forward核对实际最终q/qvel，重新计算M/J，用真实上一命令作seed及上一已生成held计划移位作warm start。

同一真实state/profile/box/rate/关节/制动/终端/.01m/.05rad条件下，继续hold cursor8有model_witness（首模型误差0.00338706387481883m/0.0351840759719188rad），resume cursor9的非线性搜索未找到witness。没有证明全局不可行，也没有恢复推进的实际证据；不能把单次保持升级为连续任务控制成功。owner actual_wait=true/returncode0、release reaped=true/returncode0，physics/model/fresh0；两个入口ruff/format通过。outputs/tracking_v5_reference_resume_numeric_20261028/verified.json SHA 95f62cc3e0a892dbf5633cb6197e47cd19f7f5832a4e0c4bf3c676e372da4a1a。

GPU保持禁用，后续只做CPU工作。下一步若继续保持/恢复对照，须实际执行并更新候选自身历史，逐周期记录相位与未消费目标，保持完整轨迹和原任务验收要求；原总目标未完成。


## 保持后原参考推进的固定初值检查（2026-10-28，CPU）

按[固定初值计划](../plans/v0.0.1/reference_resume_multistart_20261028.md)，在同一实际held endpoint378和自身因果历史上，保留原cursor9..16及全部原约束，分别从上一命令重复、held计划移位和原线性witness启动。三项均未找到非线性质量witness，响应数45/107/41，合计193，均非预算耗尽；不证明全局不可行，也未取得新的控制/任务资格。新增物理周期、世界模型调用及fresh案例均为0。actual wait0/reap0已核验，verified SHA `cb52615c6dfd6d51088958e95e6cab3636b5ec55b67159cf01a1ec28fe3544e8`。

首版owner在探针执行前因两份已提交测试的指纹变化终止，原日志保留。v2仅明确登记测试9db6efa→2f09739的旧新SHA迁移，其余原v5源码仍按冻结指纹核验；原v5注册不修改。下一步应验证多周期保持后的真实反馈是否支持恢复推进，不在相同状态继续任意重启。GPU继续禁用，18条/WAM开发/final24门槛均不变。


## 多周期实际保持/恢复复核关闭（2026-10-28，CPU）

[完整记录](reference_recovery_native_20261028.md)：从实际held endpoint378执行16周期反馈，16次hold、0次resume，最后消费cursor8；最大保持位置/旋转误差.00334615924m/.04158785168rad、最小内环关节裕量.05324711229rad，task/grasp均0。独立394周期重放对齐真实传感刷新调度后，395端点与400新内环q/qvel/time全部核验，state/q/qvel最大差异均0；actual wait0/reap0，verified SHA 807632bd10e8cec0d9a1bb61fff378622dcf2ef7e34381ee3264d1b4abbaa259。保留前两版1及58周期失败及v3预执行失败记录，总新增实际物理847，model/fresh0。只能确认短时保持，仍不能沿原轨迹前进，不取得控制或任务资格。

本次复核发现current_observation会调用sim.forward，遗漏初始/每周期刷新可导致接触段分叉；不能将其当成可省略而物理等价的读取。下一候选检验首周期可行性优先的原约束规划搜索，不继续任意增加保持/重启。GPU继续禁用，原18/WAM开发/final24门槛不变。


## 首周期可行性优先与下一参考几何分离（2026-10-28，CPU）

[先可行后性能规划](feasibility_first_recovery_20261028.md)在同一实际endpoint378/因果历史仍只取得hold8 witness；resume9的最好已检查线性可行尝试旋转误差.07532084839rad超过.05rad，拒绝。两项共238响应，actual wait0/reap0，verified SHA fddcff00e253ad7a6ec9b335bf192f0c20088bf42b548a3a1973390c6f5b21e2，不证明非线性全局不可行。

[静态IK分离检查](recovery_pose_ik_20261028.md)8固定查询证明cursor9在原关节范围内缩.05rad后仍有严格姿态witness，误差9.29328539e-6m/7.64504909e-5rad，关节最小裕量.05025039461rad；独立XML FK最大姿态差2.22e-15、Jacobian差2.104e-10。actual wait0/reap0，verified SHA 658591bb08711b3ceee9125c25da96ffd6876281acfeb0c4f6bd3f14ba956ed4。不能把控制搜索失败归为目标姿态必然不可达，也不把静态解当作已达起点/可执行action；下一步检查实际状态锚定的提前构型调整。新增物理/模型/fresh均0，相关10测试通过，原全部任务门槛和GPU禁用保持。
