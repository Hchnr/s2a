# 实验报告

056最新完整结果：[D2聚合与同步validation](wam_clean_on_policy_aggregation_20261024.md)全部完成，D2 300/10500、1260000行fit/独立QR、validation 99/3465及正常退出均合格；但原性能门槛仍失败。新头比052改善，H16相对raw位置/开度误差仍增大16.255%/139.056%，旋转改善25.799%。不进入开发或fresh，继续诊断局部残差与积分state目标不一致的问题；总目标和原24条最终验收门槛不变。以下为较早阶段快照。

056最新执行：[聚合完整资格与同步validation](wam_clean_on_policy_aggregation_20261024.md)。D2全部300/10500与聚合1260000行fit/独立完整QR、正常退出审查均通过；统一新参数已冻结。1476份来源/快照、独立metadata及实际模型CLI dry通过，固定99请求validation已在唯一允许UUID上启动。尚无新候选性能或任务收益，开发/fresh保持关闭，原055失败及原24条验收门槛不变。以下为较早阶段快照。

056最新执行：[D2完整资格与聚合拟合](wam_clean_on_policy_aggregation_20261024.md)。D2全部300请求/10500节点、300初态配对及独立数值/正常退出审查通过，GPU已释放。固定1260000行的唯一CPU fit已实际启动，完整fit/独立QR及新validation尚待完成，无新任务或fresh收益。原055性能失败、总验收门槛及fresh关闭不变。以下为更早阶段快照。

2026-10-24当前执行：[056自身策略训练轨迹聚合](wam_clean_on_policy_aggregation_20261024.md)。D2已通过完整执行准入、实际模型CLI dry及157项CPU检查，1470份来源/快照冻结；同一owner/worker持续采集，最近核实快照为123/300请求、4305节点。尚未完成全量D2数值/正常退出资格，聚合fit/QR和新validation均未执行，fresh保持关闭。相关拟合、审查、owner及validation接入组件已实现，不能把软件检查当作性能收益。

2026-10-24最新完整性能：[055接续结果](wam_synchronous_state_continuation_20261024.md)数值及退出审查通过，但性能门槛失败：旋转改善，位置和夹爪退化；[配对/合同/分布诊断](wam_synchronous_pair_diagnosis_20261024.md)定位到自身生成轨迹末节点的teacher误差反转，根因尚未唯一确定。总目标与原24条留出初态稳定≥20/24、各任务≥4/8不变。以下为较早阶段快照。

2026-10-24最新结果：[054分派错误终止与prefix审查](wam_synchronous_state_execution_20261024.md)。已完成3zero＋1raw/140节点并独立核验；首probe052在采样前KeyError，原worker/owner False及实际wait1/reap/GPU释放保留。分派回归组件已通过6项CPU测试，055仅剩余95请求的显式接续在准备，未重启模型；无新候选性能或fresh结果。下列054启动为历史快照。

2026-10-24最新执行：[054同步新头配对采样](wam_synchronous_state_execution_20261024.md)已通过124项CPU检查、1447源/快照、两完整AST证明、独立最终准入及实际模型CLI dry，并由持久化owner在允许UUID上启动固定99请求。尚无完整采样或新任务收益；原W2失败、最终门槛及fresh关闭不变。

2026-10-24最新资格：[053同步validation参考](wam_synchronous_validation_20261024.md)完成原六demo的1212物理步、24当前图像与完整独立审查，native稳定正控6/6；当前图像/实测状态/未来端点由真实物理时钟绑定。已分离current-only模型清单，0新WAM采样或fresh。下一步另登记052新头配对采样，原W2失败及旧同步未认证记录保留，总目标未完成。

2026-10-24最新候选：[052单值正则化对照](wam_clean_regularization_probe_20261024.md)已完成同一840000行的新CPU fit与独立完整QR，原容差通过；训练归一化位置/开度MSE改善29.59%/17.83%，尚无新采样或任务收益。原W2开发0/6稳定失败保留，fresh关闭。原六个validation demo来源及初态隔离预检通过，下一步实现带真实物理时钟的同步参考采集，不能用训练误差替代最终验收。

2026-10-24最新完整结果：[051开发对照与补充审查](wam_clean_development_dependency_execution_20261024.md)已完成12/12条、900请求和7200物理端点，独立补充数值审查通过；原审查失败记录保留。raw稳定1/6，W2稳定0/6，开发性能门槛失败，当前W2不进入fresh。已完成[失败定位与夹爪释放条件核对](wam_development_failure_localization_20261024.md)，尚无新候选收益；总目标及预留24条验收门槛不变。下列运行中与部分完成描述为历史快照。

2026-10-24最新观测：[051 bowl两seed完整结果](wam_clean_development_dependency_execution_20261024.md)。4/12条已完成：raw稳定1/2，W2稳定0/2；seed43/W2虽曾成功39步，但终点和末40步稳定判定失败。当前候选即使剩余W2全成功也至多4/6，且bowl为0/2，已无法达到原开发≥5/6及各任务≥1/2门槛。仍完成固定剩余8条以取得完整对照，完整独立审查尚未完成，fresh保持关闭；下列计数和运行状态为此前快照。

2026-10-24最新执行：[051修复后开发对照](wam_clean_development_dependency_execution_20261024.md)。110项组合检查、五入口AST独立证明、583来源/585快照及真实CLI dry通过；新namespace实际启动，首client已越过原缺失合同失败点并完成模型请求，物理进度60端点/8chunks。尚无完整case/稳定率，STATE gate=False及原049失败保留，fresh关闭；下列049停止或051准备为此前快照。

2026-10-24最新结果：[049开发首client依赖失败](wam_clean_development_execution_20261024.md)。首client因旧tool_contract挂载路径缺失在输入指纹阶段退出，0请求/物理；1次模型初始化已abort、wait/reap和GPU释放，原False/562源保留。已找到与历史SHA相同的新挂载合同并建立安全副本，实际sim输入预检通过；051新namespace有限接续在准备，无新执行或fresh。下列049启动段为历史快照。

2026-10-24最新执行：[STATE完整资格与049开发启动](wam_clean_development_execution_20261024.md)。STATE独立99/3465数值、来源及正常收尾通过，但state gate=False和validation初态同步未认证均保留；049冻结562源/快照、独立准入和原CLI实际dry通过，固定12条raw/W2开发对照已启动。尚无完整任务结果或fresh收益，总目标未达到；下列STATE运行中描述为历史快照。

2026-10-24最新准备：[049固定开发接续](wam_clean_development_handoff_preparation_20261024.md)代码及125项组合检查通过，QA绑定8份源码/测试；实际绑定正确拒绝未完成048，0新开发模型/物理。最新观察STATE同一模型已闭合32/99请求、1120节点，551源/快照无漂移，尚无完整STATE资格或新任务收益。

2026-10-24最新：[W2完整资格与STATE启动](wam_clean_w2_and_state_execution_20261024.md)。W2全部840000行拟合及独立QR按原容差通过，正常wait0/reap，537源/快照无漂移；048冻结551源/快照、独立准入及两个真实CLI dry通过，STATE模型已在唯一允许UUID上启动。尚无完整99请求资格或新物理/留出任务收益；下列W2运行中描述为历史快照。

2026-10-24最新：[050 D1完整资格及W2启动](wam_clean_d1_lifecycle_addendum_20261024.md)。D1完整300请求/10500节点及独立审查通过；原046收据字段错误的False保留，050独立生命周期补充核验通过。W2相关150项及收据/接续68项检查通过，537源/快照冻结、独立准入和三个真实CLI dry通过；CPU W2 producer已实际启动，拟合/QR完成资格及新任务收益仍待验证。下列D1运行中描述为历史快照。

这里保存 Git 内的精简报告、指标摘要和完成状态，完整产物位于 `outputs/`。

[当前实现进度与总目标](status_current_20261023.md)汇总已实现、已验证和未完成项；总目标与20/24、各任务4/8的完整验收口径见[036](../plans/v0.0.1/036_project_goal_20261023.md)。下列按日期保留历史推进记录。

2026-10-24最新实现准备：[048生产桥与STATE入口](wam_clean_state_handoff_execution_preparation_20261024.md)已实现并通过129项相关CPU检查及真实解释器导入检查，尚无实际048准入/STATE。模型和独立CPU审查的导入环境分别固定，额外lookup绑定真实PID/birth；11:19同一D1已闭合270/300请求，519源/快照不变，W2仍未拟合。

2026-10-24最新组件：[048 W2到STATE来源桥准备](wam_clean_state_handoff_preparation_20261024.md)只读资格核验和导入隔离组件已实现，135项相关CPU检查及两解释器4次原sampler help解析检查通过。生产桥/角色收据/048执行入口尚未完成，0新模型或STATE；10:47原D1同一进程已闭合156/300请求，519源/快照无漂移，W2仍未拟合。

2026-10-24最新准备：[047 W2双轮接续](wam_clean_w2_preparation_20261024.md)实现及142项相关CPU检查通过，实际sim运行时组件核验通过；真实live D1和绑定CLI均被正确拒绝，0实际W2 fit/QR。10:22快照同一D1进程已闭合66/300请求，原519源/快照未变；完整D1资格仍待完成，最终门槛不变。

2026-10-24最新启动：[046显式D1接续](wam_clean_d1_continuation_20261024.md)完成219项相关CPU检查、519源/快照、独立准入和实际CLI dry；持久化supervisor已持有原041 collector实际模型worker，仅允许UUID对应当前GPU2/minor6。10:07快照已闭合10/300请求，前2请求/70节点原独立重算通过，尚无完整采集/退出资格；W2、任务及fresh未启动，原失败与最终门槛不变。

2026-10-24最新完成：[045实际CPU W1及独立QR](wam_clean_runtime_fit_20261024.md)均通过完整420000行/150标签窗口，参数按原容差一致，508源/快照及两角色实际wait0/reap通过。原043失败保留，累计W1尝试2次/成功1次；原合成60通过/4严格参考比较失败及Decimal诊断如实保留。原D1 head合同只读接入通过，D1/W2/任务及fresh尚未执行，下一步为046显式接续。

2026-10-24最新：[044完整CPU数值兼容资格](wam_clean_runtime_qualification_20261024.md)已核150窗口、10500节点和420000行两路径逐字节相同，498源/快照及实际wait0/reap通过。0fit/QR/模型/物理/fresh；原失败保留，下一次拟合需另行固定新输出和累计预算。

2026-10-24最新：[043 CPU接续与标签运行时核查](wam_clean_training_postcapture_20261024.md)。147项相关测试、495源/快照和独立准入通过；实际W1在首标签重建处停止，1尝试/0参数/0QR。sim首窗口逐值相同、main保留微小差异False，完整数值兼容资格尚未建立；无重试、D1或fresh。下列启动和准备段均为此前快照。

2026-10-23最新接续：[041准入与实际启动](wam_clean_training_continuation_20261023.md)。730 passed/1 skipped、Ruff、484源码/快照及完整独立metadata/执行审查通过；真实D0 dry通过，独立session supervisor在短launcher退出后PPID1存活并持有新D0实际模型。原失败不改、不用于fit，仍无整轮采集/实际fit/任务/fresh成绩。下列准备与旧启动为此前快照。

2026-10-23中断核查：[D0中断与独立接续准备](wam_clean_training_interruption_20261023.md)。原进程消失、20完整＋partial25节点、无wait/reap/exit资格；失败原址保留不用于fit，041新根/明确累计预算及持久化owner准备中，尚无新模型/fit/任务/fresh。下列D0启动段为此前快照；[最终验收差距](wam_final_acceptance_gap_audit_20261023.md)仍未闭合。

2026-10-23执行推进：[040准入与D0启动](wam_clean_training_execution_20261023.md)已完成613 passed/1 skipped、Ruff及独立完整执行审查，462源/快照冻结。原63位seed不可执行问题在模型调用前以新31位seed-plan明确修订，原metadata保留；D0唯一模型已启动，尚无整轮采集、候选fit、开发或fresh成绩。下列训练准备状态保留为此前快照。

2026-10-23训练准备：[038实现与准备](wam_clean_training_preparation_20261023.md)的150当前锚点/300独立noise及24 validation锚点/96矩阵已完成独立metadata审查，执行仍未就绪。新软件不计作实际训练、任务或fresh收益。

2026-10-23最新阶段：[真实clean桥接联合资格通过](wam_clean_bridge_20261023.md)。037实际7请求/245节点及039严格原生mask CPU补充审计、独立peer均完成；原False和准备/peer失败保留，0fit/任务/fresh收益。下一阶段为038有限训练与开发协议的实现/准备。

2026-10-23 10:51+08历史快照见[实现进度核查](status_review_20261023.md)，总体目标及验收见[036计划](../plans/v0.0.1/036_project_goal_20261023.md)。当时034新数据30/30合格，035缺完整结果和退出收据；后续037＋039联合桥接资格见上列新报告。以下内容按历史阶段保留。

当前稳定成功阶段的完整结果见下表。结果均为开发集证据；尚未执行预留新初态验收，无真机。

| 报告 | 已完成范围 | 主要结果 |
| --- | --- | --- |
| [WAM夹爪位置/保持](wam_position_hold_20261021.md) | 8800/15400，24条WAM，14400步 | 曾成功2/24，终态/稳定0/24；映射错误减少没有转化为任务稳定提升 |
| [8400同协议对照](wam_earlier_position_20261021.md) | 12条WAM，7200步 | 稳定1/12，仅alpha0/cream；raw各组0/3 |
| [最新15600的state](wam_latest_state_screen_20261021.md) | 48预测，0物理 | 九项H8/H16/H40指标均差于8400及15400；不将latest作为best |
| [更强guidance的state](wam_guidance_state_20261021.md) | 48预测，0物理 | 旋转/开度退化、位置未达10%改善，不推广 |
| [采样步数的state](wam_sampler_steps_state_20261021.md) | 35/70/140各48，共144预测，0物理 | 70/140九项均差35，不推广；关闭ledger缺失另列 |
| [训练attention backend](wam_backend_state_20261021.md) | 48真实初态配对预测，0物理 | H8/H16位置略差，未过推广门槛；退出内部ledger缺失另列 |
| [双种子检查器](wam_seed_aware_checker_20261021.md) | 27旧case只读精确回归 | 只扩显式seed/来源接口，不新增任务成功 |
| [真实43执行资格](wam_seed43_execution_qualification_20261021.md) | 3oracle，1800步 | 3/3，和对应42逐步相同；未验证43 WAM/新初态 |
| [60秒执行资格](wam_long_qualification_20261021.md) | native3＋oracle3，7200步 | 全6稳定，实际控制前600精确；原终点诊断日志strictFalse与舍入原因保留 |
| [60秒原始WAM](wam_long_raw_20261021.md) | 3WAM，3600步，450请求 | 仍1/3稳定，仅bowl尾43.1秒保持；wine/cream后半跟踪约1.9mm仍失败 |
| [raw8第二seed复测](wam_raw8_seed43_20261021.md) | 3WAM，1800步，225请求 | seed43全0/3；不能复现seed42唯一bowl成功，不拼seed/任务 |
| [固定短间隔WAM](wam_short_replan_20261021.md) | 12WAM，7200步，678请求 | raw8仅bowl1/3；其余三组0/3；外部未使用历史权重缺失的收尾例外保留 |
| [按跟踪误差重观测](wam_tracking_replan_20261021.md) | 3oracle通过后3WAM，共3600步 | WAM完整0/3稳定、97请求；不推广 |
| [短重观测执行资格](short_replan_qualification_20261021.md) | 12条oracle，7200步 | 12/12稳定，只证明正确参考附近可执行 |
| [整episode参考增益诊断](wam_reference_gain_diagnostic_20261021.md) | 缓存拟合/留出，0新推理/物理 | 数毫米位置收益、朝向/开度有退化，不推广 |
| [世界旋转漂移诊断](wam_rotation_drift_diagnostic_20261021.md) | 19次缓存拟合、六整episode留出，0新推理/物理 | H8/H16旋转改善10.50%/21.63%，仍差于简单对照，不推广 |

下一项已登记[020因果参考集成](../plans/v0.0.1/020_causal_temporal_ensemble_20261021.md)，执行进度见[结论](../plans/v0.0.1/conclusion_20261021.md)和[记录](../plans/v0.0.1/record_20261021.md)。


本轮报告为 `droid_episode0_20261016.md`，摘要为 `droid_episode0_20261016.json`；对应 [实施记录](../plans/v0.0.1/record_20261016.md) 和 [结论](../plans/v0.0.1/conclusion_20261016.md)。所有实验均标明代码/数据指纹、控制接口、划分与证据范围。

重建报告：`LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage report --plots`。它读取已有指标，不重新选择模型或增益。图在 `outputs/droid_episode0/figures/`，报告链接保留本地完整产物的位置。

首轮仅有日志、几何和单 episode 局部模型内证据；增益调度/小网络、独立仿真与真机未执行。不能以日志 action 误差或局部模型的自洽宣称控制成功。

跨数据集扩展见 `cross_dataset_franka.md` 和 `cross_dataset_franka.json`：RoboMIND-v1真实机器人日志与M1/A1仿真日志分别列出，不计作真机执行。完整逐帧样本、指纹和日志在 `outputs/cross_dataset_franka/`。

后续独立物理仿真见 `libero_simulation_20261016.md` / `.json` 及对应validation文件。分别列出原生expert、oracle状态参考、旧5600模型预测固定参考，以及新8800模型实时图像闭环；执行状态以报告完成计数为准。它不改变首轮DROID日志/模型内实验的证据等级，也不构成真机验证。重建入口为 `scripts/summarize_libero_simulation.py --seed 42`。

2026-10-19控制合同审计见 `libero_control_audit_20261019.md/.json`。它区分HDF5 oracle、发布R2参考、保存模型参考和实时模型策略，另列延迟/噪声、首次成功及完整轨迹终态；数据来源核查与软件/打包验证摘要分别保存。所有生成trace、环境、原始预测、视频和图写在outputs，Git只保留精简报告。

## 最新继续阶段

`libero_followup_20261019.md/json`：catalog与发布修订、同源图像WAM目标、当前端点测量及35项异步物理验证；完整性与任务成功分别记录。软件与打包摘要为 `validation_followup_20261019.json`，完整产物/图/归档索引在 `outputs/libero_followup_20261019/`。
