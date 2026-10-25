# 061固定闭环开发对照：执行审查通过，候选性能失败

## 最终完整结果

固定12/12条、900请求、31500去噪/头调用、7200物理步、1次模型初始化全部完成。独立审查`review/DEVELOPMENT_verified.json`（SHA03fdc14d4aa76e52fa9c4faf66943e1f17c9d0764339e5ea572b447755427bfe）complete/passed=True，独立复算31500节点及全部机械日志、6组首当前条件配对，消费前后SHA一致，0未来标签解码。该审查没有新增模型、物理或拟合。

最终owner `orchestration/DEVELOPMENT/result.json`（SHA848c23c5fa4ceaa51a242d7ccc951e65bf97f73bd2b40922edb0b955f46515be）complete/passed=True。审查与模型实际wait0/reaped=True/signal_sent=False；GPU释放收据显示允许UUID内存0、compute_apps空、released=True。原owner与审查进程均已退出。

| 方法 | 稳定成功 | task0 | task1 | task2 |
| --- | --- | --- | --- | --- |
| raw | 1/6 | 1/2 | 0/2 | 0/2 |
| terminal058 | 0/6 | 0/2 | 0/2 | 0/2 |

候选ever与terminal也均0/6。原开发稳定≥5/6、各任务≥1/2及严格适配收益门槛全部失败；eligible_for_later_fresh_candidate_freeze=False、fresh_performance_authorized=False。060参考state改善未转化为061任务收益，不采用058作为最终候选。独立仿真暂停物理等待推理，不能认证实时或真机；总目标及原fresh24条稳定≥20/24、各任务≥4/8尚未达成。

后续先结合完整参考路径、对象离线位移与固定控制器受限执行定位接近/抓取/搬运失败，再登记新的固定候选与开发对照。保留本轮全部失败和原门槛，不用诊断阈值触发在线动作。以下为按发生顺序保留的运行快照。

060通过原参考state门槛后，061完整准备与独立metadata、真实CLI dry已完成，现已实际启动固定开发对照；不是参考轨迹评分的重复。058参数SHA30210a2bda69962bbd24f8e86849f58c4516d2209bfb3d8a5903fc45ae543b4f及控制器不变。

实际component QA共83项通过；保存完整命令、returncode0及源码引用。开发准入代码在接入期间更新后，另存input_preflight_py311_v2，实际Python3.11重新完成全部12案例输入指纹检查，原预检记录不覆盖。准备冻结1505份源码及1505快照。

prepared SHA1ebc56899ac59df16bc30877748df8752bba4f4f17fb3348c9a6ac6d0ba6e1d8；独立metadata SHA4e4e478e697ebe927c6aebfa6ef549734cf6a66c43e7d27cd2ee5fde03553d10；registration SHA9410294c56482cd61a8f3b5cd4306107c4797ac2ee1964759e1f1c887c184cd4。监督、实际模型解释器服务及全部12实际仿真客户端共14次CLI dry全部return0，0模型初始化/0物理步，successful_cli_dry.json保存各命令/stdout/stderr。

启动前实际允许UUID GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f空闲（visible2/minor6），HTTP31363与MASTER_PORT31364均空闲。owner完成真实写入fsync、至少10GiB空间检查及GPU锁/资源检查。只按此UUID启动模型，CPU/OSMesa仍空CUDA/LD。

短launcher正常返回。持久化owner PID3136757/birth580895280，模型PID3136783/birth580895457，首CPU物理client PID3137539/birth580901312；三者均按/proc实际birth复核存活。首task0_init0/seed42/raw已完成模型服务ready、case绑定、/info及在线generate往返；观察快照为4份请求、3份响应、20条已落盘物理端点（含正在执行中的chunk，不作精确最终计数）。不把日志缓冲时的落盘数当最终物理计数。

固定预算12案例：raw/terminal058各6条，3任务×seed42/43；总900请求、31500去噪节点、7200物理步，最多1次新模型初始化。节点0–33使用052、34使用058；同一固定S2A/控制器、8步重观测、H40/20Hz、每例600步及末40步稳定判据。候选开发稳定≥5/6、各任务≥1/2门槛不变。暂停物理等待推理，不能称为实时成功。

当前尚无完整案例、独立31500节点/7200端点审查或开发稳定率结果。继续观察同一owner/service/client，不因观察超时重启。开发通过并统一候选冻结后才另登记原24条fresh稳定≥20/24、各任务≥4/8；fresh仍未执行，实时和真机另验，总目标未完成。

后续实际复核：同一owner3136757、模型3136783及首物理client3137539均按原birth存活。首task0/init0/seed42/raw已返回22/75份模型响应，已落盘160条物理状态记录，未发现新终态或错误。落盘受缓冲影响，仍不作最终步数或稳定率判断；固定12案例及全部预算不变，未追加或重启实验。

后续快照达到47/75份响应、360条已落盘状态，三进程同birth存活。等待期间实际完成只读首查询回归：当前input_0000.png、首个完整物理Action128和normalized Action128与原051同task/init0/seed42/raw逐值一致，消费输入前后SHA不变；raw_first_query_regression.json passed=True、partial=True。该诊断不读取新未来目标、不调用模型或物理，不证明整条raw轨迹、terminal058效果或最终稳定率。首查询迁移基线成立后继续原固定批次，无参数调整或重启。

## 首个raw完整结果与首候选开始

task0/init0/seed42/raw完整600步、75请求收尾，worker passed/complete=True；实际客户端wait0/reap、signal_sent=False。独立读取全部600个native success bool并使用冻结outcome函数复算：ever/terminal/stable均True，successful_steps=262、trailing_success_steps=262，末40步全部True，与原051该raw案例的原生结果计数一致。first_raw_case_observation_v2.json保存worker、physical_result、steps及实际退出收据，消费输入前后SHA不变。只读observer首次局部变量lint命名随后修正，原source及产物通过observation_source_revision.json和observation_sources保留；不改生产闭包、参数或实验。

这仅是raw基线的1条完成案例，不能写成058候选或整个开发成功。固定剩余11条继续。对应terminal058客户端PID3157088/birth581033392已按实际/proc确认存活，同一owner/model未变；观察快照为候选12份响应、80条落盘状态，包含正在执行的chunk。

首候选已封闭查询的35节点由独立061数学helper实际核对通过：0–33使用052、34使用058，changed_active_values=13404。与raw的首当前PNG、原始噪声/条件/文本数组逐值配对通过；first_candidate_request_witness.json passed=True、partial=True，candidate_stability_qualified=False。该检查0新模型调用/0物理步，不替代完整75请求、600步或最终31500节点/7200端点审查。总目标与fresh关闭不变。

后续对已完成raw实际调用冻结的source_proof及完整verify_case(full=True)作独立600步机械回放，输入前后指纹复核通过，outcome仍为ever/terminal/stable=True、成功/尾段262步。first_raw_mechanical_witness.json passed/complete=True；独立callback明确标记standalone mechanics only，没有冒称已完成全服务75查询绑定或历史003回归。该离线检查0新模型/0物理步，whole_development_qualified及candidate_stability_qualified均False。

同一058候选客户端继续运行，观察快照达到29/75份响应、220条已落盘状态，owner/model/client均按原birth存活。尚无候选完整600步结果，不调整参数或提前统计整个候选稳定率。

## 首组成对案例完整：参考门槛收益尚未转化为本条闭环成功

task0/init0/seed42的terminal058已完整600步/75请求，客户端真实wait0/reap、signal_sent=False。独立读取600个native bool：ever=False、terminal=False、stable=False，successful_steps=0、trailing_success_steps=0。对应raw为ever/terminal/stable=True、成功/尾段262步。worker passed=True仅表示完整预算执行及正常退出，不能写成任务成功。

first_pair_observation.json绑定两条worker、physical_result、600步日志及各自退出收据，partial_development=True，记录2/12案例、150请求、1200物理步。随后实际对terminal058调用同一冻结verify_case(full=True)完整机械回放通过，first_candidate_mechanical_witness.json确认其native outcome仍为0/600失败；它是数学/日志一致性通过，不是性能通过。两次只读核验均0新模型/0物理步，生产闭包不改。

对各自生成参考的tracking指标：raw位置RMSE0.007029629m、旋转RMSE0.021189123rad、开度MAE0.036805254；terminal058分别0.010839048m、0.030543418rad、0.030642039。参考轨迹不同，这些不是到同一任务目标或专家轨迹的误差，不能单独归因于S2A或世界模型的唯一根因。060参考指标通过不能替代本次实际闭环任务验收。

当前只完成1条候选，尚非完整6条稳定率。剩余10条按原固定计划继续；同一owner3136757和模型3136783仍存活，新task0/init0/seed43/raw客户端PID3176919/birth581164112已核实存活，观察时已有12份响应。没有根据首条失败调整参数、案例或门槛，fresh保持关闭，总目标未完成。

## task0两seed完成：原开发性能门槛已无法达到

固定12条已全部完成：最后task2/init0/seed43/terminal058完整600步/75请求，实际wait0/reaped=True/signal_sent=False；600个native bool全部False，末40步0/40。`task2_seed43_candidate_native_observation.json`绑定worker、steps和退出收据，消费前后SHA一致，0新模型/物理/fresh。原生结果观察为raw稳定1/6、terminal058稳定0/6，候选开发门槛失败。模型与客户端均已退出，原owner启动实际独立审查PID3374959并按/proc确认存活；完整服务/机械审查及最终owner收尾仍待完成，不能把执行passed当性能通过。

最新运行观察：最后一条task2/init0/seed43/terminal058已返回45/75份响应，落盘360条状态记录；原owner3136757/birth580895280、模型3136783/birth580895457、客户端3357359/birth582331742均按/proc复核同birth存活。缓冲中的落盘数不作最终步数；尚无最后案例或完整独立审查结果，固定参数与fresh关闭不变。当前挂载df显示可用约151TB（使用率86%），未执行清理。

第11条task2/init0/seed43/raw完整600步/75请求，实际wait0/reaped=True/signal_sent=False；600个native bool全部False、末40步0/40。`task2_seed43_raw_native_observation.json`绑定完整日志与退出收据，消费前后SHA一致，仅为原生结果及正常退出观察。当前11/12条完成，raw六条已全部完成，观察稳定计数1/6（最终独立审查待完成）。最后一条task2/init0/seed43/terminal058客户端PID3357359/birth582331742已按实际/proc确认存活，原owner/model不变。继续最后固定案例及其后的完整审查，fresh关闭。

第10条task2/init0/seed42/terminal058完整600步/75请求，实际wait0/reaped=True/signal_sent=False；600个native bool全部False、末40步0/40。`task2_seed42_candidate_native_observation.json`绑定完整日志与退出收据，消费前后SHA一致，仅为原生结果及正常退出观察。当前10/12条完成，候选已完成5条均不稳定。第11条task2/init0/seed43/raw客户端PID3338799/birth582202773已按实际/proc确认存活，原owner/model不变。最后2条继续固定计划，完整独立审查尚待完成，fresh关闭。

第9条task2/init0/seed42/raw完整600步/75请求，实际wait0/reaped=True/signal_sent=False；600个native bool全部False、末40步0/40。`task2_seed42_raw_native_observation.json`绑定完整日志与退出收据，消费前后SHA一致，仅为原生结果及正常退出观察。当前9/12条完成，第10条task2/init0/seed42/terminal058客户端PID3316825/birth582072762已按实际/proc确认存活，原owner/model不变。最后3条继续原固定计划，完整独立审查尚待完成，fresh关闭。

第8条task1/init0/seed43/terminal058完整600步/75请求，实际wait0/reaped=True/signal_sent=False；600个native bool全部False、末40步0/40。`task1_seed43_candidate_native_observation.json`绑定完整日志与退出收据，消费前后SHA一致，仅为原生结果及正常退出观察。当前8/12条完成，task1两seed的raw和terminal058均稳定0/2，候选累计已完成4条全部不稳定。第9条task2/init0/seed42/raw客户端PID3296711/birth581943228已按实际/proc确认存活，原owner/model不变。最后4条按固定计划继续，最终全服务/机械审查仍待完成；原门槛失败及fresh关闭保留。

第7条task1/init0/seed43/raw完整600步/75请求，客户端实际wait0/reaped=True/signal_sent=False；600个native bool全部False、末40步0/40。`task1_seed43_raw_native_observation.json`绑定完整日志与退出收据，消费前后SHA一致，范围仅为原生结果及正常退出观察。当前7/12条完成，第8条task1/init0/seed43/terminal058客户端PID3276854/birth581813230已按实际/proc确认存活；原owner/model不变。固定剩余5条继续，最终全服务/机械审查尚待全批次完成，fresh关闭。

第6条task1/init0/seed42/terminal058完整600步/75请求，实际wait0/reaped=True/signal_sent=False，600个native bool全部False、末40步0/40。`task1_seed42_candidate_native_observation.json`绑定完整日志与退出收据，输入前后SHA一致，仅记录原生结果和正常退出，不替代最终全服务/机械审查。固定队列已完成6/12，候选已完成3条且均不稳定；第7条task1/init0/seed43/raw客户端PID3255696/birth581684331已按实际/proc确认存活，原owner/model不变。剩余6条继续，原开发门槛失败及fresh关闭保持。

随后第5条task1/init0/seed42/raw也完整600步/75请求，实际wait0/reaped=True/signal_sent=False；600个native bool全部False，末40步0/40。`task1_seed42_raw_native_observation.json`绑定完整输入和退出收据，SHA前后复核一致；仅记录原生结果及正常退出，不冒称完整服务/机械审查。共5/12条完成，第6条task1/init0/seed42/terminal058客户端PID3235505/birth581554408已核实存活，owner/model仍为原3136757/3136783。固定剩余7条继续，候选门槛失败及fresh关闭保留。

第4条task0/init0/seed43/terminal058完整600步、75请求，worker complete/passed=True；客户端实际wait0/reaped=True/signal_sent=False。独立读取600个native bool全部False，末40步0/40；`seed43_candidate_native_observation.json`绑定worker、steps、退出及回收收据，消费前后SHA一致。该观察不替代最终机械/全服务审查。

截至此时4/12条完成，task0的raw稳定1/2、terminal058稳定0/2。候选已不可能达到原各任务≥1/2及总稳定≥5/6开发门槛，即使其余4条候选全稳定也至多4/6。保持原失败与门槛，不进入fresh；固定剩余8条继续完成以取得完整三任务对照和独立审查。第5条task1/init0/seed42/raw客户端PID3216163/birth581424684已按实际/proc确认存活，同一owner3136757/model3136783继续，未重新初始化模型或改参数。总目标未达到，后续改进应在完整结果诊断后另行固定候选验证。

## 首对只读描述：末段已打开，物体仍偏离目标

后续固定批次已完成第3条task0/init0/seed43/raw：600步、75请求，worker complete/passed=True，客户端wait0/reaped=True/signal_sent=False；全部600个native bool均False，末40步0/40。`seed43_raw_native_observation.json`绑定worker、steps及实际退出/回收收据，并检查消费前后SHA不变；该记录仅为原生结果与正常退出观察，不替代全服务或机械审查。当前共3/12条完成，候选仍仅1条完成。第4条task0/init0/seed43/terminal058客户端PID3196760已按实际birth确认存活，原owner3136757/model3136783未变；固定余下9条继续，fresh关闭。

新增可复现入口`scripts/observe_wam_terminal_pair_failure.py`，只消费首对已完成的worker、chunks和全部600步日志；逐项检查完整600/75、原生bool、75段前缀对应及输入前后指纹，保存`outputs/terminal_development_20261025/first_pair_failure_description.json`。复用既有release horizon分析，脚本和helper源码SHA均绑定。Ruff检查及格式化后实际执行成功；0新模型调用、0新物理步、0fresh，候选全量资格仍False。

两条首步日志中碗相对盘中心位置完全相同，约(-0.1513,0.0194,-0.0041)m。raw终点相对位置约(-0.0190,0.0088,0.0056)m，候选约(-0.1565,-0.0235,-0.0041)m。候选碗最大高度增量约0.01019m，raw约0.02792m；两条碗grasp_contact日志均为0，因此该布尔量不能当作是否完成搬运的判据。中心位置只是离线描述，不能替代原生任务谓词或进入在线策略。

候选末40步参考开度0.96034–1.0，实测0.99032–0.99058，夹爪原因全部为retain_quantized_motor_position；raw末段同样已打开。因此本条失败不能简单归因为末段闭合保持或缺乏打开参考。候选131步饱和，raw47步；这提示后续完整队列结束后应共同检查参考空间路径和受限跟踪，现有日志不足以唯一归因，也不证明任何控制修改会改善任务。固定队列及参数继续保持不变。

## 全12条生成开度前缀/后缀描述

在完整12条日志上实际执行既有`scripts/analyze_wam_release_horizon.py`，保存`outputs/terminal_development_20261025/full_release_horizon_description.json`（绑定脚本、各chunks和steps指纹）。每条75段均检查执行前缀/未执行后缀的对应关系，0新模型或物理；这是离线描述，不是独立完整执行审查或任务验收。

仅task0/seed43/raw在末5段全部出现“执行前缀最大开度<0.9、未执行后缀≥0.9”；其全程43/75段如此。同初态候选全程5/75段、末5段为0。其余11条的末5段均无该现象；因此不能把整个候选0/6失败统一解释为反复重规划截掉打开后缀，也不足以据此统一延长执行窗口。task1/seed43候选末5段生成最大开度仅0.19846，而task0/seed42候选末段已经打开，失败模式存在差异。下一步需结合各任务参考空间路径、实际接触和受限跟踪作完整定位，不将这些诊断阈值用作在线触发或替代原生谓词。

## 完整对照自身参考的跟踪误差描述

`full_tracking_description.json`保存全部12个完整600步producer result及读取前后SHA复核；0新模型/物理/fresh。这里是既有执行器报告值的汇总，独立回放仍由正在运行的完整审查负责；误差分别针对各自生成参考，不能直接当共同任务目标距离。

候选位置RMSE在task1/seed42、task2/seed42、task2/seed43分别为0.002076、0.005020、0.006491m，低于对应raw的0.006407、0.011361、0.015716m；这些候选仍全部原生失败。因此单纯减小对自身参考的跟踪误差不足以证明任务改善。task0/seed43候选位置RMSE则为0.052246m（raw0.028315m），需要另检查受限执行和参考路径。不同参考之间不能据此作唯一因果归因；下一阶段应同时检验参考是否包含有效接近/抓取/搬运过程及固定控制器能否执行，而不是只优化全局跟踪平均值。

## 全量目标物运动：不存在单一释放失败解释

新增`scripts/observe_wam_terminal_object_motion.py`通过Ruff格式与检查并实际运行，要求完整独立061审查通过、固定12条完整600/75以及输入前后SHA一致；结果`full_object_motion_description.json`绑定源码、审查与日志。0新模型/物理/fresh，物体真值仅离线描述，不认证因果阶段。位移均从首条落盘位置计算，而非未记录的物理初始帧。

关键区分：task1/seed42候选位置跟踪RMSE约2.08mm且0步饱和，但酒瓶最大位移仅0.046mm、日志接触0步；task2/seed43候选cream cheese最大位移仅0.0071mm。两条几乎未搬动物体，不能仅用末段释放解释。task0/seed43候选则有293步日志接触、最大抬升0.564m、最大位移1.313m及362步饱和，依旧native全失败；这条具有显著搬运/偏移，不能与前两条归为未接近物体。日志接触不是原生成功必要条件（成功raw仍为0步日志接触）。

该结果支持下一步分别检查“参考未促成有效物体交互”和“显著物体运动但目标失败”的路径，而不是统一放松夹爪保持或增加控制增益。需要接续核对坐标/TCP约定及任务参考空间路径，才能决定改进模型参考处理还是控制跟踪；当前不能唯一归因。

## TCP转换代数核对

实际逐条核对12条×600日志：desired_source @ source_from_tcp 与desired_tcp的最大绝对元素误差2.22e-16；`full_tcp_transform_description.json`绑定全部config、模板、安全合同、worker及steps读取前后SHA。合同为绕z轴90度旋转及局部z=-0.0036m平移。执行源码先measured_source @ h生成查询锚点，再将desired_tcp @ inverse(h)交回控制接口（`src/s2a/stability_position_simulation.py:176`、`:237`、`:246`、`:341`）；不存在日志中漏乘/重复乘该固定变换的证据。

这仅验证已声明合同的代数一致性，不能单独证明真实物理TCP标定或模型训练时的TCP定义正确，也未排除参考路径本身问题。0新模型/物理/fresh；下一步应继续检查低误差但目标几乎静止案例的参考空间覆盖及原oracle正控合同，而不是直接修改已闭合的矩阵转换。

## 参考空间覆盖与正控证据边界

`full_reference_spatial_description.json`描述12条全部执行端点到对应目标物中心的最小距离与xyz范围，并绑定worker/steps读取前后SHA，0模型/物理/fresh。task1/seed42候选desired TCP原点最小距离0.09583m，实际source原点0.09203m；task2/seed43候选分别0.10086m、0.10162m。它们与“目标几乎不动且低跟踪误差”一致，但TCP/source原点不是手指接触点，不以该距离设抓取阈值或直接认证参考不可行。task0/seed43候选desired TCP最小距离0.02901m，提示该条与前述静止物体案例不同。

另核对053正控报告：其6/6稳定来自原生demo命令与SettledGoalEnvironment采集、1212物理步及独立几何/控制状态审查；不能冒充061固定position executor在本轮初态的oracle反解成功（见`reports/wam_synchronous_validation_20261024.md:26`）。现有证据下一步需要明确核对同固定控制接口的参考oracle正控是否覆盖三任务/初态，必要时登记只用CPU的匹配正控，再决定参考承诺、逆模型或反馈修改；不直接归罪于训练TCP或盲调增益。

## 找到匹配历史oracle与安全副本核验

进一步定位到015完整oracle审查（`reports/wam_seed43_execution_qualification_20261021.md:3`）：三个task/init0、seed43、600步/H40/每8步重观测/position_hold/no governor均3/3稳定，并与007对应seed42全部1800个step字段精确相同。实际读取资格JSON SHA9cd4cf46260eda2d87e13bd992c5ce871469d4ff2d3f38734129e2a1bd5fbba3，complete/qualified=True。其fixture SHA、inverse_weight、max_command_rate、全部gripper_position_hold、执行/重观测长度及variant与061配置逐字段一致。

对该审查记录的全部389输入，将旧仓库前缀明确映射到当前安全仓库后实际逐文件复核：389 SHA精确一致、0缺失、0变更；保存`historical_oracle_availability.json`，未访问写入或恢复旧目录。由此历史匹配正确参考资格有完整安全副本，不必仅因053证据不匹配就重复新仿真。仍需进一步核对061执行源码与该正控源码/初态绑定，才能声称完整当前接口桥已闭合；正控结果不计WAM成功或fresh。

## 历史oracle到061源码/初态桥：XML身份仍需解释

实际保存`historical_oracle_current_bridge.json`：历史oracle绑定的18份S2A源码与当前字节全部一致；12个061 manifest与对应task oracle逐项比较name/task/init、完整initial_state及SHA、fixture model_xml、expert SHA均一致。按明确仓库前缀迁移归一化后，模拟器源码集合均165份、0新增/缺失/变更，package_versions一致。

但12条的initial_model_xml_sha256均与历史oracle不同。因此目前不能宣布整个物理环境身份完全相同；需继续定位是否仅资产路径迁移产生XML字节差异，还是存在实际模型差异。在解释并验证该差异前，不把历史3/3直接当当前物理环境的新资格，也不调整控制器掩盖差异。本轮0新模型/物理/fresh。

## XML差异已定位，资产历史指纹边界保留

`historical_oracle_xml_migration.json`对12条保存XML逐字比较：仅将旧仓库前缀替换为当前前缀即可完全相等，没有其他XML差异。96份唯一mesh/texture当前文件已实际指纹，但015历史389输入没有其资产字节SHA覆盖；故XML语义路径迁移解释成立，历史资产字节同一性仍不以推断代替。为有限闭合该实际环境边界，登记[062匹配oracle正控](../plans/v0.0.1/062_matched_oracle_requalification_20261025.md)，仅三条CPU/1800步、0模型/fresh，保持061控制参数并做完整独立审查；当前尚未启动。

## 匹配oracle参考空间覆盖描述

062当前环境正控完整独立资格通过后，实际离线比较061各自600个已执行desired TCP位置与对应task的062正确参考600位置，保存`oracle_spatial_coverage_description.json`，读取前后SHA复核通过。使用双向最近邻空间距离，忽略时间、朝向、开度及接触，不要求模型模仿唯一专家路径；2cm仅描述阈值，不作成功门槛或在线触发。

成功task0/seed42/raw覆盖94.2%的oracle记录位置至2cm内；同seed候选仅1.3%，oracle到最近候选位置RMS约69.5mm。wine两seed候选该RMS约175.7/167.6mm，cream候选约113.5/131.0mm。末尾保持会重复加权位置，且任务可能存在其他成功路径，因此这些数值不能证明唯一根因；但与低跟踪误差却物体静止的证据共同表明，继续仅改善自身参考tracking不足以覆盖当前任务缺口。0新模型/物理/fresh。

已有150训练锚点覆盖报告表明训练并非完全没有闭合/释放事件（每任务50锚点），不能把失败直接归因于缺少任何阶段标签；接下来应按参考路径/当前状态偏移检查训练与闭环分布，而非重新重复已完成的全局正则化容量扫描。

## 闭环当前机器人状态与训练锚点距离

新增`scripts/observe_wam_anchor_state_coverage.py`通过Ruff并实际执行，读取全部150注册锚点的当前source_pose/opening（不消费future标签行）及12条每8步查询前的当前机器人状态，共900查询。`anchor_state_coverage_description.json`绑定源码、轨迹和完整日志，消费前后SHA不变。每任务只对其50锚点求最近位置，附同一邻居旋转/开度误差；位置距离不包含场景/物体，不是已标定OOD或成功判据。

raw/seed43三任务最近训练位置距离中位数分别0.369/0.441/0.403m，而seed42约0.049/0.056/0.029m；候选seed43约0.243/0.113/0.152m，task0候选最大0.917m。成功raw/seed42同样不是所有状态都紧贴训练锚点，不能据距离设在线成功条件。证据显示显著闭环分布偏移，且训练存在阶段标签并不足以覆盖漂移状态；下一步应检查偏移出现之前的生成参考与已有D2聚合覆盖，避免仅重复同150锚点的拟合。

## D2覆盖缺口已实证：去噪分布不等于物理闭环分布

实际核对D2 current_manifest与原prepared的全部150个anchor：150/150当前image_ref完全一致，D2虽增加300条采样/自身去噪历史，但新增物理状态锚点图像为0。`d2_physical_coverage_gap.json`绑定manifest/prepared/当前状态诊断读取前后SHA，并保存所有案例首次跨离线描述10cm距离的查询（该阈值不进入在线策略）。这不是否定D2数值资格，而是明确它没有采集闭环机器人漂移后的新状态分布。

seed43/raw三任务首次跨10cm在物理step112/96/128；它们初始最近训练位置均约5–10mm，说明不能只用初始状态接近训练锚点推断后续覆盖。候选也在多个任务跨出。下一步应针对仅训练split的可控状态扰动建立实际恢复参考/图像及前向执行资格，避免继续在相同150张图像上增加去噪轮次。训练恢复标签必须来自该扰动状态实际可执行的轨迹，不能把原demo未来端点未经验证地当作偏移状态的真实可达未来；开发/fresh轨迹不得拿来拟合。先做CPU小规模可达性资格，再固定后续数据和模型候选；总目标不缩减。
