# WAM稳定成功阶段结论 — 2026-10-21（Asia/Shanghai）

状态：实施中，稳定成功目标尚未达标。上一轮完整结论见conclusion_20261020.md，不覆盖其历史原字节。

最新完成项：[018旋转漂移诊断](../../reports/wam_rotation_drift_diagnostic_20261021.md)固定19次拟合/六整episode留出及独立重算全部完成，但推广判定为False。H8/H16/H40旋转RMSE从.090896/.182583/.470659降至.081348/.143087/.320673rad；短窗口仍差于hold、历史外推及独立noWAM/shuffle对照。没有据此重调参数或追加模型、物理、fresh。017三条1200步raw8结果仍待完整核验；下文早期“018未拟合”是保留的时间记录。当前全仓289 passed / 1外部integration deselected、Ruff243文件通过，验证收据SHA `399ebbb6f28ab4399712e0fb53eb887328f492fb90c638d10007e7148465961d`。

本轮将分别验证新15400权重/采样、因果SE(3)跟踪与夹爪反馈，先在旧开发集合筛选并通过执行资格，再冻结到全新初态。沿用总体至少80%、各任务至少50%、末40控制端点稳定成功门槛；不将oracle、保存参考、单例成功或完成实验预算当成目标达成。

目前已完成的证据：

- 新23源run的15400已完成99文件/32,399,051,175字节SHA核验，确认是从旧8800继续的另一个run；LIBERO完整128维合同相同。旧8800服务固定样本完整128维输出精确复现，max_abs=0。
- 4模型设置（8800/15400×guidance1.5/1.0）的192次多阶段state诊断已完整结束。每配置48请求/6旧开发episode、22 distinct anchors、1778有效H40端点；9项位置/朝向/开度指标存在取舍，无唯一全优候选。H40位置/SO3/openMAE：8800g1.5为80.463mm/.584876rad/.139449，8800g1为82.414mm/.542640rad/.146364；15400g1.5为91.595mm/.596121rad/.100781，15400g1为89.913mm/.507355rad/.116766。hold为128.832mm/.309419rad/.255167。这是开发state诊断，不直接等同任务能力。
- 因果共同参考重定时已实现并通过7项几何/速度界/连续性测试。新统一执行器中，旧三task/init0/s42的oracle正控baseline与governor-only各3/3稳定成功（各600步），TCP误差约1–2mm；这是正确参考的资格证据，不是WAM成功。
- 第一版夹爪位置反馈在正确参考上仅2/3稳定，旧intent为3/3；cream回归，**不具资格，不提升为默认**。独立第二版GripperIntentHold保留原关闭时机、关闭至motor limit后0保持、确认释放，同样三旧场景3/3稳定并通过raw审计。两批12条/7200步；这是电机位置目标保持，没有力传感器证实握持力，也不是WAM成功。[资格报告](../../reports/gripper_tracking_qualification_20261021.md)。
- bounded旋转反解修正其自身SITE/BODY绝对goal混帧及float校验后，24条/4320真实物理步已完整。大阶跃位置21.783/25.379mm明显差于原response3.688/3.724mm；局部朝向收益伴随位置退化，**仍不提升为默认**。旧response的world-left误差对固定右乘offset不变，不能据此认定旧接口错误。
- 第一版governor有桥接自锁和开度意图插入问题。独立修正版在4条/800步真实合成参考试验中，重query后.70/.75s出桥、端点误差.041/.059mm；旧版5s内桥进度0。但物理TCP平台条件未满足，原observed_blockage及其资格gate仍false，**不能冒称广义堵塞资格通过**。支持的窄证据是zero-scale参考暂停后的有界新参考恢复。新组合还需chunked oracle正控。
- 恢复版统一执行器6条chunked oracle资格已完成：intent_hold与recovering_intent_hold各3/3，均600步、末40端点native连续成功。独立审计重建控制命令、共同参考时钟、三项速度界、实际motor目标及完整SHA，基本和完整oracle gate均为true；它不是WAM成功。恢复组合的admitted TCP pooled误差1.838mm，而未重定时nominal参考为12.856mm，两者口径不同，必须同时保留。
- 更早8400/8600追加96次state诊断已完整。guidance1.0相同48请求下H40位置/SO3/opening：8400为85.521mm/.479894rad/.114660，8600为103.871mm/.546207rad/.118672；8400 H8/H16三项均最低，H40与8800的位置存在取舍。642项SHA与独立积分核验通过，仍只属6历史episode/22 distinct anchors的开发state证据。[四阶段报告](../../reports/wam_earlier_state_screen_20261021.md)。8400因此追加同003四variant×三旧场景的12条/7200步，现完整核验：raw alpha1两组各0/3、alpha.5为0/3、alpha0为1/3稳定（cream，最后456端点/22.8s连续成功）；全批1/12，136请求，无错误，正常释放。不是原完整checkpoint或多任务稳定成功。[完整报告](../../reports/wam_earlier_recovery_20261021.md)。同权重新夹爪[006](006_earlier_position_hold_20261021.md)现12条/7200步/133请求全核验：raw两组各0/3，alpha.5末端1/3但稳定0/3，alpha0稳定1/3（cream尾486端点/24.3s）；整体1/12稳定，未提升多任务稳定率。[完整报告](../../reports/wam_earlier_position_20261021.md)。
- 两seed等权生成参考中点（0新推理/物理）的位置/开度小幅改善，H40位置约74.66–82.87mm，但SO3仍.491–.582rad，均差于hold .309rad。独立Slerp/GT/分组聚合核验通过，暂不提升为物理候选，不把改变推理策略的结果归于原单seed checkpoint。
- 最近全仓基础检查：Ruff check通过、243文件format通过，289 tests passed / 1 integration deselected（新增018生产13项、独立审计5项）。另有真实MuJoCo资格，不拿CPU模型测试代替物理成功。

首批24条真实WAM开发pilot已全部独立核验，合计14400物理步、无运行异常，ever/terminal/稳定均0/24，8个模型/执行方式cell各0/3。12条governor的7200步中3704步scale=0、4130步在bridge（二者可重叠）；这限制了模型参考消费。8800g1 bowl目标双指接触0→78端点、最高抬起31.4mm，但仍未完成。8800g1.5三个baseline的45张图、全部原始预测和1800步state-command与历史execute40逐值相同，max_abs=0，证明统一baseline未改变旧行为。完整快照：`outputs/wam_stability_20261021/pilot_audits/complete24_verified.json`，SHA `c57b13ef682ce22eb7115b255e0c4475c3faa42e8fc86c06632c2ea58f120211`；另有4组PNG/PDF及独立SHA复核。

固定两个checkpoint/guidance1.0的恢复+握持/朝向消融见[003计划](003_wam_recovery_20261021.md)，24条/14400步现已全部独立核验：ever2/24、terminal0/24、stable0/24，无运行/审计错误，完整1199输入SHA前后重验通过。两种raw alpha1方式共12条均未成功；alpha.5中8800 bowl仅step154/157两个非连续端点曾成功；alpha0的15400 bowl曾连续228端点/11.4s成功，随后丢失。alpha0丢弃生成朝向、每个chunk保持当前实测anchor，重新观测及bridge仍可改变朝向；它只说明位置/夹爪与该保持策略的组合能力，不能认证完整WAM。两服务GPU5/7正常释放。[正式报告及图表](../../reports/wam_recovery_20261021.md)。**新初态验收尚未启动，稳定目标仍进行中。**

新的直接失败证据需要分开归因：15400恢复bowl有端点TCP误差仅2.529mm、但参考横向距bowl body origin206.84mm且低开度，说明该次生成参考自身偏离目标；现有intent_hold又会在reference仍.922时让实际夹爪闭合到.148，不能将这种早闭合全归于模型。只读几何/消费/真实命令与896项SHA保存在`reference_geometry_diagnosis/`。独立[004位置/闭合保持](004_gripper_position_hold_20261021.md)9条oracle/5400步已全部稳定：旧v2基线三条逐值复现、新position_hold无gov与加恢复各3/3，TCP1.30–2.06mm、SO3 .00259–.00679rad。新模块不从task/contact生成控制。主核验与5400前后qpos/motor开度及5391连续关系补核均通过；root再次重核434输入SHA后，已于19:41:56UTC在物理GPU5/7启动[005固定24WAM对照](005_wam_position_hold_20261021.md)。不能将9条资格当新WAM成功，新24现全部完整核验：14400步、ever2/24、terminal0/24、stable0/24；raw两种方式12条均未成功，alpha.5六条也均未成功，alpha0只有15400 bowl短暂成功226端点（最长连续7.4s）和cream一个端点。全部首query图像/full raw/normalized/anchor/初态与003对应逐值相同。共同首query前缀中的错误高开度完全闭合锁存11→0，但稳定任务率没有提升；不能将后续不同闭环输入称相同参考因果对照。[完整报告](../../reports/wam_position_hold_20261021.md)。

[008三参数参考校正](008_reference_gain_diagnostic_20261021.md)只读缓存、0新推理/物理，按6整episode留出，44去重queries且全方法共同1398有效端点；目前位置仅数毫米收益（约78–79mm残留），朝向仍差于hold、两组开度反退，不推广或追加该校正器物理。首份opening MAE汇总signed/abs错误已保留原证据并修正到新目录，拟合和逐query误差精确未变，独立54模型/18整episode folds/132held queries/924 query-method结果已全部重算，497输入SHA再核一致，[正式诊断](../../reports/wam_reference_gain_diagnostic_20261021.md)；不promote。该校正改变模型参考，不能认证原checkpoint。[007短重新观测](007_short_replan_qualification_20261021.md)12条CPU资格已完整独立通过，7200步、12/12末40稳定，TCP1.339–2.459mm/SO3 .002595–.007153rad，351输入SHA再核；17自有PID正常退出，正确参考附近无整query零进度。[正式资格报告](../../reports/short_replan_qualification_20261021.md)。这只允许后续考虑短间隔WAM预算，不是模型成功。

12个新物理初态已仅按metadata/seed42预留并独立核验，未进行任何候选性能评测：每task4、12初态SHA唯一且与旧5real fixture无重叠、expert/model XML/initial与HDF来源一致、重现抽样完全相同。除9完整命令及1严格后缀匹配外，另2历史episode只有唯一全长XYZ+gripper部分匹配、旋转仍差约.0075；保守将这两个疑似来源也排除，不升级为完整控制/历史物理同步认证。源旋转、图像/pose/初态同步及预训练可见性未知继续保留。此为输入准备，不是新初态任务验收。

新23源并非已知最佳checkpoint：validation关闭、15400是首次冻结时的latest complete；20:17UTC再查已有同run完整15600，[009](009_latest_state_screen_20261021.md)的48次state-only对照现全部核验通过、798输入SHA与正常release绑定。H8/H16/H40位置mm/SO3rad/opening分别29.185/.104531/.075429、52.861/.221368/.125687、98.696/.563030/.140309，九项均差于8400及15400。仅六旧episode证据，不作全局训练退化或任务能力排名；不因为latest追加物理。[完整报告](../../reports/wam_latest_state_screen_20261021.md)。新加入源是FastUMI-eef，RoboMIND2旧run已有。LIBERO eef/joint合计名义packed-microbatch份额由200/10194=1.961938%降至10/10000=.1%（约19.62倍下降）；不是实际历史消费、token/loss份额或失败因果证明。两run当前LIBERO条件/clip/归一化审查未发现新run专属推理错接，完整数值一致性、历史消费字节和上游图像/pose物理同步仍未证实。

下一步已在完整005/006/009结果之后预注册[010短重观测](010_short_replan_wam_20261021.md)：同8400、无gov、每8/16步重query、raw与alpha0全部三旧任务，12条/7200步/最多678请求，物理GPU4/5；不改变控制器或按任务拼接方案。独立[011采样步数诊断](011_sampler_steps_state_20261021.md)固定8400/35、70、140各48次state请求，GPU6，0物理；先全部48份traced35输出exact准入，再以实际sampler入口SHA配对比较70/140。010已在GPU4/5运行：interval8/raw bowl已完整600步/75请求核验，原生尾262端点/13.1s连续成功；interval8/raw wine及cream完整失败，interval16 raw/alpha0各三个任务完整失败，8步alpha0三条完整0/3，独立补核12/7200/678完成，不报多任务或新初态稳定。011在GPU6的48真实35已全部exact、全144个实际noise/condition配对已核，70/140九项均差35，未通过推广门槛；[正式报告](../../reports/wam_sampler_steps_state_20261021.md)。另[012](012_guidance_state_20261021.md)单guidance3/8400的48state-only已完整，不满足六项/10%门槛，不promote，[报告](../../reports/wam_guidance_state_20261021.md)。[013](013_attention_backend_state_20261021.md)在012释放与011完整35准入后，48个actual-noise配对的flash3_video state对照已完整，预定gate失败，不追加该方向物理，[报告](../../reports/wam_backend_state_20261021.md)；不改010/011。fresh性能继续未打开。

仅允许物理GPU4–7，CPU/OSMesa禁CUDA，外部Cosmos/R1/R2只读。方案与决策见[002计划](002_wam_stability_20261021.md)及[实施记录](record_20261021.md)。上述数字为阶段进度，随完整结果追加更新；上一轮20261020报告/封存证据不覆盖。

014新双种子审计接口的27条旧日志精确回归已完成，原checker保留；015真实seed43正确参考资格3/3完整通过，和42的1800步逐值相同，不算WAM能力。[016按跟踪误差重观测](016_tracking_replan_20261021.md)已预注册：raw参考原样，正常最多执行40步，误差持续超界才在至少8步后重观测，先3条oracle资格、条件3条WAM旧场景。该方向已实现并通过11项CPU因果/边界测试，独立正确参考资格3/3已完成，1800共享步和旧40步方案逐值相同；首配置漏oracle路径在0物理前失败，已保留并用v2修正，不改变算法/门槛或预算。未宣称能解决不断截断关闭后缀的问题。

服务退出可观测性发现独立缺项：013的48原始输出完整、受控SIGTERM与PID退出保存，但actual Uvicorn重发信号绕过Python finally，两个仅finally写的内存终态收据缺失。已CPU因果复现；保留严格拒绝，另核逐request/trace/raw与外部关闭证据，不补造ledger或把退出旁证缺失当模型质量失败。011已用相同窄旁证独立完成144请求质量核验，两个内部终态ledger仍明确未知。

016自适应完整WAM三任务稳定/曾成功均0/3：bowl26query/TCP25.117mm，wine40query/TCP24.892mm，cream31query/TCP25.041mm，共1800步/97请求。实际触发与全部case逐值审计通过，服务97个HTTP请求及完整预算、正常退出均独立核实；未胜过raw8，不推广。[完整报告](../../reports/wam_tracking_replan_20261021.md)。[017](017_long_raw_development_20261021.md)另预注册三任务统一60秒/1200步原始raw8诊断（native3/oracle3资格先行，共六条/7200步；条件WAM3/3600步/450query），同时报告600截点，不追溯改原600失败。新隔离环境1300上限/ignore_doneFalse、无额外pre-roll，actual初始timestep按0现场核。017 native三条3600步通过、oracle三条3600步也全部稳定（TCP1.005–1.300mm/SO3 .001835–.004769rad）。实际控制前600逐值相同；原严格all-row门禁False明确保留，因为599端点四个不参与控制的诊断值有≤2.22e-16舍入。独立分支/消费者证明及原容差重构后，新physical-core资格通过；[完整说明](../../reports/wam_long_qualification_20261021.md)。三条raw8 WAM已22:49:15UTC在GPU5/6/7启动，完整结果尚待核验，fresh未打开。时长延长有可能检验完成较慢/持续丢失，不预设改善。

010全部12条实际物理及最终独立补核已结束。原interval8 coordinator在收尾重查**非本轮15400**历史权重时发现外部98项文件消失，原错误与release均保留；本轮实际8400的163项stat及旧3477项源SHA仍一致。此异常不计任务失败、不伪造原complete成功；新独立入口完整核验12/7200步/678请求、6993源SHA、本轮8400全163stat、全部owned释放与精确HTTP/generation预算；raw8唯一bowl正例，另三组均0/3。第二审查方完整CPU复跑与7200条success逐值复算一致。


017现已完成前600可比性里程碑：三任务1800实际端点及225完整query图像/raw/normalized与旧raw8全部canonical精确，30秒仍1/3（仅bowl），1200结果未定。并行[018角漂移缓存诊断](018_rotation_drift_diagnostic_20261021.md)只研究六旧episode的三参数世界旋转偏置，19fit/0模型/0物理，需整episode留出优于无WAM/打乱/保持/历史外推才有进一步验证资格；模块6项几何测试已通过，真实拟合尚未开始。它改变模型参考，不能认证原checkpoint，且不改正在跑的017。
