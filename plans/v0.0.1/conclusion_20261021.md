# WAM稳定成功阶段结论 — 2026-10-21（Asia/Shanghai）

状态：实施中，稳定成功目标尚未达标。上一轮完整结论见conclusion_20261020.md，不覆盖其历史原字节。

最新完整结果：[027训练集尺度readout](../../reports/wam_train_scale_readout_20261021.md)**通过固定teacher门槛**，开发validation velocity MSE总体改善57.45%，sigma=.2/.5/.8分别为原模型的.3763/.6063/.9422，相对原026 bias额外改善17.82%。一个实际candidate fit及一个独立AugLS完成，mu/s差0、W最大差4.79e-15，bias预测逐位复现；两CPUchild实际wait0/reap、1256项SHA重核通过。全仓582 passed/1 integration deselected、Ruff331文件通过。它仅获得新路径GPU及完整生成资格研究的条件，实际`eligible_for_sampling=False`，0新WAM前向/完整生成/state评分/物理/fresh，尚无任务收益证明。下一步另登记GPU6数值资格、3zero完整复现和96成对state生成；原全部state与fresh24稳定门槛不变，025/026失败保持。下文027“尚未fit”为历史阶段记录。

[028实际state生成](028_scaled_readout_state_20261021.md)已实现并通过source冻结QA614 passed/1 integration deselected、Ruff341文件及96源/快照独立准备。GPU6实际20场景/10原语精度资格与独立bits审计通过，toy子进程actualwait0/reap、显存0；实际已启动唯一WAM worker，先3zero完整identity后原96paired/3465 denoise，当前初始化及全组结果仍待核验。完全冻结027参数，原027samplingFalse保持，由新028精度资格另授生成诊断；新fit/参数重建/物理/fresh0。尚无新state/task质量结论，稳定目标不变。

028目前已独立验证3zero完整raw/normalized exact及首4项validation的两对真实初态和7条实际sigma trace（245委托）。这是明确partial的运行协议证据，尚未读取future state标签或评分；原GPU6 worker与owner仍live，继续同一运行至完整99/3465及退出/资源释放后再独立判state门槛，未重复模型或修改冻结源。不能据teacher通过或zero复现宣布稳定成功。

最新[026 flow时间readout](../../reports/wam_flow_consistent_readout_20261021.md)已完整完成纯合成GPU6精度资格、两组固定CPU fit及独立AugLS重建：开发validation teacher速度MSE总体改善49.79%，但sigma=.8退化13.23%，超过固定5%上限，gate False。量化预测逐值相同、参数最大差4.705e-15；实际sampling dryrun在模型加载前拒绝，0新WAM前向/完整生成/state评分/物理/fresh。新运算路径15组实际sigma CPU/CUDA全128位一致，owned正常退出/显存0；它只证明该计算实现一致，不是task成功。完整源码冻结QA546 passed/1 integration deselected、321文件Ruff通过。原025失败保持；下一项先只读诊断特征尺度与sigma残差结构，再登记新干预，fresh24稳定门槛不变。以下旧阶段的“正在实现/尚未fit”均为历史记录，以本段完整结果及正式报告为准。

纯统计支持继续检查hidden尺度与正则prior的匹配，但没有证明该因素足以解释任务失败。[027训练集尺度readout](027_train_scale_readout_20261021.md)已另登记：仅train计算中心和整体L2 RMS，保留原时间函数/监督/权重/λ及质量门槛，一个新fit加一个独立重建，原026bias复用不重拟合。正在实现，真实fit尚未执行；不同时增加三个sigma头。本项无GPU/WAM生成或物理授权，后续先有teacher证据再单独验证新路径的实际GPU运算和完整state生成，稳定目标不变。

最新完整物理结论：[020/021同端点因果集成](../../reports/wam_temporal_equal_20261021.md)三任务全核验，ever2/3、terminal/stable1/3（仅cream，尾303端点/15.15秒）。原raw8/seed42也是stable1/3，但仅bowl，不能拼成两任务成功；该候选未建立多任务收益，不promote。225请求/1800步/225HTTP、14128输入SHA与全部owned正常释放通过，原task2零步启动故障不算任务失败、4次服务初始化与3评分轨迹明确分开。其集成参考TCP误差45–162mm，不能把失败全归checkpoint；正确参考3/3只证明已验证运动范围内的执行链可用。

022已完整96次state请求并独立核验：48份基线full128精确复现、48对实际caption仅command变化，全部4个owned身份正常释放。扩展文本的H8/H16位置24.384→24.371mm、44.623→46.298mm，H40位置85.521→88.331mm，未通过组件/10%改善/episode门槛，不推广。不能将短任务文本认定为主因。完整证据SHA `010e404aa1df40b2daaaeab0e93675e51029f8c1503e32cf11fd446bcc40c456`。

[023局部生成参考校正](../../reports/wam_generated_reference_calibration_20261021.md)固定19组fit及206输入的独立增广LS重算已完成。H8/H16共同状态分数改善11.07%/14.36%，且优于四对照，说明存在部分可利用结构；但H8开度、H16/H40位置退化、仅4/6 episode双短窗口不退化，未通过固定推广门槛。0新推理/物理/fresh，不部署第19组参数、不改变失败门槛，不能将校正收益当原checkpoint成绩。完整独立证据SHA `224d60eef89a4b19361de2bad0d50d8dac7a6e1dfd8e3c0d53e2bebc174662ba`。

024入口f8c2e3a、独立审计e7691cf及自有进程入口8a501df已提交。最新完整CPU验证433 passed/1外部integration deselected，Ruff check与292文件format通过；收据SHA `dfa25ec0aa92ae549deb07bdd5cd5d3d956ea9e27d461acb8d957ec570b0fd58`。真实future准备SHA `7a412c0fe6c1cbc1760e788dd065627ffd16e5acc84b3c518af5a6c1cdad804b`、两方独立CPU准入SHA `edcccddcd6620bf57711416117127660495e222357cd70ddb5ba476185b58beb`均通过，当前27次模型前向尚未执行。上述运行均已释放；本次收尾现场八卡均0MiB。后续仍只允许物理4–7。新登记[024当前训练—推理数值桥接](024_current_training_inference_numerics_20261021.md)：GPU6、8400/H40/mask10/正确三任务，真实未来RGB/action下published训练目标、task-only训练目标与同xt在线conditional共9组三路/27次denoise；0完整采样/物理。模型专项待实际执行。它补具体联合覆盖缺口，不替代任务验收。以下“运行中/未fit/旧全仓数字”为历史进度，以上完整结果为准。

最新进度：020前两项已完成完整独立核验，bowl仅34端点短暂成功、末端未保持，wine未成功，均未达稳定。021第三项零步故障接续已在GPU6/9946正常运行，新四身份与ready/go门禁现场通过；尚无全组完成率。[022](022_static_command_state_20261021.md)另行检查当前8400是否受简短任务与训练扩写command差异影响，仅96次state请求预算/0物理；启动前已纠正空task_plan占位的条件隔离问题，正在核全部48对真实CPU预处理。它不改变020/021，不提供示范阶段信息，不能预先宣布模型或新策略成功。最新完整CPU基础检查384 passed/1外部integration deselected、Ruff277文件通过，后续新增定向测试另列记录，未冒称新全仓数字。

当前020因果参考集成已进入WAM仿真：原task0/1正常执行；task2在创建环境和生成之前因进程身份检查异常退出，0物理步/0生成，属于基础设施故障，不能算任务失败。原错误和两项实际八身份partial快照完整保留，不伪称全组三项完成。已登记[021显式接续](021_temporal_zero_step_continuation_20261021.md)，只用新入口补未执行task2，GPU6/9946/新输出，全部效果参数与物理函数不变。新握手和精确清理先验证再启动；实际服务初始化累计4次，效果分母仍三旧任务/1800步/225生成。稳定成功与fresh验收尚未达成。

020正确参考资格的补充已完成：物理门槛全部通过，原strict False仍保留；仅225个非控制标签float1.0/int1差异获得独立显式旁证，不改变控制或评分。[资格说明](../../reports/wam_temporal_execution_qualification_20261021.md)。下文“待sidecar/未启动020 WAM”属于历史阶段状态，以本段和记录文档最新条目为准。

最新完整结果：[017三条1200步raw8](../../reports/wam_long_raw_20261021.md)已全部独立核验，稳定仍1/3。bowl从16.95秒起连续保持到60秒，尾862端点/43.1秒；wine/cream追加600步仍未成功，后半TCP跟踪仅1.880/1.903mm、SO3 .010904/.010629rad，0饱和。这有限排除了两条后缀“只差更多时间或更准执行”的解释，不独占归因所有早期失败。全部450请求/3600端点/12历史身份与正常释放通过，原600结果保持。下一项[019](019_raw8_seed43_replication_20261021.md)登记统一三旧任务raw8、seed43、原600步，3/1800/225预算，复用015真实同协议43资格和未改执行器；未打开fresh。

[018旋转漂移诊断](../../reports/wam_rotation_drift_diagnostic_20261021.md)固定19次拟合/六整episode留出及独立重算全部完成，但推广判定为False。H8/H16/H40旋转RMSE从.090896/.182583/.470659降至.081348/.143087/.320673rad；短窗口仍差于hold、历史外推及独立noWAM/shuffle对照。没有据此重调参数或追加模型、物理、fresh。下文早期“018未拟合”和“017运行中”是保留的时间记录。最近全仓289 passed / 1外部integration deselected、Ruff243文件通过，验证收据SHA `399ebbb6f28ab4399712e0fb53eb887328f492fb90c638d10007e7148465961d`；019新辅助入口另作针对性检查，尚未计入该全仓数字。

019已完整独立核验，seed43的ever/terminal/stable全0/3；原raw8三旧场景的seed42为1/3，seed43为0/3，未建立重复性，不挑成功seed。其3/1800/225与全部12owned退出通过，[正式报告](../../reports/wam_raw8_seed43_20261021.md)。[020因果参考集成](020_causal_temporal_ensemble_20261021.md)尝试消费历史预测对同一未来时刻的运动；等权平均改变参考，不能认证原始checkpoint。实际三条正确参考各600步均成功，1800共同row/控制canonical精确，原物理门槛全部满足；原审计额外比较chunk标签时因float1.0与int1的表示差而qualified=False，原False保留，暂不启动020 WAM，待独立补充只核此元数据边界。预留fresh保持未打开。

最新全仓检查已更新为335 passed / 1外部integration deselected、Ruff261文件通过，收据SHA `da32403e0fd58240bf5bb5f06bb586208896fccf8e826f8571d8852ec0cb050a`。020完整执行函数仅11处明确参考/来源/日志差异，其余初始化/控制/夹爪/物理/评分AST保持；旧007缓存1800目标全走数值identity且逐canonical相同。它是启动新物理资格的准备证据，不能代替实际资格或任务成功。

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


024当前8400/正确三任务task-only/H40/mask10专项现完整通过：27个实际前向及18组真实future训练loss独立重算，匹配训练/推理conditional的有效velocity逐值相同。实际wait0、owned reaped、GPU6显存0及外部tracked源码身份未变均确认，见[正式报告](../../reports/wam_current_training_inference_numerics_20261021.md)。这补上当前覆盖的数值一致性证据；只检查三个已查看样本和三sigma，不认证完整历史训练、生成质量或任务稳定性。

下一干预[025冻结主体readout](025_frozen_wam_readout_20261021.md)已事前登记、正在完成实现，真实feature/fit/完整采样尚未执行：固定6训练EP/6开发验证EP、12,810参数LIBERO专用输出残差及bias对照，先量化teacher门槛，后条件96次完整state生成。原WAM冻结，不修改26其它domain共享的projection0；新adapter是模型适配，不把潜在收益记为原checkpoint或S2A转换有效。TorchFP64→BF16双舍入已按[数值补充](025_bfloat16_rounding_addendum_20261021.md)实现数学单次ties-even，指定GPU6的真实合成CPU/CUDA bitwise对照通过；这不是任务成功证据。当前稳定目标仍未达，预留新12初态×2seed性能继续未打开。


025完整实现与独立CPU准入现完成：最终冻结758ca4c、整仓499 passed/1外部integration deselected、Ruff305文件通过，source SHA检查前后相同。真实12旧EP/48anchor/144teacher/96条件采样准备已经独立核验，prepared SHA `202b81afa0a2fc39d803c74b9430fb3d03e2d0a15ef403b60c1268d1b2d5d658`；实际owner dryrun及错SHA负控通过。当前只是运行准入，尚无025 teacher拟合或生成质量结果，不能视为稳定任务门槛改善。


025实际features阶段已在唯一物理GPU6进程启动，先3份zero输出exact后144 teacher，预算249denoise；实际Python命令/parent/环境身份已核实。尚无完成或teacher门槛结论，2组fit及96条件完整采样仍未执行。


025实际features已完整3zero+144teacher/249denoise，原模型参数冻结，3zero全raw/normalized exact；模型actual wait0/reap0。原owner的10秒资源观测超时导致False，该失败不修改；当前GPU6显存0/computeapps空及两原birth消失已独立核实，具体延迟原因未知。按[资源观测补充](025_release_observation_addendum_20261021.md)以新隔离入口继续尚未执行的CPU正式全组审计/原2fit/条件96采样，不重复NN或放宽质量门槛。此刻尚无adapter teacher/state质量结果。


025本轮结束于teacher质量失败：[正式报告](../../reports/wam_frozen_readout_20261021.md)。249实际模型前向及2组固定fit、独立AugLS参数与全部量化评分已完整通过重算；留出开发EP总体标准化velocity MSE下降28.53%，但sigma=.8退化2.043倍，原保护门槛False。constant bias已解释大部分总体收益，readout只额外改善0.464%。原资源观测owner False保留，新隔离CPU接续正常完成，没有重复NN。实际sampling dryrun在模型加载前拒绝；96完整生成、物理/fresh均0。此负结果否定本固定适配方案的推广资格，未检验其state或任务成功率；下一方案须另登记，不修改025参数或阈值挽救原失败。稳定目标仍未达，fresh24验收继续未打开。


下一项[026 flow-consistent readout](026_flow_consistent_readout_20261021.md)已在新fit/NN前登记并开始实现：按实际UniPC公式delta_v=-delta_x0/actual_solver_sigma，保留原velocity监督/权重，不把BF16 xt量化误差偷换入clean标签，不用整数t反推sigma。仍只复用原144缓存、两固定fit；单一新prior研究，原025False保持。teacher通过后才3zero+96paired/3465次denoise，0物理/fresh，新模型阶段尚未执行。目标仍为完整多任务新初态稳定门槛，未因本轮负结果降低。


[028完整生成state](../../reports/wam_scaled_readout_state_20261021.md)现已结束并双重独立核验：3zero/96paired/3465denoise及模型/CPU正常wait/reap/资源释放全部通过。H8/H16 pooled joint改善10.64%/16.14%，但H16 opening、H40位置及seed42短窗保护失败，最终gate=False；EP6/6与seed43改善不能抵消这些失败。原027 teacher57.45%改善只部分传递，不能认证稳定任务能力；不追加本候选物理/fresh，下一项须另登记。独立来源peer及完整图表已保存，模型只用了物理GPU6，原96源与163权重stat不变。后续优先有限比较同actual sigma下的自由轨迹与真future加噪输入，而非在已失败的028上改阈值或重拟合。


[029自由轨迹诊断](029_free_trajectory_diagnostic_20261021.md)在028完整失败后另行登记：同六开发EP/两seed/三个实际solver节点、不拟合，目的是量化true-future noising与自由轨迹的差异；网络时间和actual sigma分开匹配。实现及13项独立checker回归包含真实入口顺序正负控，全仓649 passed/1integration deselected、Ruff353文件；111源/快照和2023安全输入已复核。GPU6的20实际算子场景各CPU/CUDA预定gold通过，跨backend product/x0有4/20不同，属于已登记数值顺序，未当模型错误。新模型的48raw+144teacher/1824预算已在唯一GPU6进程启动，首raw已full128精确复现；完整诊断与任务收益尚无结论，0新fit/物理/fresh，稳定目标继续进行。


029追加独立复核：20场景实际数值与source/lifecycle完整通过，首2raw/6point的精确复现及CUDA转换通过且明确partial。唯一采集进程仍在运行，尚未形成完整轨迹诊断，不据此认证模型质量、拟合资格或任务成功。最新main冲突复核无待解决项；稳定成功目标及fresh24门槛不变。


029现完整：[正式报告](../../reports/wam_free_trajectory_diagnostic_20261021.md)。48raw精确复现、144+144点和1824调用独立通过，模型/CPU正常退出、GPU6释放；额外2426SHA/7800组重核及五图/PDF已完成。固定027的teacher收益明显依赖seed：42改善55.06%、43反退51.42%；raw自由轨迹actualclean反事实总体+1.60%，三个sigma/两个seed均退化。原027 teacher通过不等于一般噪声或自由轨迹有效；本结果不唯一认定video/hidden/噪声因果。0新fit/物理/fresh，028False保持。下一阶段采用来源明确的同步多EP训练数据、与评测seed隔离的多样噪声、显式当前clean估计及自由上下文监督，并在fit前固定有限任务验证；最终仍须原fresh24稳定门槛。
