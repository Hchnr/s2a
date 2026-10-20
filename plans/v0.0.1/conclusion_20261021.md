# WAM稳定成功阶段结论 — 2026-10-21（Asia/Shanghai）

状态：实施中，稳定成功目标尚未达标。上一轮完整结论见conclusion_20261020.md，不覆盖其历史原字节。

本轮将分别验证新15400权重/采样、因果SE(3)跟踪与夹爪反馈，先在旧开发集合筛选并通过执行资格，再冻结到全新初态。沿用总体至少80%、各任务至少50%、末40控制端点稳定成功门槛；不将oracle、保存参考、单例成功或完成实验预算当成目标达成。

目前已完成的证据：

- 新23源run的15400已完成99文件/32,399,051,175字节SHA核验，确认是从旧8800继续的另一个run；LIBERO完整128维合同相同。旧8800服务固定样本完整128维输出精确复现，max_abs=0。
- 4模型设置（8800/15400×guidance1.5/1.0）的192次多阶段state诊断已完整结束。每配置48请求/6旧开发episode、22 distinct anchors、1778有效H40端点；9项位置/朝向/开度指标存在取舍，无唯一全优候选。H40位置/SO3/openMAE：8800g1.5为80.463mm/.584876rad/.139449，8800g1为82.414mm/.542640rad/.146364；15400g1.5为91.595mm/.596121rad/.100781，15400g1为89.913mm/.507355rad/.116766。hold为128.832mm/.309419rad/.255167。这是开发state诊断，不直接等同任务能力。
- 因果共同参考重定时已实现并通过7项几何/速度界/连续性测试。新统一执行器中，旧三task/init0/s42的oracle正控baseline与governor-only各3/3稳定成功（各600步），TCP误差约1–2mm；这是正确参考的资格证据，不是WAM成功。
- 第一版夹爪位置反馈在正确参考上仅2/3稳定，旧intent为3/3；cream回归，**不具资格，不提升为默认**。独立第二版GripperIntentHold保留原关闭时机、关闭至motor limit后0保持、确认释放，同样三旧场景3/3稳定并通过raw审计。两批12条/7200步；这是电机位置目标保持，没有力传感器证实握持力，也不是WAM成功。[资格报告](../../reports/gripper_tracking_qualification_20261021.md)。
- bounded旋转反解修正其自身SITE/BODY绝对goal混帧及float校验后，24条/4320真实物理步已完整。大阶跃位置21.783/25.379mm明显差于原response3.688/3.724mm；局部朝向收益伴随位置退化，**仍不提升为默认**。旧response的world-left误差对固定右乘offset不变，不能据此认定旧接口错误。
- 第一版governor有桥接自锁和开度意图插入问题。独立修正版在4条/800步真实合成参考试验中，重query后.70/.75s出桥、端点误差.041/.059mm；旧版5s内桥进度0。但物理TCP平台条件未满足，原observed_blockage及其资格gate仍false，**不能冒称广义堵塞资格通过**。支持的窄证据是zero-scale参考暂停后的有界新参考恢复。新组合还需chunked oracle正控。
- 恢复版统一执行器6条chunked oracle资格已完成：intent_hold与recovering_intent_hold各3/3，均600步、末40端点native连续成功。独立审计重建控制命令、共同参考时钟、三项速度界、实际motor目标及完整SHA，基本和完整oracle gate均为true；它不是WAM成功。恢复组合的admitted TCP pooled误差1.838mm，而未重定时nominal参考为12.856mm，两者口径不同，必须同时保留。
- 更早8400/8600追加96次state诊断已完整。guidance1.0相同48请求下H40位置/SO3/opening：8400为85.521mm/.479894rad/.114660，8600为103.871mm/.546207rad/.118672；8400 H8/H16三项均最低，H40与8800的位置存在取舍。642项SHA与独立积分核验通过，仍只属6历史episode/22 distinct anchors的开发state证据。[四阶段报告](../../reports/wam_earlier_state_screen_20261021.md)。8400因此追加同003四variant×三旧场景的12条/7200步，现完整核验：raw alpha1两组各0/3、alpha.5为0/3、alpha0为1/3稳定（cream，最后456端点/22.8s连续成功）；全批1/12，136请求，无错误，正常释放。不是原完整checkpoint或多任务稳定成功。[完整报告](../../reports/wam_earlier_recovery_20261021.md)。同权重的新夹爪映射另12条已按[006条件协议](006_earlier_position_hold_20261021.md)在GPU4启动，须与原12完整配对，不选择唯一成功任务。
- 两seed等权生成参考中点（0新推理/物理）的位置/开度小幅改善，H40位置约74.66–82.87mm，但SO3仍.491–.582rad，均差于hold .309rad。独立Slerp/GT/分组聚合核验通过，暂不提升为物理候选，不把改变推理策略的结果归于原单seed checkpoint。
- 最近全仓基础检查：Ruff check通过、180文件format通过，193 tests passed / 1 integration deselected。另有真实MuJoCo资格，不拿CPU模型测试代替物理成功。

首批24条真实WAM开发pilot已全部独立核验，合计14400物理步、无运行异常，ever/terminal/稳定均0/24，8个模型/执行方式cell各0/3。12条governor的7200步中3704步scale=0、4130步在bridge（二者可重叠）；这限制了模型参考消费。8800g1 bowl目标双指接触0→78端点、最高抬起31.4mm，但仍未完成。8800g1.5三个baseline的45张图、全部原始预测和1800步state-command与历史execute40逐值相同，max_abs=0，证明统一baseline未改变旧行为。完整快照：`outputs/wam_stability_20261021/pilot_audits/complete24_verified.json`，SHA `c57b13ef682ce22eb7115b255e0c4475c3faa42e8fc86c06632c2ea58f120211`；另有4组PNG/PDF及独立SHA复核。

固定两个checkpoint/guidance1.0的恢复+握持/朝向消融见[003计划](003_wam_recovery_20261021.md)，24条/14400步现已全部独立核验：ever2/24、terminal0/24、stable0/24，无运行/审计错误，完整1199输入SHA前后重验通过。两种raw alpha1方式共12条均未成功；alpha.5中8800 bowl仅step154/157两个非连续端点曾成功；alpha0的15400 bowl曾连续228端点/11.4s成功，随后丢失。alpha0丢弃生成朝向、每个chunk保持当前实测anchor，重新观测及bridge仍可改变朝向；它只说明位置/夹爪与该保持策略的组合能力，不能认证完整WAM。两服务GPU5/7正常释放。[正式报告及图表](../../reports/wam_recovery_20261021.md)。**新初态验收尚未启动，稳定目标仍进行中。**

新的直接失败证据需要分开归因：15400恢复bowl有端点TCP误差仅2.529mm、但参考横向距bowl body origin206.84mm且低开度，说明该次生成参考自身偏离目标；现有intent_hold又会在reference仍.922时让实际夹爪闭合到.148，不能将这种早闭合全归于模型。只读几何/消费/真实命令与896项SHA保存在`reference_geometry_diagnosis/`。独立[004位置/闭合保持](004_gripper_position_hold_20261021.md)9条oracle/5400步已全部稳定：旧v2基线三条逐值复现、新position_hold无gov与加恢复各3/3，TCP1.30–2.06mm、SO3 .00259–.00679rad。新模块不从task/contact生成控制。主核验与5400前后qpos/motor开度及5391连续关系补核均通过；root再次重核434输入SHA后，已于19:41:56UTC在物理GPU5/7启动[005固定24WAM对照](005_wam_position_hold_20261021.md)。不能将9条资格当新WAM成功，新24结果尚待完整验证；前4条无gov（两权重bowl/wine）已完整独立审计、均未成功，首query图像及完整raw/normalized输出与003精确一致，其余在跑不计失败。

[008三参数参考校正](008_reference_gain_diagnostic_20261021.md)只读缓存、0新推理/物理，按6整episode留出，44去重queries且全方法共同1398有效端点；目前位置仅数毫米收益（约78–79mm残留），朝向仍差于hold、两组开度反退，不推广或追加该校正器物理。首份opening MAE汇总signed/abs错误已保留原证据并修正到新目录，拟合和逐query误差精确未变，完整独立重算进行中。该校正改变模型参考，不能认证原checkpoint。[007短重新观测](007_short_replan_qualification_20261021.md)12条CPU资格正在执行，8步的6条物理均stable，须全部独立通过才考虑新的WAM预算。

12个新物理初态已仅按metadata/seed42预留并独立核验，未进行任何候选性能评测：每task4、12初态SHA唯一且与旧5real fixture无重叠、expert/model XML/initial与HDF来源一致、重现抽样完全相同。除9完整命令及1严格后缀匹配外，另2历史episode只有唯一全长XYZ+gripper部分匹配、旋转仍差约.0075；保守将这两个疑似来源也排除，不升级为完整控制/历史物理同步认证。源旋转、图像/pose/初态同步及预训练可见性未知继续保留。此为输入准备，不是新初态任务验收。

新23源并非已知最佳checkpoint：validation关闭、15400仅latest complete；新加入源是FastUMI-eef，RoboMIND2旧run已有。LIBERO eef/joint合计名义packed-microbatch份额由200/10194=1.961938%降至10/10000=.1%（约19.62倍下降）；不是实际历史消费、token/loss份额或失败因果证明。两run当前LIBERO条件/clip/归一化审查未发现新run专属推理错接，完整数值一致性、历史消费字节和上游图像/pose物理同步仍未证实。

仅允许物理GPU4–7，CPU/OSMesa禁CUDA，外部Cosmos/R1/R2只读。方案与决策见[002计划](002_wam_stability_20261021.md)及[实施记录](record_20261021.md)。上述数字为阶段进度，随完整结果追加更新；上一轮20261020报告/封存证据不覆盖。
