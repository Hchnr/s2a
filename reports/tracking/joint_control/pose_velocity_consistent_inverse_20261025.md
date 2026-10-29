# 083 平移位置/速度一致性：纯转换与控制接入证明通过，未执行物理

按083预登记实现src/s2a/pose_velocity_inverse_osc.py：固定位置误差+dt²速度误差的名义PD一周期二状态目标，解析求goal delta，再保留原仿射范围和变化率约束；不扫描权重、不调整实际kp/kd，旋转仍081半补偿及原大角保护。desired_linear_velocity为显式输入，与当前measured_velocity分开，返回目标和名义预测速度，不混作实际观测。

reference_linear_velocity.py只用当前已经生成source位置端点和本次实际query anchor：内部中央差分、首端点anchor中央差分、尾端点后向差分、单端点anchor差分；保存mode/索引/可用mask，缺失不默默补未来数据。这是参考轨迹差分估计，不是native瞬时速度真值。

9项实际检查通过：无约束两状态目标梯度为零且不劣位置单目标代价；实际box/速率投影为约束最优；两种旋转尺度保持旧行为；首尾/单点/内部参考速度与未用生成尾部因果性；非法索引/间隔拒绝。position/velocity量纲使用一周期dt换算，未引入可调超参数。

pose_velocity_loaded_simulation.py在生成reference转换为source坐标后估计目标速度，保持当前twist观测与其区分；完整控制函数恢复AST对照证明仅改变arm转换器/目标速度与相应日志，夹爪路由/控制器/时钟/状态标签不变。该1项检查也通过，当前共10项；Ruff通过。尚无独立完整机械checker/执行入口，尚未进行新物理或拟合，不能据名义公式最优推断真实稳定任务收益。

082仍由原owner冻结执行，最终WAM开发/fresh门槛保持，必须其完整关闭后才能另行准入083。总目标未完成。

## 独立回放与逐步checker补充

独立arm回放以闭式阻尼响应和2×1最小二乘复算命令，在欠阻尼/临界阻尼/过阻尼的连续状态测试中通过3项检查。新pose_velocity_loaded_acceptance_checker.py独立重建source参考速度、模式/索引/布尔mask，核验实际命令及名义下一周期线速度；不导入生产参考差分函数。新增2项检查覆盖全部差分边界模式，以及完整verify_case AST还原后与082机械/夹爪审查一致。此前10项加本次5项，共15项通过，相关Ruff检查通过。该checker尚未对083物理日志执行；083尚无物理性能结果，执行入口与准入配置仍待完成。082原owner仍运行，未重启，最终结论待完整关闭。

执行准备：collect/verify/run_pose_velocity_loaded_control.py及configs/pose_velocity_loaded_control_20261025.json已接入；全部18条真实3.11解释器dry-run通过，保存outputs/pose_velocity_pipeline_preflight_20261025/result.json，0物理/模型调用。持久化owner要求082完整关闭凭据；新增close_translation_loaded_control.py用于082全18条/4907步、36阶段及owner实际wait/reap、来源指纹和对081的开度误差比较。083尚未启动物理。

准入强化：启动前逐项比较083与082全部profile相等、seed42、总4907步、固定位置/速度目标、生成参考差分定义及旋转0.5；新增close_pose_velocity_loaded_control.py保留完整机械/owner关闭核验和对立即前项082的开度误差比较。Ruff及导入检查通过，未启动物理。

## 083实际启动

082完整关闭后启动准入，20个检查进程全部通过：51项CPU测试、全部18条实际CLI dry-run及checker导入。来源快照与新namespace outputs/pose_velocity_loaded_control_20261025冻结；execution owner PID3638955/start_ticks583917716、首采集worker PID3638963/start_ticks583917753已从/proc交叉核实。CPU/OSMesa空CUDA/LD，0世界模型请求/新fit/fresh。原18条4907步与18/18门槛不变，尚无083完整物理性能结论，不进入WAM/fresh。

## 首条已核验结果与配对

case00完整采集/独立核验通过，稳定True、末尾80步成功，位置RMSE3.217mm、SO3 RMSE0.004225rad，原性能门槛通过。相同参考082位置1.404mm，因此新目标有精度代价；使用083保存的差分目标速度，对齐下一实际端点twist，采样速度RMSE083/082为0.019055/0.032960m/s，平移归一化命令增量RMS为0.114781/0.327972。这只是首条描述性配对，差分目标不是native瞬时速度真值，不能推断完整任务收益或唯一因果。只读脚本scripts/compare_pose_velocity_tracking.py与outputs/pose_velocity_tracking_pairs_20261025/verified_prefix_01.json保存数据；0新物理/拟合/模型/fresh，完整18条仍在原owner执行。

第二条case01独立核验通过：稳定True，末尾63步成功，位置2.008mm、SO3 0.009613rad，原性能门槛合格；082同条不稳定。现2条已核验均合格，不能代表剩余16条。只读完整配对保存outputs/pose_velocity_tracking_pairs_20261025/verified_prefix_02.json；原owner PID3638955/start_ticks583917716再次核实存活，继续冻结执行，无新增模型/拟合/fresh。

## case03原资格失败与只读抓持诊断

前三条均通过；case03独立机械/标签/图像核验通过，但任务从未成功，位置RMSE4.739mm、SO3 0.006197rad虽在门槛内，性能仍失败。因此083原18/18资格已不能通过，完整回归仍按冻结计划运行，不进入WAM/fresh。

只读比较081/082/083同case03对象离线真值和命令，保存outputs/pose_velocity_tracking_pairs_20261025/case03_grasp_snapshot.json（输入SHA前后核验）。083周期55对象最高抬升105.265mm，实际开度0.26436、参考0.31239，motor[-1,+1]处于关闭限位、命令0保持；周期60高度已落至13.390mm、实际开度0.02745、参考0.30806，仍同motor/保持路由。可观察到对象脱落与实际开度收缩，而非简单的提前发出打开命令；接触标志为False但该标志不证明完整接触力状态，不能据此单独断言夹持力或arm时序为唯一原因。对象真值仅用于事后诊断，未加入在线输入。0新物理/模型/拟合/fresh。

已核验前6条：case00/01/02/04/05达标，case03任务失败，合计5/6。新增case04位置4.148mm/SO3 0.007509rad，case05位置1.696mm/SO3 0.009913rad；全部独立机械/标签/图像审查通过。前6配对保存outputs/pose_velocity_tracking_pairs_20261025/verified_prefix_06.json。原owner身份PID3638955/start_ticks583917716仍存活，进入case06采集，未调整已冻结方案；原18/18资格失败与WAM/fresh关闭不变。

case06完整独立审查通过，任务稳定、末尾87步成功，位置RMSE3.8583mm、SO3 RMSE0.010555rad，原门槛通过。相同参考082任务稳定但位置10.475mm超限，083在这条同时满足任务与跟踪门槛。累计前7条6条合格，case03失败不变；原owner PID3638955/start_ticks583917716仍经/proc身份核实存活。剩余回归按原冻结配置继续，不进入WAM/fresh。

提交575d280后，原owner身份保持不变并继续推进至前10条独立核验完成，9/10合格。case07位置1.748mm/SO3 0.004657rad、case08位置2.839mm/SO3 0.005926rad均通过；case09稳定True、末尾92步成功，位置4.086mm/SO3 0.007484rad，修复了082同条不稳定现象。case03共同失败尚未解决，不能由9/10推断原18/18或WAM/fresh资格，总目标未完成。

case10完整独立核验通过但任务从未成功；位置RMSE7.951mm、SO3 0.014291rad在门槛内，仍性能失败。082同条稳定且达标，构成083新增任务退化。累计前11条9条合格、case03和10失败；前11条只读配对保存outputs/pose_velocity_tracking_pairs_20261025/verified_prefix_11.json。原owner仍存活，完整剩余7条继续执行；不得以名义两状态目标或局部采样速度改善代替实际任务成功。

case10只读放置诊断：082与083首次close均周期36、首次后续open均82、最高抬升均周期62且高度约108mm；单凭首次开闭时刻不能解释差异。周期82碗相对盘水平距离082/083为19.390/31.813mm，末端为19.912/33.985mm，083最终落点偏移更大。仅为离线对象真值描述，不作唯一因果结论或在线对象输入。来源SHA前后复核及选取周期明细保存outputs/pose_velocity_tracking_pairs_20261025/case10_release_snapshot.json，0新增物理/模型/拟合/fresh。

合并2287adf后原owner仍按相同PID/start_ticks存活，全部699份冻结来源指纹一致。前13条已独立核验10条合格；case12曾成功5步但terminal/stable均False，位置RMSE2.174mm、SO3 0.010278rad，构成对082的新增任务退化。失败case03/10/12，剩余5条继续冻结执行。原18/18资格失败与WAM/fresh关闭不变。

前15条独立核验完成，12/15合格。case13位置RMSE1.796mm/SO3 0.004139rad、case14位置2.931mm/SO3 0.006026rad，均稳定并达标。失败case03/10/12不变，剩余3条world_reference仍在原owner监督下执行；全量4907步和owner关闭尚未完成，原资格及最终目标均未达成。

前15条完整只读配对汇总保存outputs/pose_velocity_tracking_pairs_20261025/verified_prefix_15.json：083采样速度误差在14/15条低于082，平移命令增量RMS在15/15条更低；综合资格新增通过case01/06/09，同时新增失败case10/12，共同case03失败。由此只能确认描述性平滑/速度改善，不能声称任务可靠性已解决。最后3条仍执行中，完整关闭待完成。

## 完整关闭

18条4907步及逐条独立机械/标签/图像审查全部完成，36个采集/核验阶段和execution owner实际wait0/reap、无信号、来源不变核验通过。原输出目录regression_summary.json、closure_verified.json已保存。任务稳定15/18、综合资格15/18；失败case03/10/12。对082综合14/18有所改善，但任务稳定同为15/18且有新增退化，不满足原18/18资格，不采用进入WAM/fresh，总目标未完成。
