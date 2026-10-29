# S2A — State to Action

最新CPU机制结果：[完整约束残差修正](reports/constraint_residual_repair_20261029.md)通过131项专项QA、唯一QP及11响应生产/11响应独立复算。全步恢复全部安全与前缀优化余量，但仅保留49.488%原下降，未通过预登记50%条件，故不选中；其余缩短步仍安全失败。cursor9仍11.46492mm/.05732601rad、完整witness0，实际仍cursor8/task0/grasp0。2540来源核验关闭，不放宽原门槛或自动追加微步搜索；GPU继续禁用。

最新CPU结果：[保存3543的末周期修复](reports/terminal_only_tail_repair_20261029.md)完成155项专项QA、14响应生产和14响应完整独立复算。12候选中2个合格，选中末周期x减.001，terminal恢复至2.762279319e-5；保留cursor9微小改善，但其误差仍11.46520mm/.05732641rad超原caps，其他安全最小slack仍仅6.13467e-8。首次环境引用错误的零响应失败保留。完整witness0、实际cursor8、task/grasp0，不自动重开同一SQP；GPU禁用及原18/12/24门槛保持。

前项CPU结果：[从恢复候选重新开始完整H12 SQP](reports/restored_anchor_sqp_20261029.md)的889项native QA、生产搜索、3657响应完整独立复算及2526来源核验全部关闭。272条ray、47份anchor、55个QP和46次曲率更新核验通过；仍prefix9、完整witness0，55阶段后信赖域停滞。selected的cursor9仍11.46528mm/.05732691rad超原caps，实际仍cursor8、task/grasp0。GPU继续禁用，原18/12/24门槛及总目标保持。

运行时验证：[保存命令CPU回归](reports/native_runtime_replay_20261029.md)通过27项专项QA与实际准入，唯一真实重放完成377控制周期/9425积分步；378个完整端点、225内环样本的最大状态差异均0，actual368完整状态逐字节一致。固定控制器与llvmpipe软件OSMesa通过，2808来源不变；首次QA参数格式拒绝的零运行时失败保留。此前[v3 bootstrap](reports/native_runtime_bootstrap_v3_20261029.md)及其两次失败证据保持。本次只复现已执行分支，实际仍cursor8、task/grasp0，不授予新轨迹控制或历史二进制等价资格，GPU继续禁用。

当前CPU结果：[末周期平移恢复终端余量](reports/tail_translation_reserve_20261029.md)的14次生产响应、隔离v2完整独立复算及2485来源闭合均已完成。6/12候选合格，选中请求5（末周期y +.001），terminal余量恢复到7.733855175988188e-5并通过原经验门槛；前11周期、旋转goal及history/repeat的exact检查通过。完整H12 witness仍0，cursor9仍11.47494mm/.05737520rad，实际仍cursor8、task/grasp0。其他安全最小值仅3.267282409069594e-7，不能称全约束鲁棒内点。

原95源QA为766通过，新100源QA为796通过；v1 checker在14条复算后发生布尔数组比较异常的失败历史保留，原producer未重跑。三阶段实际合计42响应/12600内步/13107查询，新增物理/world-model/fresh0。后续[完整H12 SQP登记](plans/v0.0.1/restored_anchor_sqp_20261029.md)的首次完整响应exact门及生产结果见上文。GPU继续禁用，原18/12/24门槛与总目标保持。

前项CPU结果：[终端单行割线外推](reports/terminal_secant_probe_20261029.md)完成655项native QA、17调用及全量独立复算、2422来源闭合。两个未拟合混合方向均未达到误差减半门槛，最大新绝对残差约2.28e-6高于原终端裕量3.20e-9；修正不采用。15个探针均未保持旧达标前缀，完整witness0；后续终端恢复结果见上文。

当前CPU推进：[实际接纳点SQP](reports/accepted_point_sqp_20261029.md)搜索与完整独立复算均正常关闭：1919响应/575700数值内步、118条ray、25份anchor、27个QP和24次曲率更新全部核验，严格分类一致；仍prefix9、完整witness0。cursor9仍11.47mm/.05738rad超原caps。[固定方向导数诊断](reports/directional_derivative_consistency_20261029.md)保留v1零响应准入失败；隔离v2通过529测试及实际dry，77完整调用/76唯一命令正常关闭、witness0，全量独立复算完成77响应后因旧stencil无容差边界守卫失败；隔离check-only v3通过573测试、同77响应全量复核及2381来源闭合；全固定步长分析确认旧terminal线性预测偏差，小步长差分未显示收敛，尚无新控制解。保持原18/12/24任务门槛，实际仍cursor8、task/grasp0，GPU继续禁用。

当前CPU进度：[固定前缀优化余量](reports/prefix_optimization_reserve_20261029.md)搜索与完整独立复算均正常关闭，4000响应/1200000数值内步、177条ray和54份anchor全量核验，严格分类一致，53次下降后仍为prefix9、完整witness0；cursor9仍11.47mm/.05737rad超原caps。差分占3822/4000，53阶段均在首非零callback截断并重置SQP状态，后续需检验实际接纳点间的曲率保留。[运行时文件快照](reports/native_runtime_path_snapshot_20261029.md)105项native QA、实际准备及独立文件核验已关闭，完整1090文件/454957103字节；尚未验证实际加载或bootstrap，[历史身份缺口](reports/native_runtime_path_followup_20261029.md)保持。实际仍cursor8、task/grasp0，新增物理/模型/fresh0，GPU禁用，原任务门槛与总目标保持。

最新CPU机制检验（2026-10-28）：[固定保存方向回缩](reports/saved_direction_backtracking_20261028.md)96条响应和完整独立复核已关闭，14/16方向、36/64内部点恢复可行下降，但原cursor9达标0/96、完整witness0。独立checker元数据错误已保留失败记录并通过隔离修复，84项native QA通过。下一步[同warm的H12有限信赖域A/B](plans/v0.0.1/prefix_trust_region_control_20261028.md)已登记、尚未执行；实际仍止于cursor8，新增物理/模型/fresh0，GPU继续禁用，总目标未完成。

最近CPU接续（2026-10-28）：[H13提前/临近相位分配](reports/paired_reference_phase_20261028.md)两个同源650ms搜索与完整独立复算均正常关闭，各4000响应/1300000内步，严格分类一致、完整witness0。新相位连续10端点包含一个中点，仍仅原cursor0..8达标；原cursor9两组仍超原位置/旋转门槛。后续固定方向回缩结果见上文。新增物理/模型/fresh0，GPU继续禁用，总目标未完成。

最新CPU多周期结果（2026-10-28）：[原生目标H12搜索与完整独立复核](reports/state_updated_trajectory_control_20261028.md)已关闭，4000响应确认完整witness0；[保留达标前缀修复](reports/prefix_preserving_trajectory_control_20261028.md)的新4000响应/54迭代已正常结束，连续预测达标点6→9，但cursor9仍11.52mm/.05758rad超原门槛，完整witness0。新搜索全部4000轨迹/1200000内步的独立复算已正常关闭，严格分类一致、完整witness仍0。新增物理/模型/fresh0，实际仍止于cursor8，GPU继续禁用。 [只读相位诊断](reports/saved_trajectory_phase_diagnostic_20261028.md)已完成：新selected最后三点即使另选路径相位仍超caps，不能仅重新标记时间补救；后续需重新联合规划动作。

最新CPU停止点结果（2026-10-28）：[原生目标与状态更新受限规划](reports/state_updated_stop_control_20261028.md)及独立7250数值内步复核完整关闭；281候选/32迭代未找到cursor9执行解，selected旋转.07346693rad仍超原.05rad门槛。新增物理/模型/fresh0，实际仍止于cursor8；下一步从更早actual368开展包含cursor9的多周期规划，原实际门槛和GPU禁用保持。

最新CPU动力学证据（2026-10-28）：[冻结与状态更新响应](reports/state_updated_joint_response_20261028.md)九个已执行周期及独立1350数值内步核验关闭。更新2ms模型最大关节预测误差.000115660rad，固定M/J同数值法.000452429、旧解析.004138869；尚无新action或任务成功，实际仍止于cursor8。下一步接入受限规划和实际前向验证，原验收门槛及GPU禁用保持。

最新CPU约束诊断（2026-10-28）：[首周期约束归因](reports/fk_stop_constraint_audit_20261028.md)及[制动时间敏感性](reports/fk_braking_time_numeric_20261028.md)独立核验关闭；去掉未来/终端及缩短额外制动时长后仍无cursor9首点执行解，最好候选旋转.06631rad仍超原.05rad门槛。新增物理/模型/fresh0；下一步核对冻结与状态更新动力学，实际任务进度及原验收门槛不变，GPU继续禁用。

最新CPU同起点对照（2026-10-28）：[首周期FK预览8/12/16](reports/fk_first_long_native_20261028.md)三组实际及独立重放完整关闭，各9周期达标但都在cursor9前停止，task/grasp0；1131独立重放周期/675新内环样本的state/q/qvel差异0。更长窗口未扩大实际推进范围，后续定位首周期与未来/terminal约束影响。总目标及原任务门槛未达，GPU继续禁用。

v4最新完整结果（2026-10-28）：[全量独立复核](reports/tracking_wam_integration_20261027.md)已actual wait0/reap，12案例、900请求和31500节点全部校验关闭；任务稳定成功raw 0/6、terminal 0/6，开发门槛未通过，最终留出仍关闭。v5 GPU任务已按用户要求停止，等待明确恢复通知；不能据审核完成宣称总目标实现。

最新运行状态（2026-10-28）：[v5释放阈值WAM开发矩阵](plans/v0.0.1/tracking_wam_v5_20261028.md)已通过129项组件QA、12输入预检、独立metadata及14 CLI dry；模型服务实际就绪，前三案例各600周期/75请求生产已关闭，任务成功均0；第二、三条保存状态确认q7越界，第四条按用户要求停止，GPU任务等待明确恢复通知。保持原12条/900请求/7200周期与任务门槛，尚无新任务成功结论，最终留出仍关闭。完整证据与历史失败见[集成报告](reports/tracking_wam_integration_20261027.md)。

最新CPU控制诊断（2026-10-28）：[多周期实际保持/恢复](reports/reference_recovery_native_20261028.md)及独立重放完整关闭；16新周期全部保持、推进0，保持误差与关节裕量达标但task/grasp均0，未取得原轨迹控制资格。独立395端点和400内环状态核验差异0，GPU继续禁用。

最新恢复分析（2026-10-28）：[首周期可行性优先规划](reports/feasibility_first_recovery_20261028.md)仍未找到推进解；[独立XML复核的静态IK](reports/recovery_pose_ik_20261028.md)确认原下一姿态在.05rad关节裕量内存在合法构型，问题仍在实际状态与固定控制接口下的动态推进。新增物理/模型/fresh均0，未取得任务资格。

最新预览合同检查（2026-10-28）：[完整8点精度约束](reports/trajectory_caps_numeric_20261028.md)在五个较早真实状态中仅首状态找到完整预览模型解，原首点策略全部接收；当前只确认能更早拒绝后续预测失准的计划，无新控制/任务收益。下一步检查同一实际起点的更长因果预览，GPU继续禁用。

最新模型一致性发现（2026-10-28）：[长窗口与可行起点](reports/feasible_start_trajectory_20261028.md)仍未找到12/16点合规解；[独立XML核验](reports/preview_fk_consistency_20261028.md)发现被接收8点模型解的Cartesian预测与同一预测q的FK最大相差17.01mm，后者仅5/8满足原精度。下一步优先修正预测内部运动学一致性，不能据局部模型标签宣称实际控制成功；GPU继续禁用。

最新FK修正与物理核验（2026-10-28）：[FK一致8点模型解](reports/fk_consistent_trajectory_20261028.md)通过独立XML/MuJoCo核验；[原生反馈](reports/fk_feedback_native_20261028.md)两个实际周期误差均达标，随后step370完整未来精度搜索无解而停止。独立370周期/371端点/50内环状态差异0，task/grasp仍0，未到达关键参考点；GPU继续禁用，尚无完整控制或任务资格。

最新首周期反馈证据（2026-10-28）：[同状态对照及原生仿真延续](reports/fk_first_feedback_native_20261028.md)新增7周期实际达标，独立377周期/378端点/175内环状态差异0；连同前两周期仅到cursor8，关键cursor9前仍因模型无解停止，task/grasp0。下一步从早期实际起点检查更长未来成本窗口；原真实验收门槛及GPU禁用保持。

107最新完整结果：[追加wrench覆盖forward](reports/wrench_coverage_forward_20261025.md)5000方程唯一fit、独立增广SVD及三个stage/owner正常退出和来源审查完成；holdout23/24、局部22/24，资格失败，不进入控制。106追加147条/13209周期训练数据完整资格通过。原总目标与最终门槛保持。

104最新完整结果：[当前实测wrench条件forward](reports/wrench_conditioned_arm_forward_20261025.md)4937方程唯一fit、独立增广SVD与三个stage/owner实际wait0和来源核验完成；holdout21/24、局部22/24，资格失败，不进入控制。103[原30轨迹wrench数据](reports/native_wrench_data_20261025.md)6180周期/6210端点及全量数据资格通过。原总目标与最终验收门槛保持。

103当前执行：[原30轨迹wrench采集](reports/native_wrench_data_20261025.md)4核心测试与30实际CLI dry准入通过，持久owner采集/独立检查进行中，尚无全量数据资格。104[当前wrench条件对照](reports/wrench_conditioned_arm_forward_20261025.md)入口已准备，等待103关闭，尚未拟合。原总目标与控制/WAM/fresh门槛保持。

102最新反馈核验：[当前wrench传感器合同](reports/current_wrench_sensor_contract_20261025.md)六baseline/477周期/483端点及12阶段与owner实际wait0完整关闭，直接getter/缓冲和坐标/参考点变换通过；仅确认仿真反馈当前可读，不是新模型或控制资格。原总目标和最终门槛不变。

101最新完整结果：[干预斜率/自然偏置分阶段辨识](reports/separated_response_identification_20261025.md)固定两阶段fit、两独立增广SVD及三个stage/owner实际wait0和来源核验完成；holdout5/24、局部18/24，资格失败，不采用。原控制/WAM/fresh门槛及总目标保持。

100最新完整结果：[固定非线性条件基forward](reports/nonlinear_conditioned_arm_forward_20261025.md)4937方程唯一fit、独立增广SVD及三个stage/owner实际wait0和来源核验完成；holdout22/24、局部22/24（未训练10/12），资格仍失败，不进入控制。原18/18、WAM开发/最终24条验收与总目标不变。

098最新完整结果：[独立干预覆盖forward](reports/coverage_conditioned_arm_forward_20261025.md)4937方程唯一fit、独立增广SVD及三个stage/owner实际wait0与来源核验完成；holdout20/24、局部20/24，资格失败，不进入控制。097[训练响应数据](reports/training_response_coverage_20261025.md)147条/8281周期及完整数据生命周期资格通过。原18/18、WAM开发/最终24条验收和总目标不变。以下为历史快照。

097当前执行：[独立训练响应覆盖](reports/training_response_coverage_20261025.md)21基础测试、147实际CLI dry及检查器导入通过，147条/8281周期采集由持久owner执行，部分结果已独立核验，尚无全量数据资格。096[因果局部响应](reports/causal_local_response_20261025.md)完整离线审查仍21/24局部响应失败；098仅登记下一拟合，尚未执行，原控制/WAM/fresh门槛和总目标保持。

093最新机制核验：[原六baseline控制器追踪](reports/native_controller_trace_20261025.md)477显式控制周期、150内环观察及12阶段/owner实际wait0完整关闭，全部前缀匹配。实际uncoupling均true，目标周期均无限幅，group4全25次触桌；仅诊断，无新控制资格，原总目标未完成。

090最新完整结果：[因果响应历史条件化forward](reports/history_conditioned_arm_forward_20261025.md)唯一4874方程fit、独立增广SVD及三个stage/owner实际wait0和来源核验完成。holdout24/24、局部响应21/24（未训练9/12），离线资格失败，不进入控制；新增物理/世界模型/fresh均0，原最终门槛及总目标不变。

087最新数据资格：[原生arm前向响应数据](reports/native_arm_response_data_20261025.md)完成旧24fit/6holdout共6180行及全部XML/数值速度、实际goal协议、标签和生命周期审查；086六anchor成对响应数据亦已完整关闭。尚无新模型/控制收益，085原18/18资格失败及WAM/fresh关闭不变。

083最新完整结果：[位置/速度一致性回归](reports/pose_velocity_consistent_inverse_20261025.md)18条4907步及36阶段/owner实际wait0、来源核验完成，任务稳定与综合达标均15/18，未达18/18资格；case03/10/12失败。WAM/fresh仍关闭，总目标未完成。

082最新完整结果：[平移逆补偿回归](reports/translation_inverse_qualification_20261025.md)18条4907步及36阶段/owner实际wait0、来源核验完成，任务稳定15/18，原位置/旋转与稳定门槛合格14/18，未达18/18资格。083[位置/速度一致性候选](reports/pose_velocity_consistent_inverse_20261025.md)已通过20个准入进程并启动固定18条正确参考回归；无完整候选性能结果，WAM/fresh仍关闭，总目标未完成。

079最新完整离线资格：[当前速度先验响应模型](reports/velocity_residual_response_20261025.md)5009行独立数据/增广最小二乘及9条冻结一步/8步响应验证通过原保持基线门槛；未全面胜过075/常速度，尚无新action控制或任务收益。下一步按[080](plans/v0.0.1/080_velocity_model_inverse_control_20261025.md)登记的有限原生命令规划和18条正确参考物理回归推进，原WAM/fresh门槛不变。

078最新数据资格：[受控夹爪响应采集](reports/native_gripper_response_interventions_20261025.md)完整12条1568步、三native基线及全部独立机械/标签/速度/命令审查通过；24个stage及execution owner实际wait0/reap/来源核验通过。079[当前速度先验响应模型](reports/velocity_residual_response_20261025.md)固定新数据隔离划分后完成唯一5009行fit，独立数值及一步/8步验证尚未执行，无新控制性能或任务成功，原WAM/fresh门槛不变。

074最新完整结果：[统一残差夹爪物理资格](reports/residual_motor_physical_qualification_20261025.md)18条4907步及逐条独立机械/标签/图像审查完成，稳定14/18，原18/18性能门槛失败；18个物理worker实际wait0/reap/无信号与来源不变均核验，根owner已消失但无其实际wait凭据，生命周期缺口单列保留。075[全阶段前向state辨识](reports/all_phase_gripper_forward_identification_20261025.md)数值复核通过，但wine/demo39单步误差差于保持基线，离线资格失败。两者均不进入WAM/fresh，总目标未完成。

068最新完整结果：[位姿耦合开度预览](reports/coupled_gripper_preview_20261025.md)24条6098步及完整独立审查通过，候选稳定17/18，未达原18/18资格；新增轨迹与baseline均5/6。修复原预览退化但未解决共同wine失败，不进入WAM/fresh，原最终目标不变。

067最新完整结果：[六步开度预览回归](reports/gripper_preview_regression_20261025.md)15条4430步及完整独立审查通过，但候选稳定7/9，留出4/6低于baseline5/6，有两条明确回归；不采用无条件预览，不进入WAM/fresh。原066三训练参考通过仅为局部证据，总目标未完成。

062最新完整结果：[当前环境匹配oracle正控](reports/matched_oracle_requalification_20261025.md)三任务1800物理步及独立机械/退出审查通过，正确参考稳定3/3，原位置/旋转门槛均通过。此为固定S2A正确参考资格，不能计入WAM成功；061候选0/6与fresh关闭不变。

061最新完整结果：[固定闭环开发对照](reports/wam_terminal_development_execution_20261025.md)12/12条、900请求、31500节点、7200物理步及完整独立审查/正常退出/GPU释放均通过；raw稳定1/6、terminal058稳定0/6，原开发性能门槛失败，不采用058进入fresh。继续定位参考路径与受限执行，原总验收门槛不变。以下为历史快照。

060最新完整结果：[058末节点参考验证](reports/wam_terminal_identity_execution_20261025.md)99请求/3465节点及独立数值、正常退出/GPU释放全部通过，原state门槛通过；H16位置/旋转/开度误差相对raw下降17.764%/22.492%/30.569%，全48候选历史隔离对照通过。尚无新闭环任务或fresh结果；下一步按[061](plans/v0.0.1/061_terminal_development_20261025.md)接入固定12条开发对照，原最终24条门槛不变。以下为历史快照。

060最新接续：[请求身份修复与新验证](reports/wam_terminal_identity_execution_20261025.md)已通过36项CPU检查、1483源码/快照、独立metadata、真实CLI dry及模型解释器的实际父类身份接口检查，由新owner在允许UUID上启动。原059首候选接口失败、4请求/140节点及正常释放记录保留。当前无完整候选性能结果，开发/fresh关闭，原最终门槛不变。以下为历史快照。

059最新执行：[058末节点同步验证](reports/wam_terminal_state_validation_20261025.md)已完成1481源码/快照冻结、独立99请求metadata、真实CLI dry和34项CPU检查，并由持久化owner在唯一允许UUID上启动。058仅离线数值资格通过，当前无完整state或任务结果，开发/fresh关闭；原24条最终验收门槛不变。以下为历史快照。

056最新完整结果：[D2聚合与同步validation](reports/wam_clean_on_policy_aggregation_20261024.md)全部完成，D2 300/10500、1260000行fit/独立QR、validation 99/3465及正常退出均合格；但原性能门槛仍失败。新头比052改善，H16相对raw位置/开度误差仍增大16.255%/139.056%，旋转改善25.799%。不进入开发或fresh，继续诊断局部残差与积分state目标不一致的问题；总目标和原24条最终验收门槛不变。以下为较早阶段快照。

056最新执行：[聚合完整资格与同步validation](reports/wam_clean_on_policy_aggregation_20261024.md)。D2全部300/10500与聚合1260000行fit/独立完整QR、正常退出审查均通过；统一新参数已冻结。1476份来源/快照、独立metadata及实际模型CLI dry通过，固定99请求validation已在唯一允许UUID上启动。尚无新候选性能或任务收益，开发/fresh保持关闭，原055失败及原24条验收门槛不变。以下为较早阶段快照。

056最新执行：[D2完整资格与聚合拟合](reports/wam_clean_on_policy_aggregation_20261024.md)。D2全部300请求/10500节点、300初态配对及独立数值/正常退出审查通过，GPU已释放。固定1260000行的唯一CPU fit已实际启动，完整fit/独立QR及新validation尚待完成，无新任务或fresh收益。原055性能失败、总验收门槛及fresh关闭不变。以下为更早阶段快照。

2026-10-24当前执行：[056自身策略训练轨迹聚合](reports/wam_clean_on_policy_aggregation_20261024.md)。D2已通过完整执行准入、实际模型CLI dry及157项CPU检查，1470份来源/快照冻结；同一owner/worker持续采集，最近核实快照为123/300请求、4305节点。尚未完成全量D2数值/正常退出资格，聚合fit/QR和新validation均未执行，fresh保持关闭。相关拟合、审查、owner及validation接入组件已实现，不能把软件检查当作性能收益。

2026-10-24最新完整性能：[055接续结果](reports/wam_synchronous_state_continuation_20261024.md)数值及退出审查通过，但性能门槛失败：旋转改善，位置和夹爪退化；[配对/合同/分布诊断](reports/wam_synchronous_pair_diagnosis_20261024.md)定位到自身生成轨迹末节点的teacher误差反转，根因尚未唯一确定。总目标与原24条留出初态稳定≥20/24、各任务≥4/8不变。以下为较早阶段快照。

2026-10-24最新结果：[054分派错误终止与prefix审查](reports/wam_synchronous_state_execution_20261024.md)。已完成3zero＋1raw/140节点并独立核验；首probe052在采样前KeyError，原worker/owner False及实际wait1/reap/GPU释放保留。分派回归组件已通过6项CPU测试，055仅剩余95请求的显式接续在准备，未重启模型；无新候选性能或fresh结果。下列054启动为历史快照。

2026-10-24最新执行：[054同步新头配对采样](reports/wam_synchronous_state_execution_20261024.md)已通过124项CPU检查、1447源/快照、两完整AST证明、独立最终准入及实际模型CLI dry，并由持久化owner在允许UUID上启动固定99请求。尚无完整采样或新任务收益；原W2失败、最终门槛及fresh关闭不变。

2026-10-24最新资格：[053同步validation参考](reports/wam_synchronous_validation_20261024.md)完成原六demo的1212物理步、24当前图像与完整独立审查，native稳定正控6/6；当前图像/实测状态/未来端点由真实物理时钟绑定。已分离current-only模型清单，0新WAM采样或fresh。下一步另登记052新头配对采样，原W2失败及旧同步未认证记录保留，总目标未完成。

2026-10-24最新候选：[052单值正则化对照](reports/wam_clean_regularization_probe_20261024.md)已完成同一840000行的新CPU fit与独立完整QR，原容差通过；训练归一化位置/开度MSE改善29.59%/17.83%，尚无新采样或任务收益。原W2开发0/6稳定失败保留，fresh关闭。原六个validation demo来源及初态隔离预检通过，下一步实现带真实物理时钟的同步参考采集，不能用训练误差替代最终验收。

2026-10-24最新完整结果：[051开发对照与补充审查](reports/wam_clean_development_dependency_execution_20261024.md)已完成12/12条、900请求和7200物理端点，独立补充数值审查通过；原审查失败记录保留。raw稳定1/6，W2稳定0/6，开发性能门槛失败，当前W2不进入fresh。已完成[失败定位与夹爪释放条件核对](reports/wam_development_failure_localization_20261024.md)，尚无新候选收益；总目标及预留24条验收门槛不变。下列运行中与部分完成描述为历史快照。

2026-10-24最新观测：[051 bowl两seed完整结果](reports/wam_clean_development_dependency_execution_20261024.md)。4/12条已完成：raw稳定1/2，W2稳定0/2；seed43/W2虽曾成功39步，但终点和末40步稳定判定失败。当前候选即使剩余W2全成功也至多4/6，且bowl为0/2，已无法达到原开发≥5/6及各任务≥1/2门槛。仍完成固定剩余8条以取得完整对照，完整独立审查尚未完成，fresh保持关闭；下列计数和运行状态为此前快照。

2026-10-24最新执行：[051修复后开发对照](reports/wam_clean_development_dependency_execution_20261024.md)。110项组合检查、五入口AST独立证明、583来源/585快照及真实CLI dry通过；新namespace实际启动，首client已越过原缺失合同失败点并完成模型请求，物理进度60端点/8chunks。尚无完整case/稳定率，STATE gate=False及原049失败保留，fresh关闭；下列049停止或051准备为此前快照。

2026-10-24最新结果：[049开发首client依赖失败](reports/wam_clean_development_execution_20261024.md)。首client因旧tool_contract挂载路径缺失在输入指纹阶段退出，0请求/物理；1次模型初始化已abort、wait/reap和GPU释放，原False/562源保留。已找到与历史SHA相同的新挂载合同并建立安全副本，实际sim输入预检通过；051新namespace有限接续在准备，无新执行或fresh。下列049启动段为历史快照。

2026-10-24最新执行：[STATE完整资格与049开发启动](reports/wam_clean_development_execution_20261024.md)。STATE独立99/3465数值、来源及正常收尾通过，但state gate=False和validation初态同步未认证均保留；049冻结562源/快照、独立准入和原CLI实际dry通过，固定12条raw/W2开发对照已启动。尚无完整任务结果或fresh收益，总目标未达到；下列STATE运行中描述为历史快照。

2026-10-24最新准备：[049固定开发接续](reports/wam_clean_development_handoff_preparation_20261024.md)代码及125项组合检查通过，QA绑定8份源码/测试；实际绑定正确拒绝未完成048，0新开发模型/物理。最新观察STATE同一模型已闭合32/99请求、1120节点，551源/快照无漂移，尚无完整STATE资格或新任务收益。

2026-10-24最新：[W2完整资格与STATE启动](reports/wam_clean_w2_and_state_execution_20261024.md)。W2全部840000行拟合及独立QR按原容差通过，正常wait0/reap，537源/快照无漂移；048冻结551源/快照、独立准入及两个真实CLI dry通过，STATE模型已在唯一允许UUID上启动。尚无完整99请求资格或新物理/留出任务收益；下列W2运行中描述为历史快照。

2026-10-24最新：[050 D1完整资格及W2启动](reports/wam_clean_d1_lifecycle_addendum_20261024.md)。D1完整300请求/10500节点及独立审查通过；原046收据字段错误的False保留，050独立生命周期补充核验通过。W2相关150项及收据/接续68项检查通过，537源/快照冻结、独立准入和三个真实CLI dry通过；CPU W2 producer已实际启动，拟合/QR完成资格及新任务收益仍待验证。下列D1运行中描述为历史快照。

[当前实现进度与总目标](reports/status_current_20261023.md)：控制与独立仿真链路已建立，真实世界模型参考的多任务稳定效果尚未达标。当前版本终点固定为原预留24条中稳定≥20/24、各任务≥4/8；实时和真机分别另验。下列按日期保留推进记录。

2026-10-24最新实现准备：[048生产桥与STATE入口](reports/wam_clean_state_handoff_execution_preparation_20261024.md)已实现并通过129项相关CPU检查及真实解释器导入检查，尚无实际048准入/STATE。模型和独立CPU审查的导入环境分别固定，额外lookup绑定真实PID/birth；11:19同一D1已闭合270/300请求，519源/快照不变，W2仍未拟合。

2026-10-24最新组件：[048 W2到STATE来源桥准备](reports/wam_clean_state_handoff_preparation_20261024.md)只读资格核验和导入隔离组件已实现，135项相关CPU检查及两解释器4次原sampler help解析检查通过。生产桥/角色收据/048执行入口尚未完成，0新模型或STATE；10:47原D1同一进程已闭合156/300请求，519源/快照无漂移，W2仍未拟合。

2026-10-24最新准备：[047 W2双轮接续](reports/wam_clean_w2_preparation_20261024.md)实现及142项相关CPU检查通过，实际sim运行时组件核验通过；真实live D1和绑定CLI均被正确拒绝，0实际W2 fit/QR。10:22快照同一D1进程已闭合66/300请求，原519源/快照未变；完整D1资格仍待完成，最终门槛不变。

2026-10-24最新启动：[046显式D1接续](reports/wam_clean_d1_continuation_20261024.md)完成219项相关CPU检查、519源/快照、独立准入和实际CLI dry；持久化supervisor已持有原041 collector实际模型worker，仅允许UUID对应当前GPU2/minor6。10:07快照已闭合10/300请求，前2请求/70节点原独立重算通过，尚无完整采集/退出资格；W2、任务及fresh未启动，原失败与最终门槛不变。

2026-10-24最新完成：[045实际CPU W1及独立QR](reports/wam_clean_runtime_fit_20261024.md)均通过完整420000行/150标签窗口，参数按原容差一致，508源/快照及两角色实际wait0/reap通过。原043失败保留，累计W1尝试2次/成功1次；原合成60通过/4严格参考比较失败及Decimal诊断如实保留。原D1 head合同只读接入通过，D1/W2/任务及fresh尚未执行，下一步为046显式接续。

2026-10-24最新资格：[044完整CPU数值兼容审查](reports/wam_clean_runtime_qualification_20261024.md)通过150训练窗口、300请求/10500节点及两路径420000行字节比较；实际sim运行时逐值通过原生产门，498源/快照和正常wait0/reap通过。0新fit/QR/模型/物理/fresh，原043失败与main首窗口False保留；下一步另行固定新输出及累计尝试预算的W1接续。

2026-10-24最新：[043实际CPU接续](reports/wam_clean_training_postcapture_20261024.md)。147项相关测试、Ruff、495源/快照及独立准入/真实dry通过；W1尝试1次，在标签重建检查处停止，0参数/成功fit/QR，失败保留。首窗口main与sim环境有float64差异，sim逐值复现，完整运行时资格尚未建立；没有重试、D1或fresh。代码已提交`2d218d8`，双语PR说明见[报告](reports/pr_clean_training_20261024.md)。

2026-10-23最新结果：[041完整D0与W1前置失败](reports/wam_clean_training_continuation_20261023.md)。新D0的300请求/10500节点、独立审查及正常退出/GPU释放已通过；W1源码引用准入错误发生在fit前，0fit/QR/未来标签decode，supervisor保留False并停止，D1未启动。[043有限CPU修复](plans/v0.0.1/043_clean_training_postcapture_20261023.md)另行准备，不重采合格D0，不覆盖失败。等待期间[最终接入隔离组件](reports/wam_fresh_acceptance_implementation_20261023.md)339项组合检查通过，仍无实际候选或fresh性能。下列“D0运行中”为此前快照。

2026-10-23最新接续：[041独立准入与实际启动](reports/wam_clean_training_continuation_20261023.md)。最终23模块730通过/1跳过、Ruff通过，484源/快照冻结；新metadata/执行独立审查及实际D0 dry通过。短launcher已正常返回，PPID1的持久化supervisor持有新D0 owner及唯一允许UUID上的模型；尚无整轮采集、fit、开发或fresh收益。原中断数据不进入fit。下列“041准备中”和旧D0启动段均为此前快照。

2026-10-23中断核查：[原D0已停止且缺退出资格](reports/wam_clean_training_interruption_20261023.md)，保留20完整请求＋25个partial节点，精确调用数未知；原462源及全部失败产物已封存，旧数据不用于fit。新041有限接续在实现/独立准入准备，尚无新模型启动、fit或任务/fresh收益。下列“D0已启动”为此前历史快照。

2026-10-23执行推进：[040执行准入与D0启动](reports/wam_clean_training_execution_20261023.md)。19相关模块613通过/1跳过，Ruff通过；462源/快照及完整独立执行审查已通过。原63位训练seed不兼容服务/RNG的问题已按运行前040修订为显式31位映射，原metadata不改；D0唯一模型进程已启动，尚无整轮采集/真实fit/任务/fresh收益。总目标及20/24、各任务4/8门槛不变。

2026-10-23训练准备进展：[038实现与准备](reports/wam_clean_training_preparation_20261023.md)已绑定30来源/150当前锚点/300独立训练noise及24 validation锚点/96生成矩阵，独立metadata审查通过。采集、固定fit/独立QR及owner已有CPU验证，执行仍未就绪，0新候选fit/任务/fresh收益；尚须完整生成诊断和固定开发任务接入。

2026-10-23桥接进展：[037完整模型采集＋039独立补充审计已联合通过](reports/wam_clean_bridge_20261023.md)，7请求/245 denoise，三raw历史及三组zero精确回归，非零残差真实改变102值/29节点，模型正常退出并释放GPU。原037检查器将inactive118零钳制误判为未来条件的False完整保留，补充仅CPU复核，未重跑模型。034数据与真实桥接两项前提已建立；[038有限训练协议](plans/v0.0.1/038_clean_residual_training_20261023.md)的实现/准备继续推进，0新fit/任务/fresh收益，总目标仍未达到。

2026-10-23 10:51+08目标核查（历史快照）：[036总目标和验收标准](plans/v0.0.1/036_project_goal_20261023.md)、[当时实现核查](reports/status_review_20261023.md)。已建立可执行S2A及正确参考独立仿真正控，真实WAM新初态稳定成功尚未达标；当时034数据30/30合格，035缺完整结果/退出收据，桥接尚未获资格，后由上列037＋039新证据推进。当前版本终点仍为原预留12初态×seed42/43的24条中稳定≥20/24、各任务≥4/8；最终性能未运行。

2026-10-22工作区更新：后续全部在`/share-mixed/project/eai_pwm/home/hcr/repos/s2a`执行；旧目录不再写入。GPU仅当前可见2、3，按AGENTS中的UUID限定。[迁移记录](reports/workspace_relocation_20261022.md)列出恢复、测试失败边界及当前运行状态。

2026-10-22进度核查：正确参考经固定S2A的独立仿真正控已通过，真实WAM的多任务、新初态稳定成功总目标仍未达到，最终预留24条性能验收未打开。029诊断已完整结束，030原生训练数据资格为28/30、False，032 motor保持未改善失败案例；[034新收尾协议](reports/settled_native_training_20261022.md)现已完整30/30稳定并独立核验（原False保留），下一项为035真实clean桥接资格，尚无新真实拟合或WAM任务收益。[现状、证据与下一步](reports/status_audit_20261022.md)。下文保留各日期的阶段记录，最新判断以该核查及对应正式报告为准。

S2A 把世界模型给出的期望 state/delta_state 转成固定控制器可接收的 action，使执行后的实际状态接近目标。

2026-10-21继续以**稳定成功**为目标。完整WAM曾在旧初态成功，但尚未达到多任务、新初态的稳定门槛：新v3夹爪8800/15400共24条稳定0/24；8400完整12条稳定1/12，仅alpha0/cream（修改了生成参考）。[010短闭环](plans/v0.0.1/010_short_replan_wam_20261021.md)目前raw每8步三个旧任务稳定1/3，仅bowl尾13.1s成功且保留完整模型旋转；每16步raw/alpha0均0/3，8步alpha0三条也完整0/3；全12/7200步/678请求已独立补核。原收尾发现未使用历史权重15400外部缺失，错误保留，未伪造原门禁成功。单例不算整体达标。见[本轮结论](plans/v0.0.1/conclusion_20261021.md)和[记录](plans/v0.0.1/record_20261021.md)。

state对照不支持盲目更换推理配置：[latest15600](reports/wam_latest_state_screen_20261021.md)九项误差均差于8400及15400；[guidance3](reports/wam_guidance_state_20261021.md)和[训练attention backend](reports/wam_backend_state_20261021.md)各48次完整诊断均未通过事先规定的改善门槛，不追加其物理任务评测。[011采样步数](reports/wam_sampler_steps_state_20261021.md)的144次已完整核验：35步48次全部精确复现原输出，70/140步九项误差均更差，不推广。模型实际只使用物理GPU4–7；CPU/OSMesa空CUDA/LD，外部Cosmos及发布数据只读。

[020/021同端点因果集成](reports/wam_temporal_equal_20261021.md)完整三任务稳定1/3（cream），原raw8/seed42也是1/3（bowl），不按任务拼接；seed43的raw8为0/3。集成参考跟踪仍有45–162mm误差，不能将失败全归WAM。[022固定扩展文本](plans/v0.0.1/022_static_command_state_20261021.md)96请求完整、原48输出exact，扩展文本未过state门槛；[023局部参考校正](reports/wam_generated_reference_calibration_20261021.md)19组fit及独立重算完整，短期共同误差改善11%–14%但组件与跨episode门槛失败，均不推广到物理或fresh。[024当前训练—推理数值桥接](reports/wam_current_training_inference_numerics_20261021.md)已完成：8400/真实future RGB及action/正确三任务的27次denoise及18组训练loss独立重算全部通过，匹配条件的训练与推理有效输出逐值相同，GPU6进程已正常释放；0完整采样/物理，不能代替任务验收。其准入快照CPU检查433 passed / 1外部integration deselected，Ruff292文件通过。[025冻结主体残差readout](reports/wam_frozen_readout_20261021.md)已完成249次特征前向和2组固定fit及独立重算：开发validation teacher速度MSE总体改善28.53%，但sigma=.8退化2.043倍，推广门槛False；不执行96次条件生成、物理或fresh。原资源观测超时False保留，经[独立接续](plans/v0.0.1/025_release_observation_addendum_20261021.md)完成尚未执行的CPU审计与fit，没有重复模型前向。原冻结QA499 passed / 1 integration deselected；接续相关32 passed，当前307文件Ruff通过。[026 flow时间readout](reports/wam_flow_consistent_readout_20261021.md)已完成GPU6合成精度资格及2固定CPUfit/独立AugLS：开发validation teacher MSE总体改善49.79%，但sigma=.8退化13.23%，门槛False；实际生成准入拒绝，0新WAM前向/完整生成/物理/fresh。冻结QA546 passed / 1 integration deselected、Ruff321文件通过。[027训练集尺度readout](reports/wam_train_scale_readout_20261021.md)已完成1固定CPUfit/1独立AugLS、原026bias复用不重拟合：开发validation teacher MSE总体改善57.45%，三个sigma均改善，全部固定teacher门槛通过。mu/s独立差0、量化预测exact；全仓582 passed / 1 integration deselected、Ruff331文件通过。目前只获新路径运行资格研究条件，0新WAM生成/state评分/物理/fresh，尚无任务收益证明；[028](plans/v0.0.1/028_scaled_readout_state_20261021.md)新GPU6的20场景精度资格已独立通过，614 passed / 1 integration deselected、Ruff341文件通过；[完整99生成state报告](reports/wam_scaled_readout_state_20261021.md)现已核验：3465 denoise/48精确初态对、H8/H16 joint改善10.64%/16.14%，但H16 opening、H40位置及seed42短窗保护失败，原gate=False，不推广到物理或fresh。模型/CPU正常wait/reap、GPU6释放，另有独立完整来源/分组审查和3PNG/3页PDF。

[029同实际时间的自由轨迹诊断](plans/v0.0.1/029_free_trajectory_diagnostic_20261021.md)已实现，并按两个运行前补充固定[CPU/CUDA标量数值合同](plans/v0.0.1/029_scalar_promotion_addendum_20261021.md)与[隔离源码修订](plans/v0.0.1/029_source_revision_addendum_20261021.md)。当前111源及快照、2023安全输入/163权重stat已独立复核，649 passed / 1 integration deselected、Ruff353文件通过；物理GPU6的20场景/40实际scheduler转换各对预定gold逐位通过，CPU/CUDA的4场景乘积差异按原登记如实报告。唯一模型已启动48raw+144同时间teacher采集（预算1824denoise、0fit/物理/fresh），首条完整raw已精确复现028；最终轨迹统计尚待完成及独立核验，不视为任务成功。

执行侧[短重观测正确参考](reports/short_replan_qualification_20261021.md)12/12、[真实seed43正确参考](reports/wam_seed43_execution_qualification_20261021.md)3/3通过；[双种子日志检查器](reports/wam_seed_aware_checker_20261021.md)旧27条精确回归。这些不计WAM成功。[016](plans/v0.0.1/016_tracking_replan_20261021.md)保持所有raw pose/opening，以实测误差决定8–40步重新观测，WAM完整0/3稳定。[017持续效果](reports/wam_long_raw_20261021.md)三条1200步/450请求全部独立核验：仍仅bowl成功，稳定1/3；bowl尾43.1秒连续成功，wine/cream追加30秒仍0成功，后半TCP跟踪约1.9mm。不能只归因时间不足，也未达到多任务稳定。[019第二seed复测](reports/wam_raw8_seed43_20261021.md)已完整3/1800步/225请求核验，seed43三任务均未成功；原30秒raw8的42/43各1/3、0/3，不能挑成功seed拼策略。所有自有服务正常释放；12个预留新初态未做性能评测，无真机，已有产物拒绝覆盖。

2026-10-20 独立物理评测及窗口对照已完成：主矩阵四份checkpoint统一H40/执行8，稳定成功为warm6200 0/24、8400 0/24、8600 0/24、8800 1/24；native与oracle＋冻结S2A均24/24稳定。旧集合8800执行8/16/40分别1/12、0/12、1/12稳定，成功病例不同，未观察到总体提升。全部216条评测＋27条固定参考诊断共243条/145800步完成并独立审计；能完成部分任务，稳定能力仍未达预注册门槛，不能将全部失败唯一归于权重或转换。详见[正式报告](reports/wam_independent_20261020.md)、[最终结论](plans/v0.0.1/conclusion_20261020.md)和[实施记录](plans/v0.0.1/record_20261020.md)。来源独立性仅限已记录物理评测/控制调参，并非预训练未见；历史训练消费及上游图像/pose物理同步仍未知。

后续模型实现和测试仅允许物理GPU4、5、6、7；旧八卡运行已取消并归档。当前编排入口 `run_wam_independent_partition.py` 固定partition0–3分别使用GPU4–7，本轮执行窗口对照沿用该映射且已完成。CPU测试和OSMesa仿真显式设置 `CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=''`；历史配置保留原始字节，不沿用其中旧GPU映射启动任务。

```text
已生成的期望 delta chunk → 带时间的 SE(3) 参考轨迹
                                  ↓
当前实测状态 + 因果历史 → S2A → 固定控制器 → 机器人/环境 → 新实测状态
```

主要评价 `F_C(s_now, G_C(...))` 的状态跟踪误差；历史 action 误差仅作诊断。必须区分 measured state、desired state、observed delta 和 controller command。

2026-10-19 v2四层验收已完成：2688次离线预测、90条物理仿真/51480步。确认并在S2A评测服务修正了推理无效Action128槽加噪与训练置零的差异；最新8800在原生H40窗口下曾成功1/12、末端成功1/12，成功任务为bowl→plate，约5秒成功后保持至30秒结束。H16下8400为2/12曾成功、1/12末端，8800为0/12；正确state经冻结S2A为12/12。完整结果见[四层验收报告](reports/wam_checkpoint_acceptance_20261019v2.md)、[state报告](reports/wam_state_acceptance_20261019v2.md)和[最终结论](plans/v0.0.1/conclusion_20261019v2.md)。这是固定WAM/固定S2A、不计推理延迟的独立仿真证据；修正尚未合入外部Cosmos生产代码，未执行真机实验。

已实现 DROID 单 episode 的只读抽取、几何转换、参考接口编码、局部响应辨识、在线有状态 PID adapter 和模型内扰动对照。初轮使用真实未来状态形成 oracle 参考。后续新增 LIBERO 独立 MuJoCo 仿真和固定 Cosmos checkpoint 的成对评测，执行状态见 [仿真报告](reports/libero_simulation_20261016.md)。完整设计见 [实施计划](plans/v0.0.1/001_state_to_action.md)，过程见 [实施记录](plans/v0.0.1/record_20261016.md)，完成状态及限制见 [结论](plans/v0.0.1/conclusion_20261016.md)。

2026-10-19 扩展修复了恢复仿真状态后的OSC缓存同步，区分HDF5动作后观测与当前R1/R2发布观测，并加入开度方向保持、释放锁存、输入延迟/噪声及终态保持对照。该阶段状态见 [控制审计报告](reports/libero_control_audit_20261019.md)、[本轮记录](plans/v0.0.1/record_20261019.md) 和 [本轮结论](plans/v0.0.1/conclusion_20261019.md)。候选的实验收益与部署资格分别判断，所有结果均无真机执行。

继续实施新增catalog所指当前发布帧的配对预测诊断，以及观测时间驱动的异步参考接入。`TimedReference`保留采集时的world锚点和时间轴；`AsyncReferenceBuffer`拒绝过期/乱序结果、跳过已过去的endpoint，空窗固定保持。新仿真逐步刷新当前端点反馈，只在需要请求模型时采集相机；物理控制与HTTP推理并行，实际墙钟超时单列记录。接口可执行、目标正确、任务成功和实时部署分别验证。

入口配置为 `configs/libero_followup_20261019.json`。依赖此前冻结的来源bridge和基线初始图像/预测，路径与指纹随结果保存；复跑请另设output_dir，已有结果不覆盖。以下命令均从本仓库执行，模拟器和模型服务仍是可选环境：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/prepare_libero_followup.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_libero_alignment.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_libero_release_revision.py --seed 42
# 独立终端运行本轮自有模型服务；结束后只停止其记录的进程组
CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH='' .venv/bin/python scripts/serve_libero_policy.py --config outputs/libero_followup_20261019/serving_config.json --seed 42 --gpu 4 --port 8380
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/probe_libero_training_images.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/verify_libero_camera_capture.py --seed 42
# 编排器从显式simulation_python启动物理worker；live串行避免服务排队混淆延迟
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_async.py --seed 42 --stage synthetic
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_async.py --seed 42 --stage live
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_libero_followup.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_libero_followup.py --seed 42
```

WAM 效果专项已完成55项完整物理实验/50739步：不计推理延迟，两种纯WAM转换均0/5，共享解码正控均5/5；独立几何证据确认当前生成参考与任务不符，不能只归因为缺少转换。详见 [效果报告](reports/libero_wam_effect_20261019.md)。本专项暂停物理等待推理、不使用参考过期判断，用于判断效果。`configs/libero_wam_effect_20261019.json` 冻结共享 Action128 解码的 oracle/真实 WAM 对照，`configs/libero_wam_components_20261019.json` 冻结有特权的分量干预。完整请求、原始/有效参考、物理时钟、状态摘要与实际发送命令均保留。复现入口见 [scripts](scripts/README.md)，本专项运行结果追加于本轮记录及结论。

后续四层验收已确认原服务会向118个无效Action128维度注入噪声，与训练的置零合同不一致。上述历史结果描述原服务组合，不能单独用来否定冻结权重的任务能力；S2A评测服务中的mask修正及重新验收见 [v2记录](plans/v0.0.1/record_20261019v2.md) 和 [v2结论](plans/v0.0.1/conclusion_20261019v2.md)。外部Cosmos仓库保持只读，尚未将修正合入其生产服务。

```text
src/s2a/geometry.py        column-rot6d、SE(3)、TCP/flange、SO(3) 误差
src/s2a/data.py            provenance join、物理时钟、masks、样本与指纹
src/s2a/interface.py       固定版本 DROID 参考接口与独立夹爪 position
src/s2a/trajectory.py      固定锚点参考、插值、chunk 衔接
src/s2a/control.py         P/PI/PD/PID、状态、滤波、限幅和 anti-windup
src/s2a/runtime.py         StateToActionAdapter（生成命令，无硬件 I/O）
src/s2a/identification.py  ridge 前向响应与正则逆
src/s2a/study.py           时序选择、日志留出评估
src/s2a/rollout.py         候选命令推进新状态的局部模型对照
src/s2a/reporting.py       Markdown、JSON 摘要及 Matplotlib 曲线
scripts/                  实验入口
configs/                  数据路径、划分、所有网格及扰动参数
plans/、reports/           Git 内维护的设计、记录及精简结果
outputs/                  数据子集、参数、完整指标及图（Git 忽略）
```

Python 3.11+，从本仓库执行：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' python3 -m venv .venv
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m pip install -e '.[dev,report]'
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m pytest
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m build --no-isolation --wheel --outdir outputs/dist
```

当前 NGC 主机也可用 `/opt/venv/bin/python -m venv --system-site-packages .venv` 继承系统 runtime 包；只向本仓库 venv 安装开发依赖，不修改共享环境。实际运行版本保存于 manifest，版本快照见 [constraints](configs/constraints_20261016.txt)；它不是跨平台完整锁文件。基础依赖只有 NumPy/SciPy/PyArrow，绘图需 `report` extra。

首个样本由 [配置](configs/droid_episode0.json) 固定为 DROID / Franka Panda + Robotiq 2F85 episode 0，转移 `[30,330)`，保留 endpoint 和 3 帧历史。数据根和明确的 shard 路径均由配置传入；S2A 不依赖 Cosmos 训练 dataloader、工作目录或模型权重。

```bash
# 完整 P0–P3：只读数据、参数选择、模型内 rollout、报告与曲线
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --config configs/droid_episode0.json --seed 42 --stage all --plots

# 可单独运行 extract / analytic / calibrate；报告重绘不重新拟合或打开 test
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage report --plots

# 产物/实验源码指纹与 70 个 test rollout 最终命令约束（不重新拟合）
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_artifacts.py --seed 42

# 显式读取外部发布数据的集成测试
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m pytest tests -m integration --data-config configs/droid_episode0.json
```

默认测试不读取外部数据，集成测试标为 deselected，不算通过。显式运行时，文件缺失或契约不符会失败。所有测试临时产物位于 `outputs/pytest/`。实验产物位于 `outputs/droid_episode0/`，包括 sample/manifest/audit、analytic commands、geometry/calibration/closed-loop metrics、adapter/PID 参数、完整 traces、report、figures 和 SHA256 清单。

fit `[30,210)`、validation `[220,260)`、test `[270,330)` 保持时序隔离；带历史/延迟的支持不得跨边界。所有标准化只在 fit 计算，模型及增益只在 validation 选择。每个候选、场景、评估区间单独初始化历史、延迟队列和 PID 状态，计分剔除 8 帧 warm-up；部署时跨 chunk 保留 PID 状态。

在线入口是 `s2a.runtime.StateToActionAdapter`：

```python
from s2a.control import PIDConfig
from s2a.runtime import StateToActionAdapter

adapter = StateToActionAdapter(H_FC, dt_command=planned_period, pid_config=PIDConfig())
adapter.set_chunk(measured_tcp, physical_delta_chunk, planned_knot_times)
result = adapter.update(current_measured_tcp, observation_time, now=decision_time)
command = result["command"]  # 6D arm + 绝对 gripper，gripper_action_space="position"
```

示例变量由调用方提供；`H_FC` 来自 tool contract。chunk 必须已反归一化、有有效 mask，时间轴必须是事先生成的执行参考；`now` 与观测时间使用同一物理时钟。`dt_command` 是计划时域，不读取未来观测才知道的 `transition_dt`。默认 PID 增益为零，实验搜索的参数不作为真机部署配置。

canonical TCP 的局部 delta 在 SE(3) 上右乘积分，再完整转换到 flange；DROID `cartesian_velocity` 是归一化接口量，不能当作 m/s、rad/s。夹爪 slot 9 是绝对 opening 命令，与 arm PID 分离。历史控制器版本和 command/state 时刻仍未确认，编码器属于固定公开版本的 reference-interface。

实验矩阵保留解析、线性、P/PI/PD/PID、增益调度、直接逆及残差模型。后三者需要验证同设备的多 episode 数据后进入 P4；初轮未执行独立仿真和真机。后续 LIBERO 仿真不等于 DROID 真机验证。日志回放、局部模型自洽、独立仿真和真机证据分别记录。开发约定见 [AGENTS.md](AGENTS.md)。

同型号跨数据集的解析验证入口：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_cross_dataset.py --seed 42
```

名单在 [cross_dataset_franka.json](configs/cross_dataset_franka.json) 冻结，分别评估 RoboMIND-v1 的30条真实Panda日志，以及 M1/A1 的20/10条仿真日志。各自使用原始TCP/方向约定与 source row provenance，不迁移DROID控制器参数。RoboMIND master joint 另经名义Panda FK作命令代理诊断；所有结果仍属于日志验证，未执行真机或独立仿真控制。设计见 [扩展计划](plans/v0.0.1/cross_dataset_franka_plan.md)，最终指标见 `reports/cross_dataset_franka.md`。

LIBERO 的仿真入口使用独立 Python 3.11 环境，固定 MuJoCo2.3.7、robosuite1.4.1、NumPy1.26.4，环境版本见 [snapshot](configs/libero_sim_requirements_20261016.txt)。`src/s2a/osc.py` 针对固定 OSC 内环做二阶响应解析反解，使用当前末端速度并施加保守权重与最终命令变化率限制；不使用 DROID 接口常数。实验方案见 [仿真计划](plans/v0.0.1/libero_simulation_plan.md)。

```bash
# 原生expert正控、相同oracle轨迹与独立小幅运动（CPU物理仿真）
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_matrix.py --seed 42 --stage qualification
# 旧checkpoint已保存预测的固定参考回放，不重新查询模型
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_libero_matrix.py --seed 42 --stage prediction_replay
# 历史五副本policy启动映射已归档；后续模型任务使用上文GPU4–7编排
# 检查所有物理步、控制约束、成对输入与指纹，再生成报告/曲线
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_libero_simulation.py --seed 42
```

单trial可直接用 `outputs/libero_env311/bin/python scripts/run_libero_simulation.py --seed 42 --stage oracle --candidate compensated --trial 0`，Python前仍加 `LD_LIBRARY_PATH=''`。新增仿真不读取Cosmos训练dataloader、不修改外部仓库；只有policy服务通过显式路径只读加载Cosmos推理实现和权重。policy仿真不读取专家未来状态。重跑矩阵可复用完全相同配置的已完成项；完整目录已存在但执行中断时需先归档该项，避免覆盖证据。

2026-10-16配置保留历史协议。当前正控使用 `run_libero_audit.py` 显式刷新恢复后的控制器反馈，按case隔离LIBERO环境配置。当前图像闭环使用 `configs/libero_policy_20261019.json`：baseline是原arm+position，候选是保守响应逆+intent，两者都刷新初始化缓存。接口提供可选实验配置，默认夹爪映射仍为position。

```bash
# 65项已见场景的时间/缓存/夹爪/arm因子对照；从clean commit执行
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --seed 42 --workers 4

# metadata预先冻结的额外10条：70项普通对照，120项压力，70项完整轨迹/终态
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/prepare_libero_holdout.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_holdout_20261019.json --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_stress_20261019.json --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_terminal_20261019.json --seed 42

# h5py和PyArrow使用各自兼容的解释器，通过outputs中的数字bridge核对R1/R2来源
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/audit_libero_provenance.py --seed 42 --stage export
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_libero_provenance.py --seed 42 --stage verify
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/prepare_libero_published.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_published_20261019.json --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_published_terminal_20261019.json --seed 42

# 保存模型目标的独立前向回放：需先完成本轮实时baseline，和实时图像闭环分开报告
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/run_libero_audit.py --config configs/libero_policy_replay_20261019.json --seed 42

# 完整性/相同参考前缀/成对模型输入、报告及PNG/PDF
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_libero_audit.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_libero_audit.py --seed 42
```

每个新矩阵拒绝覆盖已有目录；`--resume`仅复用配置一致、无错误且预算完整的结果。未完成目录先归档再重跑，任务失败的完整结果同样保留。`--allow-incomplete`的报告明确标记未完成，不能作为完整验收通过。

2026-10-19的历史模型服务使用 `serve_libero_policy.py` 和 `configs/libero_policy_20261019.json`，当时五副本GPU/端口映射仅保留为实验记录，不能直接用于后续启动。当前模型任务使用上文GPU4–7编排。检查历史模型输出合同用 `audit_libero_predictions.py --policy-config configs/libero_policy_20261019.json --seed 42`。同步推理期间暂停物理时钟，不将该评测当作真实延迟下的实时控制证明。

### WAM checkpoint 四层验收（2026-10-19 v2）

本轮以冻结 WAM 权重为被测对象，分别验证训练/推理数值合同、直接 state 预测、S2A执行资格和无推理延迟的多任务闭环。方案、保守决策与完成状态见 [实施记录](plans/v0.0.1/record_20261019v2.md) 和 [结论](plans/v0.0.1/conclusion_20261019v2.md)。

`configs/wam_checkpoint_acceptance_20261019v2.json` 保留原服务协议，`configs/wam_checkpoint_masked_20261019v2.json` 定义显式mask对照：无效Action128维度在初始噪声、网络输入和velocity中始终为零，匹配训练合同。`src/s2a/wam_runtime.py`只在独立模型进程中加载相关依赖，不更改外部Cosmos代码或任何学习参数。6200基线采用实际warm-start父权重及当前run的归一化合同，不能解释为旧TCP协议的原生成绩。

冻结配置中的caption嵌套假设已撤回，正确解释见 [协议勘误](plans/v0.0.1/wam_protocol_errata_20261019v2.md)。保留原文件字节以维持预注册哈希及严格恢复条件。

mask缺陷的源码定位、已执行回归和生产移植边界见 [serving合同说明](reports/wam_serving_contract_20261019v2.md)。当前hook只接受LIBERO十槽合同，其他domain和通用batch仍需在生产移植时单独验证。

所有命令从本仓库执行，产物写入`outputs/wam_checkpoint_acceptance_20261019v2`。已经存在的完整或失败证据均拒绝覆盖；运行前检查目录和服务，重跑使用新输出目录或显式归档。

```bash
# 固定离线episode与多阶段输入；下载并校验新任务的官方数据
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/prepare_wam_acceptance.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/fetch_libero_acceptance_assets.py --seed 42
# 固定6个独立物理初态；执行专家command与真实state正控、独立运动参考
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/prepare_wam_simulation.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py --seed 42 --stage native
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py --seed 42 --stage oracle
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py --seed 42 --stage probe
# 对应冻结服务就绪后，每个checkpoint分别运行；一个HTTP服务只接收一个请求
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_state_matrix.py \
  --config configs/wam_checkpoint_masked_20261019v2.json --seed 42 --checkpoint 8800_masked
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py \
  --config configs/wam_checkpoint_masked_20261019v2.json --seed 42 --stage wam --checkpoint 8800_masked
# 无GPU的来源与标签检查，以及明确标注是否完整的汇总
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_wam_training_history.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_wam_state_labels.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/analyze_wam_acceptance.py --seed 42 --partial
# 仅在全部预注册矩阵预算完成且完整性校验通过后生成最终汇总
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/analyze_wam_acceptance.py --seed 42
```

模型服务`serve_wam_acceptance.py`和数值审计`audit_wam_numerics.py`使用外部Cosmos的Python解释器，仅只读加载模型实现；`PYTHONPATH`显式包含外部Cosmos根目录和本仓库`src`，并设置`LD_LIBRARY_PATH=''`、`PYTHONDONTWRITEBYTECODE=1`、显式指定单张允许物理卡4/5/6/7的`CUDA_VISIBLE_DEVICES`及独立`MASTER_PORT`。编译缓存通过`TORCHINDUCTOR_CACHE_DIR`放在本仓库`outputs`。服务参数为`--seed 42 --checkpoint 8800 --port 8594 --active-slot-mask --exact-published-caption`；数值审计参数为`--seed 42 --horizon 40 --published-caption`，`--training-backend`额外复算保存的attention和固定padding配置，需要独占GPU显存。`audit_wam_caption_contract.py`在同一外部解释器中只做CPU输入准备核验。

在线只输入当前图像和简短任务指令，physics在推理期间暂停；成功以LIBERO原生谓词判定，分别报告曾成功、末端成功和抓取接触/高度/位移等连续进度。完整caption属于离线特权诊断。Study-heldout仅表示本次实验整episode隔离，当前没有证据可将其称为预训练未见数据。

完整性检查与原生窗口补充：

```bash
# 完整权重哈希只执行一次；DCP存储核验使用具备相同torch版本的外部模型解释器
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_wam_acceptance.py --seed 42 --stage fingerprint
# 主矩阵全量完成、生成summary.json之后核对每条已保存响应和执行目标
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_wam_acceptance.py --seed 42 --stage verify
# 以下prepare仅用于全新补充输出目录，复用主实验的固定fixture和权重指纹
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_wam_native_horizon.py --seed 42 --prepare
# 独占同权重masked HTTP服务后，40步生成、仍执行前8步，完整12个用例
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/run_wam_simulation_matrix.py \
  --config configs/wam_checkpoint_native_h40_20261019v2.json --seed 42 --stage wam --checkpoint 8800_masked
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_wam_native_horizon.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/verify_wam_acceptance.py \
  --config configs/wam_checkpoint_native_h40_20261019v2.json --seed 42 --stage verify
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/plot_wam_acceptance.py --seed 42
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/report_wam_checkpoint_acceptance.py --seed 42
```

H16是四checkpoint的同协议主对照，H40是最新8800的训练原生窗口诊断；二者分开报告。`--job-indices`只分割不重叠的用例集合，`--server-port`可接入已验证逐项复现相同输出的服务副本，不改变采样和控制参数。`scripts/schedule_wam_replica.py`与`scripts/schedule_wam_native_horizon.py`保留历史八卡分片，后续不可直接启动；复跑须先制定并记录只用物理4–7的新资源编排，不能只改卡号而遗漏用例。

最终报告入口要求主矩阵和H40补充的完整汇总及独立完整性检查均完成，重新核对已保存文件指纹和90条/51480物理步预算，拒绝覆盖报告。任务谓词、参考跟踪及目标物体接触/高度/位移分别呈现，不能将位移或接触自动计作成功。
