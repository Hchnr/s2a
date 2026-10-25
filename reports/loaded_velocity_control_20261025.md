# 081 闭合加载保持与模型跟踪：纯路由实现，未执行物理

基于080加载配对证据预登记081，冻结同一079参数和080跟踪规划器，不重新拟合、不扫描权重、不按案例选策略。复用默认GripperPositionHold及068固定六步耦合预览作因果intent：closing_episode时执行原生闭合/限位保持，其他时刻使用079规划器跟踪未被预览替换的原始opening参考。状态只在当前episode内跨chunk保留。

src/s2a/loaded_velocity_control.py明确返回route、原状态机和预览guards；只有model_tracking路由带完整最优plan，native_close_load不冒充MPC最优。两路均记录079对实际执行命令的响应预测及其速度包络支持标记；native加载路由不因模型跟踪失败而临时启用，模型跟踪不可行直接抛错。非有限实际命令预测亦抛错，不clip V。

7项实际检查通过：闭合阶段即使opening已匹配仍继续加载至motor限位；0保持已有闭合目标；chunk边界保留状态且新episode不共享；两周期释放确认后按原始reference规划而非预览reference；pose span guard阻止提前释放；tracking失败不借loading fallback；loading路线的模型不支持状态如实记录；非有限预测拒绝（部分项合并在同测试）。最初非有限fixture的全部1e308系数因正负相消仍有限，修正fixture为必溢出两正项后重新运行全部7项通过；未为测试改变生产逻辑。Ruff通过，测试产物outputs/loaded_velocity_control_checks_v2_20261025。

尚未接入物理、未做完整控制AST/两路线独立审查，不具备080或WAM/fresh资格。该intent假设可能牺牲自由空间部分开度精度，须报告state误差而不能只看任务稳定。080原批次仍冻结执行；当前13条独立审查7/13稳定，原18/18门槛失败不变。0新fit/物理/模型请求/fresh，总目标未完成。

## 仿真接入与独立两路核验已准备

新增loaded_velocity_simulation.py仅改变081路由初始化、跨周期实际gripper command历史与相应算法标签；完整函数AST恢复比较通过，所有其他控制函数与080一致。独立loaded_velocity_independent_replay.py重建六步耦合参考、用既有scalar GripperReplay复算全部锁存/释放字段，模型路由独立完整搜索，加载路由核验原生命令并明确plan=None；两路实际命令的响应/包络标志均独立复算。完整verify_case恢复AST比较保留原非路由门槛。

上述控制分支范围1项、两路标量回放及篡改拒绝4项、机械checker范围1项，加原纯路由7项，共13项已实际通过。实际3.11首条dry及Ruff通过（采集入口import排序修复后重查和dry通过）。完整18条配置/collect/verify/等待owner已准备，driver另汇总加载/跟踪周期与执行预测超包络数，不隐藏路由证据。080已完成关闭，稳定12/18且性能失败；081 launch将强制核对080 closure/summary，重新运行包含原23项的36项检查、18条dry与checker导入，尚未以准备就绪冒充物理通过。

## 081正式准入通过并启动，首条独立资格通过

080完整closure已核对（18条4907步、稳定12/18失败、36stage及execution owner正常wait0），此前配置仅准备未并行物理。081正式launch又完成36项测试、18条实际3.11 dry和完整checker导入，20个准入进程全部正常退出，冻结来源/模型参数/包络SHA前后不变。

等待包装进程3583726/start_ticks583592678、execution owner3583735/start_ticks583592731已核验真实存活，CPU CUDA/LD为空；首worker3583743/start_ticks583592768随后正常exit0。case00完成164步，独立双路线scalar状态机/预览、跟踪路由完整最优搜索、实际执行命令模型预测、当前速度与实际端点绑定、控制机械/标签/三图像及来源核验全部通过，independent verify exit0，performance_qualified=True，末尾连续80步稳定。证据outputs/loaded_velocity_control_20261025/case00/independent_verified.json。

相同18条/4907步固定批次由原081 owner继续，尚无完整性能结论，不进入WAM或fresh，不把首条通过当作总目标完成。原080等失败保持原样，081无新fit/世界模型请求/fresh。

## 前四条审查、加载恢复与剩余失败

同一execution owner3583735/start_ticks583592731再次实际核验。case00–03均完整独立核验，稳定3/4；case01 wine/demo1从080的失败转为081稳定，末92步，case02亦稳定；case03 bowl/demo40仍失败，原18/18门槛已经无法达到，剩余固定案例继续，不改代码/阈值或参考。

只读compare_loaded_velocity_loading.py的前两条已验证配对保存outputs/loaded_velocity_loading_pairs_20261025/verified_prefix_02.json。wine/demo1的native加载区间081每指平均执行器闭合驱动力16.068/15.035N，080约0.642/0.608N；全轨迹opening RMSE0.01054 vs0.01107，稳定True vsFalse。bowl/demo8仍稳定，但opening RMSE0.09510差于080的0.07391，保留该state退化，不只报告成功数。执行器力重建不等同于对象接触力，且081也改变了释放规则，不能把配对差异当作加载单因素因果证明。分析0新积分步/fit/模型/fresh。

case03/grasp_timing_diagnosis.json从已保存实际轨迹/对象运动日志读取，输入SHA前后不变：actual首次闭合cycle32，native cycle33；cycle56物体抬升0.10905m、motor[-1,+1]仍native_close_load、TCP误差4.07mm；cycle66开度由期望0.301降至实际0.035、物体高度仅剩0.01613m，motor仍[-1,+1]且command0保持、TCP误差6.11mm。说明完整闭合加载仍未阻止该次丢失，不能继续以“没有完全闭合”解释所有失败。闭合相位和抓取几何需进一步检验，未证明唯一根因；对象/接触日志仅离线分析，没有进入在线策略。

完整关闭脚本已准备，将同时报告任务、TCP/opening、相对080的state误差退化及模型支持/路由周期，不以局部成功代替总目标。

## 13条独立审查进度

同一owner3583735/start_ticks583592731仍实际存活，case00–12均完整审查，稳定10/13；失败03 bowl/demo40、06 wine/demo46、12 wine/demo11。相同前13条080为7/13；081修复01/05/07/11，但12由080成功转失败，不能把加载路由当作统一资格已通过。原18/18已失败，剩余固定5条继续，不改阈值或时间安排。

## 完整18条关闭：稳定15/18，性能门槛失败

081完整18条4907步及36个collect/verify stage全部正常退出、独立机械/两路回放/实际端点速度与状态/标签/图像核验通过。close_loaded_velocity_control.py核对execution owner实际wait0/reap及原身份消失、完整来源SHA不变。closure_verified.json已写，owner_normal_wait_verified=True；稳定15/18，失败03 bowl/demo40、06 wine/demo46、12 wine/demo11，原18/18门槛失败，不进入WAM/fresh。

对照080的12/18，081增加任务成功，但11条opening RMSE退化，regression_summary.json逐条保留变化；实际执行预测超模型速度包络0周期。不能把任务数上升当作所有state指标提升，也不能宣称总目标达成。0新fit/世界模型请求/fresh。下一步按已登记082仅检验S2A平移逆补偿，夹爪及实际控制器保持冻结。
