# 同warm H12信赖域对照

本轮按[执行前登记](../plans/v0.0.1/prefix_trust_region_control_20261028.md)比较冻结前缀求解器与有限信赖域候选。两组使用同一actual368状态、completed367控制历史及chunk46原时序H12参考；共同初始命令取自已完成独立核验的固定回缩响应`cases.prefix.entries[55]`。不使用GPU。

## 当前状态与证据范围

实现、61项同版本native CPU测试和8条实际准入命令全部通过，两组真实数值搜索已启动、尚无完成结果；完整独立检查待各臂正常退出后启动。不得将合成测试通过或程序准备就绪作为控制性能证据。实际仍止于原cursor8，task/grasp0；本轮新增物理、世界模型调用和fresh案例均为0。原18条正确参考、12条WAM开发及最终24条留出任务门槛保持，总目标未完成。

候选保留原完整响应、九组约束审计、位置/旋转caps和全轨迹目标；增加每阶段最多5迭代、固定三半径、完整响应验收后重新定中心及逐请求计费记录。普通下降可以在prefix不增长时获接纳；完整H12 witness仍必须通过全部原门槛。此对照比较整套控制流，不能单独归因于半径。

每臂最多4000唯一完整响应、60累计迭代和60阶段；每条响应300数值内步/312查询，另计每臂一次current查询。两臂不重置跨阶段预算，不免费评价warm或差分探针。独立检查重新构造全部唯一响应并核对compact trace；只有initial/selected/closest保存完整轨迹，因此仅对这三种记录比较完整q/v/FK/torque数组，候选接受anchor另逐条比较完整审计。

## 实现与准入

- 求解器：`src/s2a/prefix_trust_region_plan.py`，冻结基线仍为`prefix_preserving_trajectory_plan.py`。
- 来源准入：`src/s2a/prefix_trust_control_evidence.py`，绑定上轮96响应及完整独立检查的成功wait/reap、来源SHA和共同warm bytes。
- 生产、独立检查及进程管理分别为`scripts/probe_tracking_v5_prefix_trust_control.py`、`scripts/check_tracking_v5_prefix_trust_control.py`和`scripts/execute_tracking_v5_prefix_trust_control.py`。
- 三个组件测试分别覆盖求解器验收/差分/预算、独立分类/计费/排序及实际执行入口的重置与计费。首次同版本native QA为59通过/1失败：真实SLSQP合成例在60迭代预算内未达到测试所预期的完整witness。失败凭据保留于`outputs/prefix_trust_native_qa_20261029_0rg1svty/result.json`（SHA`7f6fff11aaf009453d3d1927ad790529535a27584ebd752e05532129674a90e7`），actual wait退出1、9份来源未变。原测试源码按原QA SHA精确恢复并保存在同目录`failed_solver_test_source.py`（SHA`5ffa7d855d6b6acde68b74af7aede240b20a434aa316fdd551c7a0e08bb26a2b`），没有重跑该失败QA。

该困难合成例有显式合法零误差命令，但固定5迭代短阶段的callback反复失去首点prefix，只能接受较小合法差分探针；845响应/60迭代后仍无witness。生产算法、登记预算、原caps均未修改。测试现在保留已知可行但有限搜索失败的证据及诚实预算终止；另一个具有严格余量的合成例由真实SLSQP找到完整witness。不能从后者宣称原H12搜索必定成功。

最终统一QA：`outputs/prefix_trust_native_qa_20261029_4oymj468/result.json`，SHA`b6835c1c5816225c3c000d9402ec057c417541287ef758c0582bd7a21436c5ee`；61 passed/5.66s，actual wait退出0，9份来源在QA前后相同。Python3.11.15/NumPy1.26.4/SciPy1.15.3，CPU及空CUDA/LD；2条SciPy bounds-clipping warning来自上述保留困难例，不是正式搜索结果。

准入凭据：同目录`admission.json`，SHA`393c4bd94a1814deb4c973a7fa18cc988fee4526ceb6c664d5074ae62e552234`。Ruff检查/格式、两臂producer/checker metadata dry、两臂owner diagnose dry共8条命令actual wait退出0，全部9份源码与QA一致，新增真实响应0。共同warm SHA`16bdfbb266ef13d2588c8116e13511d93df936d72ae8a57d19df0b01ed3acee0`；1967个来源引用、8函数独立数学AST核对通过。

实现提交：`6b411c2`为solver/测试，`9df9930`为来源、生产、独立检查和进程管理。两臂将使用上述同一QA；源、测试和登记从真实搜索启动后冻结。

## 结果与后续

两臂producer由各自持有Popen句柄的owner执行。启动快照`outputs/prefix_trust_execution_checkpoint_20261029/producer_started.json`（SHA`d2c895f028c42eb72941f354b3a4135a41bb4cf89b19049a8233ffd4a679ea00`）记录了当次实际PID/start_ticks/PPID存活核对及两臂各600条已报告响应；这是运行中快照，不是完成或wait证据。baseline tool session为29743、candidate为87474。继续工作时必须复查这些原句柄/进程，不能因观察超时重启。

两臂owner登记的1969份来源SHA映射和native QA引用完全相同。输出位于`outputs/tracking_v5_prefix_trust_{baseline,candidate}_20261028`，owner为相应`_execution_20261028`；日志为owner目录的`capture/command.log`。当前未取得producer完成、独立复核或完整witness结论，不预填预算消耗或成功数。

待两臂实际退出后记录响应/迭代、原cursor9误差、全H12前缀、witness数、每次接受anchor及活动约束。独立检查未关闭前不授予局部数值资格；即便获得完整模型witness，也须另行登记同初态物理pilot并独立重放，不能据此宣称实际控制或留出任务成功。
