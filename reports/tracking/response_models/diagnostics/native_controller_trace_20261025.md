# 093 原六baseline实际控制器追踪：完整结果

六条同初态/seed42/命令/固定接口的baseline重放及独立检查全部完成。所有prefix endpoint完整sim_state、SITE pose/twist均与086原数据在1e-9内匹配，命令逐元素一致；观察器不增加forward/step、不修改命令或返回torque。每条最后一个周期记录25次内环调用，合计150次。

| group | 显式控制周期 | 实际uncoupling | 目标周期限幅调用 | 最小力矩限幅余量 Nm | 机器人触桌调用 |
| --- | --- | --- | --- | --- | --- |
| 0 | 34 | true | 0/25 | 11.1702 | 0/25 |
| 1 | 79 | true | 0/25 | 9.7085 | 0/25 |
| 2 | 77 | true | 0/25 | 11.3607 | 0/25 |
| 3 | 158 | true | 0/25 | 9.2661 | 0/25 |
| 4 | 43 | true | 0/25 | 9.8203 | 25/25 |
| 5 | 86 | true | 0/25 | 10.0401 | 0/25 |

余量为目标周期全部7关节/25调用的min(raw-low, high-raw)，不是命令变化可用裕量。独立检查逐调用核对raw、clip与实际ctrl一致，时间间隔0.002s，覆盖一个0.05s周期。实际uncoupling=True的实例证据补齐091/092的源码默认值推断。仅在group4，25次积分前控制观察均有robot/gripper与桌面几何接触；这与092 anchor的左指触桌一致。

本结果排除“这些baseline目标周期发生执行器力矩限幅”的说法，但不排除其他prefix周期或±probe分支的限幅，后两者没有追踪内环。不能仅凭baseline力矩余量排除有限扰动分支非线性。接触记录在robot.control结束/物理积分前读取，不包含实测接触力，不能唯一归因模型误差。

准入1项观察器不变性测试及6真实CLI dry-run均通过。12个worker/checker阶段及execution owner均取得实际wait0/reap凭据，冻结来源不变；完整证据为`outputs/native_controller_trace_20261025/closure_verified.json`。独立入口`scripts/verify_native_controller_trace.py`，完整关闭入口`scripts/close_native_controller_trace.py`。

新增显式控制周期477（对应11925积分子步），6次环境初始化内部步骤未纳入该命令周期计数；世界模型调用0、fresh0、参数fit0。没有改变控制策略，不能计为新任务成功或原18/18正确参考资格。090离线资格失败及WAM/fresh关闭不变，总目标未完成。

下一步优先比较group4原±translation探针的内环接触模式和力矩，明确有限扰动是否改变接触或触发限幅；同时保持object/contact真值仅作离线诊断，在线模型仍只允许当前机器人状态与因果历史。
