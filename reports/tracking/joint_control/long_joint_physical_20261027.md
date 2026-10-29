# 长动作计划的原生动力学核验

2026-10-27。总目标尚未完成。执行原登记的两个快照×H8/H16四分支；这是保存计划的开环动力学诊断，没有闭环策略资格或完整任务验收。

## 结果

独立新建原生环境，恢复全部48个分支端点以及相同时段原算法端点，重新计算 TCP 和 SO(3)误差；不推进验证环境的物理时间。四分支的初始化前缀均逐点匹配原端点收据，实际控制器目标位置/朝向通过原生精度约定检查，执行的6维臂命令逐条等于保存计划。夹爪沿用同一已生成开度参考和原因果控制器。

| 案例 / H | 原算法位置 RMSE mm | 新计划实际位置 RMSE mm | 原算法旋转 RMSE rad | 新计划实际旋转 RMSE rad | 模型与实际位置偏差 RMSE mm | 模型与实际旋转偏差 RMSE rad |
|---|---:|---:|---:|---:|---:|---:|
| raw / 8 | 6.384 | 1.204 | .024375 | .031492 | 1.134 | .004579 |
| raw / 16 | 16.185 | 3.171 | .057607 | .077231 | 3.607 | .013791 |
| terminal058 / 8 | .703 | 1.122 | .002744 | .003846 | .881 | .002850 |
| terminal058 / 16 | .684 | 1.850 | .002608 | .006232 | 1.712 | .005744 |

raw 位置改善伴随旋转退化；terminal058 两项均退化。较长计划的冻结预测偏差更大。原算法为原闭环 H4 tracker 在同一保存参考上的前8/16步，新计划为一次求解后保持全部动作；因此表格是实际行为诊断，不是等价闭环策略的公平胜负认证。四分支任务成功步数均0，且执行时尚未到原参考的首次闭合点，不能用短诊断推断整段任务不可成功。

结果不支持直接推广开环长计划。下一实现应重新使用每周期实际反馈，限制长冻结模型的影响，并保护首个执行动作的朝向误差；用当前限位/预测限位信息作为统一触发条件，不能按 raw/terminal 名称选择算法。即使改善首步模型目标，也必须复核真实整段误差，随后遵循原正确参考18/18、WAM开发和留出24条门槛。

## 实现和执行记录

- 当前 worker：`scripts/run_long_joint_physical_v2.py`，原动作与已生成参考只读；前缀复现后执行保存计划，没有读取未来实测结果来改动作。
- 原 owner：`scripts/execute_long_joint_physical_v2.py`；恢复 owner：`scripts/continue_long_joint_physical.py`。
- 最终 verifier：`scripts/verify_long_joint_physical_v3.py`；实际进程 exit0。
- 独立核验：`outputs/long_joint_physical_v3_verified_20261027/verified.json`，SHA256 `c06de8d066cad0b1ce17f6ea303b0a91bd987bb27994488e0958f19f4af28956`。complete/passed 表示数值/来源复核完成，`control_qualified=false`、`lifecycle_complete=false`，不提供控制资格。
- 恢复 owner `outputs/long_joint_physical_recovery_20261027/completed.json`，SHA256 `2a8cd0bf19aee4cbaaff095c68d942170b9864d56cb0d534f13f14d48a28dc84`；两个 terminal 子进程实际 wait0/reaped，恢复 owner实际exit0。
- 当前入口及恢复/验证脚本 Ruff check 和 format check通过。此次没有新算法改动，未重复此前已通过的8项长展开/反解测试。

## 失败记录和证据边界

1. 首个版本 raw/H8 在88步前缀后执行首条动作，SciPy期望朝向与原生结果最大元素差1.03595e-8，超过1e-8断言。原始worker与输出保留。该版本计数在断言后递增，结果中 branch_steps=0，不能据此声称没有执行动作；由实际失败控制流可定位已执行1步。该次为89物理周期，不是完整分支。v2改用已有 `rotation_goal_from_command` 原生精度函数，并在 env.step 后立即记账，未放宽阈值。
2. v2 raw/H8、raw/H16分支结果均完整；owner在第二次调用独占创建的 publish 写同名 receipts.json 时失败，实际owner exit1。raw/H8的wait0/reaped收据已保存，raw/H16缺少实际退出码收据，不能补造。恢复仅执行尚未启动的两个 terminal 分支，没有重跑已完成 raw 分支。此缺口保留在最终核验元数据中。
3. 首轮 verifier在读取旧steps的不存在 tcp_error 字段时退出；保留失败日志。v3改为同一新建环境中恢复原steps的实际sim_state，直接重算相同时段原算法误差，实际exit0。验证没有新增物理执行。

四个完整分支含前缀共272物理周期；另首轮失败89周期，总新增361周期。世界模型调用和fresh初态均0。尚未证明稳定任务完成，也没有启动留出初态性能验收。
