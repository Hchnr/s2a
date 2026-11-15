# 同状态机械臂动作短窗对照

2026-11-16。按[登记](../../../plans/v0.0.1/tracking/joint_control/correct_reference_same_state_arm_20261116.md)，
两支都从 case 03 原 reset 重放候选的前 40 个动作，然后各执行 16 个固定动作：
一支续用候选机械臂动作，另一支换为原成功正控在对应时刻的机械臂动作；夹爪
命令逐周期完全相同。两支在分支点的模拟状态、控制器端点字段与夹爪 motor
缓存最大差为 0。候选动作支同时精确复现原矩阵对应的 16 周期。

| 指标（step 40–55） | 候选动作 | 原正控机械臂动作 |
| --- | ---: | ---: |
| 已执行周期 | 16 / 16 | 16 / 16 |
| 单周期 TCP 门通过 | 16 / 16 | 16 / 16 |
| 最大 TCP 位置 / 旋转误差 | 0.001806 m / 0.004357 rad | 0.000889 m / 0.001476 rad |
| 内步最小关节余量 | 1.005101 rad | 1.005097 rad |
| grasp / 任务成功端点 | 0 / 0 | 0 / 0 |
| 末端相对原成功轨迹的物体位置差 | 128.193 mm | 115.612 mm |

同状态下，两支目标碗位置最大相差 `38.501 mm`（step 53），最后相差
`18.865 mm`；两支 TCP 位置最大差为 `2.516 mm`。因此，**在这个固定的仿真
状态与动作窗口中，改变机械臂动作确实改变了后续物体轨迹**。但换入的 16 条
动作来自另一条历史状态，不是该状态下重新求解的原控制器；两支到窗口末尾仍
都远离原成功轨迹的同期物体位置。原轨迹直到 step 69 才首次成功，本短窗在
step 55 结束，不能凭 0/16 判定完整任务是否会成功，也不能据此部署历史动作。
固定动作重放没有重新运行质量 witness 求解器，不授予新的控制资格。

事后另将实验分支点与原成功轨迹的同一时刻比较：碗位置相差 `0.713 mm`、上一
周期平均位移速度相差 `9.000 mm/s`，控制器目标位置最大坐标差 `5.356 mm`、
关节速度最大坐标差 `0.01048 rad/s`；motor 缓存相同。这些数据表明两支实验
彼此同状态，但与历史成功轨迹的状态及缓存并不相同。它们不能单独证明哪个
早期差异导致最终失败，也不能保证更早替换会成功。

## 重放、预算与限制

独立 checker 从 reset 重放两支，动作、状态、TCP 误差、全部内步 q / qvel /
时间和夹爪反馈最大差均为 0，任务和 grasp 标签一致；checker 不加载候选策略。
每阶段实际 2 个 CPU worker、6 个模型、6 个 OSMesa context，分别执行
112 控制周期 / 2,800 个 2 ms 内步；4 个 worker 均正常退出并实际 wait/reap。
两支前缀的最小关节余量为 `0.554203 rad`，干预段所有内步均高于 `0.05 rad`
缓冲。运行前相关 CPU 回归 97 项通过、0 skip；事后历史比较回归 14 项通过、
0 skip，Ruff 检查及格式检查通过。

本次没有 WAM 调用、GPU 任务或新留出初态。它只提供一例、一个短窗的受控动作
对照；18 条正确参考仍为 14/18，真实 WAM 开发仍为 0/6。下一步应针对更早的
因果状态与接触变化设计可执行策略，并在完整原分母上验证，不能将此诊断当作
最终任务收益。

| 仓库内证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `correct_reference_same_state_arm_20261116/producer/result.json` | `b87c7738629a8e728ca20ae4e2600f1ebd4233233eaf2c9f44ad38248536f89d` |
| `correct_reference_same_state_arm_20261116/checker/result.json` | `7cecd6e3e9697584d98f49b60de4fc1744e9defeaeb815f38809a9729ea72f8b` |
| `correct_reference_same_state_arm_20261116/audit.json` | `ff2b7c4a345e17b73ca9c50da307daf64db178d9d49b6a7c4cabd5ea8abb7f04` |
| `correct_reference_same_state_arm_20261116/branchpoint_history_audit.json` | `e8ee6c2a9bbb33f3009373628da5ca8040ade00ba8960a9d52cc5c50d373c837` |
| `correct_reference_same_state_arm_dry_20261116.json` | `a81029cad501489cec62591d9b6630f1f4eb4f35d3713f9a35ec54713bba661e` |
| `qa/correct_reference_same_state_arm_final_20261116/result.json` | `119cc710452134fd39fd36f0a48079f96e34c3ef9338a1996a555741e69f6851` |
| `qa/same_state_arm_branchpoint_audit_20261116/result.json` | `e6650193b522b0289954c49e5cbea8ce56a69648df0825485368b346fcb3b5b3` |

输出保存在本仓库 `outputs/`，不随 Git 提交；收据保留输入与源码身份。
