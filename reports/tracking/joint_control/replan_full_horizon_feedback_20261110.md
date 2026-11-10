# 新参考重规划起点的八点约束实际反馈

2026-11-10。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/replan_full_horizon_feedback_20261110.md)，
在两条原v4开发案例的同一原始端点16，使用当时已生成的new参考和统一候选；每次决策先求解
未来八点，再独立复算全部八点和关节约束，仅派发首个通过审核的动作。夹爪按已有对照固定
保持，历史只在实际周期结束后更新。两条案例均保留，未取得新WAM或留出参考。

| 案例 | 分支 | 已执行周期 | 实际位姿双阈值达标 | 2 ms最小关节余量 rad | 最大实际位置 / SO(3)误差 | 停止 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| task0 seed43 raw | new原反馈 | 8 | 6/8 | 0.631751 | 0.010787 m / 0.041922 rad | 无 |
| task0 seed43 raw | 首点质量约束 | 8 | 7/8 | 0.632307 | 0.010026 m / 0.042805 rad | 无 |
| task0 seed43 raw | 八点质量约束 | 8 | 8/8 | 0.632308 | 0.009996522 m / 0.043138 rad | 无 |
| task1 seed43 raw | new原反馈 | 8 | 3/8 | 0.687236 | 0.018665 m / 0.056127 rad | 无 |
| task1 seed43 raw | 首点质量约束 | 1 | 1/1 | 0.728863 | 0.008106 m / 0.018078 rad | step17派发前无witness |
| task1 seed43 raw | 八点质量约束 | 0 | 未执行 | — | — | step16派发前无witness |

task0的八点候选在这一窗口实际达到双阈值，优于相同起点和参考下原反馈的6/8及首点
约束的7/8；所有已执行的2 ms样本保持0.05 rad原关节缓冲，没有越界、目标抓取或任务
成功。最大位置误差距离0.01 m上限仅约3.48微米，不能外推为稳健裕量。task1在第一条
新动作派发前有限搜索未找到八点witness；无已执行周期可计成功率，也不能据此证明全局
无解。完整任务成功、接触路径及新候选的原18条正确参考资格均未验证。

task0八次已派发决策均经独立八点模型审核；实际臂命令变化率前六周期贴近0.25上限，最大
为 `0.25000000000011824`，属于浮点审核容差范围。八次决策墙钟约1.294–1.491秒/周期，
远超20 Hz的0.05秒周期；仿真等待期间物理时钟暂停，不能认证实时控制。实际TCP相对old
参考为2/8、相对new为8/8，old只作事后计分，未进入决策。

## 复核与费用

producer和独立checker各两个worker均exit0并实际wait/reap，源码及输入前后相同。checker
不导入候选决策，重放8条已派发动作；完整状态、动作、200个2 ms内环q/qvel/time、
实际TCP和两套参考误差的最大差均为0。task1停止位置及未派发得到核对；checker沿用
producer记录的无witness原因，没有重新求解或独立证明无解。

producer与checker各40控制周期/1000内步、6模型/6软件context，合计80周期/2000
内步；producer实际9次原refiner、9次八点搜索/854响应、8次独立witness审核计1600
端点查询/20800模型转换步，checker无新求解。61项32并发CPU专项、全仓Ruff/格式及
零模型dry通过；0新WAM/GPU/fresh。运行 `complete/passed` 表示执行与重放闭合，
不代表任务通过或后续候选已获资格。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `replan_full_horizon_feedback_20261110/producer/result.json` | `677f0fed299ccff22caf96eadaa1dc5476edcc44e238f16a4d332a16be82668f` |
| `replan_full_horizon_feedback_20261110/checker/result.json` | `ffe672c10d11e9c1ef412dba73dc536474c1ea1147ab93a29a5f65e989442ba0` |
| `replan_full_horizon_feedback_dry_20261110/result.json` | `da12a4adf7962cee68c9962696301c27ecb0d8eeae2fd92cb56d339dce627612` |
| `qa/replan_full_horizon_feedback_20261110/result.json` | `62c73be5d205c73f3c26b53e6989628e59b8c281e8b1cc1f1f279b3e365cb4cb` |

后续应优先处理task1无witness时的可执行参考与接触路径，并检查重规划参考连续性和计算
时间；任何新候选仍须统一通过正确参考、完整WAM开发和最终留出门槛。GPU继续禁用。
