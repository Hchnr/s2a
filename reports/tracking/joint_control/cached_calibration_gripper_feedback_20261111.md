# 原参考与校正参考的三任务夹爪物理反馈

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/cached_calibration_gripper_feedback_20261111.md)，
在三个任务各一个已研究初态、seed43的原始实际历史上，分别运行同一raw H40
参考和冻结的校正参考；机械臂使用严格shared-shift细化，夹爪恢复原历史并按
各分支参考产生实际命令。每组最多8周期，原位置≤0.01 m、SO(3)≤0.05 rad
及0.05 rad关节缓冲门不变。结果**不支持将该校正作为统一完成方案**。

| 开发起点 | 参考分支 | 实际执行/8 | 对所选参考位姿达标 | 对原raw位姿达标 | 所选参考最大位置/SO(3)误差 | 最小实际内环关节余量 | 停止 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| task0 step16 | raw | 8 | 8/8 | 8/8 | 0.009503 m / 0.047325 rad | 0.632306 rad | 完成短窗 |
| task0 step16 | calibrated | 8 | 8/8 | **3/8** | 0.009465 m / 0.044358 rad | 0.632300 rad | 完成短窗 |
| task1 step16 | raw | 1 | 1/1 | 1/1 | 0.006692 m / 0.027885 rad | 0.728850 rad | 无合格候选 |
| task1 step16 | calibrated | 0 | 0/0 | 0/0 | 未派发 | 未派发 | 无合格候选 |
| task2 step32 | raw | 1 | 1/1 | 1/1 | 0.008680 m / 0.014259 rad | 0.449920 rad | 无合格候选 |
| task2 step32 | calibrated | 2 | **1/2** | **1/2** | 0.010056 m / 0.032126 rad | 0.356534 rad | 无合格候选 |

task0校正组相对**原raw参考**的最大实际SO(3)误差为0.081669 rad；跟随修改后
目标达标不等于跟随世界模型原目标达标。task2校正组第二个端点的位置误差超过
原0.01 m门约0.000056 m；其模型审核通过不代表实际端点必然达标。task1校正组
没有派发新动作，不能从0/0计算通过率。有限搜索无候选不证明真实物理不可达。

20个实际新周期均无目标grasp或任务成功，夹爪并非固定为零：例如task0 raw组
命令依次为`[1,-1,0,0,0,0,0,0]`，校正组最后一个周期为`1`。全部500个新增
2 ms内环样本未越关节范围或跌破0.05 rad缓冲；仅覆盖本次实际执行的周期，
不保证原前缀、未执行周期或其他初态的安全。各分支使用已经生成的固定chunk，
其后续参考不会随分支新观测重新生成；这里没有新WAM闭环或600步任务证据。

## 独立重放与已保留失败

原checker在进入物理前因加载`loaded_velocity_control`触发策略隔离而6/6退出1；
原代码、日志、失败结果及正常reap记录保留，没有把失败改成通过。按
[修复登记](../../../plans/v0.0.1/tracking/joint_control/cached_calibration_replay_repair_20261111.md)，
新checker在不导入臂/夹爪候选和校准生产函数的进程中，从原初态重放6组已派发
动作，独立重算两个参考。20周期的完整状态及500个内环q/qvel/time最大差均为0；
夹爪开度/电机、双参考误差、grasp和任务标签核验通过。checker不重新求解停止
原因，`quality_no_witness`保持生产侧记录。

生产和新checker各148控制周期/3700内步、18个MuJoCo模型/18个OSMesa context；
合计296周期/7400内步，含原前缀重放。生产有24次严格质量调用/1313次模型
响应、20次witness审核；24次夹爪proposal中20次提交、4次因臂方案停止而回滚。
生产前缀夹爪历史共重建839808个候选序列，分支中再检查157464个；这是planner
费用，不是20个任务样本。没有新增WAM、GPU或fresh。

生产和修复重放各6个worker及各自监督进程均实际wait/reap、exit0、未发信号；
原checker的失败worker也已全部回收。生产专项52项、修复专项21项以32进程
通过，0跳过；全仓Ruff与格式检查通过，未重跑全量pytest。生产源码提交
`41327a8c`，独立重放源码提交`2ad94bde`。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/cached_calibration_gripper_feedback_20261111/producer/result.json` | `da29aa0fade42b572c427cedad831f2dcc8a3f3de54da900af9b983fa874f55b` |
| `outputs/cached_calibration_gripper_feedback_20261111/checker/result.json`（原失败） | `2d62c48d37634dc659b0ad71ba655851c6484c0c72e49e8a11a5bc5cd869fcf3` |
| `outputs/cached_calibration_gripper_replay_20261111/result.json` | `dbdb9ae6f1a222687af8e4f3c7b29f4431ab63e29511f80370aaecdb9af5e84e` |
| `outputs/cached_calibration_gripper_replay_20261111/workers.json` | `bd88b0cb79e54c84ba227f4450ff76a2f97a3295af9ee448e66abea24ac85066` |
| `outputs/cached_calibration_gripper_replay_dry_20261111/result.json` | `5a00c1c129e1404e5f7a974fed8a1750b2388f999a008c120ae132996729859d` |
| `outputs/cached_calibration_gripper_supervisor_20261111/producer_exit.json` | `39119a881532c5fc89c25adc30742a62b3e94fd227ff04122ac6b8ef4614983c` |
| `outputs/cached_calibration_gripper_supervisor_20261111/checker_exit.json` | `9754e2d4c5cb2a706c66bc128eeefd000e6e3adf2eea4f1ae08b24d79fd941b0` |
| `outputs/cached_calibration_gripper_supervisor_20261111/repair_replay_exit.json` | `c08a7fb4bc22868915ac49dbedd4b58bf74b972b3706f512f70a87ede5ae7872` |
| `outputs/qa/cached_calibration_gripper_feedback_20261111/result.json` | `1971b55fad6ec574d5537bb56be503bd653812034f695e96738338125738447b` |
| `outputs/qa/cached_calibration_gripper_replay_20261111/result.json` | `11edbbbc974d0f6043611f8320c6e0280c8183e33a9a924ffc457b2b57aee22e` |

离线状态门收益不能替代原参考实际跟踪和抓取；下一候选需同时约束原生成目标、
控制可行性和夹爪事件，并在多任务完整物理中验证。原18条正确参考、12条WAM
开发、24条留出及≥20/24、每任务≥4/8的稳定成功门槛保持。
