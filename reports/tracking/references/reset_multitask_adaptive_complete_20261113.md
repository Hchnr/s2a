# 三任务双 seed 自适应开发矩阵：六案例完整记录

2026-11-13。按[固定协议](../../../plans/v0.0.1/tracking/references/reset_multitask_adaptive_600_20261113.md)，
三个旧开发初态各使用 seed 42/43，同一模型、固定控制器、分段反馈和质量触发
重规划从 reset 实际运行。首 8 周期复用同图同 seed 已审核参考；随后每次根据本轮
当前图像请求模型。推理期间物理时钟暂停。六条均未走满 600 周期，也未记录任务
成功；它们不属于最终留出初态。

| 案例 | 实际周期 | 新请求 | pose 限值通过 | 位置 / 旋转 RMSE | 内环最小关节余量 | 目标 grasp / 任务成功 | 停止 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| bowl，seed 42 | 133 | 46 | 133/133 | 0.004843 m / 0.021666 rad | 0.051591 rad | 0 / 0 | 质量无 witness |
| bowl，seed 43 | 25 | 5 | 25/25 | 0.004272 m / 0.021924 rad | 0.617736 rad | 0 / 0 | 质量无 witness |
| wine，seed 42 | 156 | 19 | 155/156 | 0.002439 m / 0.010704 rad | 0.579096 rad | 0 / 0 | 质量无 witness |
| wine，seed 43 | 45 | 9 | 45/45 | 0.004035 m / 0.019329 rad | 0.580791 rad | 0 / 0 | 质量无 witness |
| cream，seed 42 | 462 | 75 | 461/462 | 0.003422 m / 0.018660 rad | 0.051079 rad | 346 / 0 | 请求预算耗尽 |
| cream，seed 43 | 204 | 59 | 188/204 | 0.005775 m / 0.026078 rad | 0.050683 rad | 0 / 0 | 质量无 witness |

cream/seed42 在 462 个端点中有 346 个目标 grasp、没有任务成功；grasp 是原生
周期末接触标签，不能等同于完成任务或证明采样间持续接触。其提前重规划消耗完
75 次请求，未走到 600 周期；不能假定单纯增加预算就能成功。cream/seed43 的
16 个实际 pose 超限中，step 203 位置误差约 0.01663 m，step 198 旋转误差约
0.05705 rad；相应质量模型的一步预测仍接近限值以内。预测 witness 因此不能
视作已经保证实测误差满足阈值。最小关节余量只覆盖记录的机械臂 2 ms 内步。

## 审核事件与独立重放

五条在质量停止后的 owner 因停止帧比较格式错误失败；原始失败、过程和成本均
保留。修复后对实际 producer/model 收据做只读复核，另从相同 reset 独立重放这
五条。cream/seed42 在请求预算耗尽时原 owner 通过，并由原 checker 独立重放。
因此原 producer group 的 `passed=false`、原 checker group 的五条跳过和整体
`complete=false` 仍然有效；补充重放不能改写这些原收据。完整六条的实际动作、
状态、TCP 误差、25 个内步关节/时间及夹爪反馈，producer 与各自 checker 最大
绝对差均为 0，任务/grasp 标签和全部请求图像一致。checker 不加载候选或 WAM，
也不独立重算质量停止原因。[首批事件](reset_multitask_postflight_replay_20261113.md)
保留测试与来源细节。

六条 producer 共 1,025 周期 / 25,625 内步，checker 另 1,025 周期 / 25,625
内步；两相位各使用 18 个仿真模型与 18 个 OSMesa context。producer 实际初始化
WAM 6 次，完成 213 次新请求 / 7,455 次 denoise；所有子进程已 wait/reap，模型
进程退出且四张 GPU 显存释放。修复前和隔离工作区的失败回归均按事件报告保留；
修复后的正式源码匹配全量相关回归另行记录。

| 收据 | SHA-256 |
| --- | --- |
| `outputs/reset_multitask_adaptive_600_20261113/orchestration/producer/result.json` | `189a3d362d75d8ac20c7c53d592ea57ade61641c5b2b4c4e6934ef8fa6d8f674` |
| `outputs/reset_multitask_adaptive_600_20261113/orchestration/checker/result.json` | `e53a670cd8ecb866fe7a2d75bd68b3d23bb2c7a2cd03ac4ad3233281652e4f6a` |
| `outputs/reset_multitask_adaptive_preliminary_remaining_20261113/result.json` | `c26a8659bb309e30fad4fa8e3ae79b6557c398426091e14c6a881f56aaea704c` |
| `outputs/reset_multitask_postflight_replay_20261113/wave_remaining.json` | `a37cbc0df8e559e48b413e6507912d5dd6662323dbbdd466c72f51835fae7fda` |
| `outputs/reset_multitask_adaptive_complete_audit_20261113/result.json` | `7fe0f7d17ddea1913ffab2f5a4243c5b8e4b6a5def9633269ebec95efc82f8fe` |

当前候选的完整正确参考正控、12 条 WAM 开发资格、24 条新初态最终验收，以及
实时和真机仍未完成；这些开发结果不能替代原门槛。
