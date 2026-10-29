# 四步预览完整正确参考控制资格

固定18条/4907物理周期已完整关闭，18/18任务稳定且位置RMSE≤0.01m、SO3 RMSE≤0.05rad，达成原正确参考控制准入门槛。相同18条原SQP为17/18稳定、16/18综合达标；更早232为17/18稳定并达标。本结果是独立物理仿真正确参考资格，不是真机、WAM或留出新初态成绩。

## 全量证据与退出

- `outputs/coupled_preview_control_20261026/regression_summary.json`与`closure_verified.json`：18条独立机械/标签/图像检查、4907周期、所有成功采集/核验阶段、owner正常退出和来源关闭。
- `outputs/coupled_preview_continuation_20261026/execution_result.json`：case01–17共34阶段、4743新增周期，复用原case00的164周期。原case00检查器退出1未改写；单独修复后的核验退出0纳入成功阶段统计。
- `outputs/coupled_preview_continuation_closure_20261026/execution_result.json`：完整closure实际owned wait记录，性能资格True；没有新物理或模型调用。
- `outputs/coupled_preview_wam_control_admission_20261026.json`：真实生产控制gate退出0，重验完整证据表、逐条门槛和正常退出。
- `outputs/coupled_preview_wam_independent_physical_admission_20261026.json`：新独立JSON审查退出0，不导入生产控制gate，重新核对证据SHA、逐条状态/误差/KKT、36个成功collect/verify阶段及owner的实际wait0/reap和同身份进程消失。与生产准入逐字段一致。

## 关键结果与限制

| case | 位置RMSE (m) | SO3 RMSE (rad) | 末尾连续成功步数 |
| --- | ---: | ---: | ---: |
| 06 | 0.00846329008 | 0.03831363459 | 90 |
| 09 | 0.00176271524 | 0.00638609003 | 82 |
| 14 | 0.00068456874 | 0.00202269205 | 85 |
| 15 | 0.00012354490 | 0.00012595310 | 523 |
| 16 | 0.00016702799 | 0.00033750611 | 461 |
| 17 | 0.00100790819 | 0.00253727683 | 500 |

case09与14修复原SQP不达标案例；case06却比SQP误差更高，且同时是全18条位置和旋转RMSE最大值。相对083的开度RMSE有8条退化，不能宣称每个指标全面胜出。预览模型的冻结局部动力学近似也不等于完整机器人真实动力学。

## 下一阶段

完整控制前置资格已通过；新WAM服务、采集、逐响应协议和独立机械checker已实现部分接入，但新全量verifier、owner、metadata与prepare/register尚待完成。控制准入的`wam_execution_authorized=False`保持，必须先完成这些环节和真实CLI准入。

仍按237执行raw/terminal058同控制器固定12条开发，900请求/31500去噪节点/7200周期，terminal稳定≥5/6且各任务≥1/2；通过后冻结统一候选，才执行原24条留出新初态，要求≥20/24稳定且各任务≥4/8。当前0新模型请求、0 fresh，总目标未完成。
