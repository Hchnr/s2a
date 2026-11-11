# 已研究task1释放片段的暂缓闭合反馈结果

2026-11-11。按[固定方案](../../../plans/v0.0.1/tracking/gripper/defer_close_task1_feedback_20261111.md)，
复用此前已独立重放的task1/init0/seed43、terminal058、step232最新H40参考的
八周期基线。候选从同一原初态重放232条历史命令，之后依据自己的实际状态逐周期
选择机械臂和夹爪命令；没有新增WAM查询。此片段既是提出假设的样本，也是本次
评估样本，不能作为新初态泛化或完整任务成功的证据。

| 指标，相对step232最新原WAM参考 | 既有基线 | 暂缓闭合反馈 |
| --- | ---: | ---: |
| 完成周期 | 8/8 | 8/8 |
| opening RMSE / MAE | 0.280611 / 0.236080 | 0.204035 / 0.173430 |
| 最后端点opening误差 | −0.414672 | −0.296015 |
| TCP位置RMSE / 最大值 m | 0.003376 / 0.004652 | 0.003376 / 0.004652 |
| SO(3) RMSE / 最大值 rad | 0.013325 / 0.020617 | 0.013325 / 0.020616 |
| 原位置/姿态双阈值 | 8/8 | 8/8 |
| 目标grasp / 任务成功端点 | 0 / 0 | 0 / 0 |

前两周期的动作、实际状态和评分与基线一致；step234基线夹爪`+1`，候选按固定
条件发送`0`。step235候选当前状态下的原策略提案仍为`+1`，再次暂缓为`0`；
旧基线当时实际命令也是`0`，两者已处于不同状态，不能称为同状态反事实。
step236–239两组实际夹爪命令均为`−1`。机械臂在分歧后按各自状态重算，最大
单维命令差约`1.58e-5`。虽然误差减小，最后开度仍低于目标约0.296，且没有
目标grasp或成功，不能将本次结果当作释放阶段已解决。

前缀233个状态端点逐字节复现原记录；5800个2 ms前缀内步最小关节余量为
0.120029 rad，候选8周期最小为0.542496 rad，均无低于0.05 rad缓冲或越界。
候选的latch仅在实际完整周期后提交，下一周期使用真实上一条命令。checker不
导入候选策略，从原初态重放全部实际动作；前缀和反馈内步q/qvel/time、反馈
状态的最大差均为0，独立暂缓谓词最大差0。checker使用已记录的原夹爪提案与
latch重算谓词，不重新求解原策略，也不据此证明其他状态的决策正确。

producer与checker各执行240控制周期/6000内步，共480周期/12000内步；候选
历史重建和反馈共225次夹爪规划/1476225条候选序列、8次质量求解/657次响应。
实际执行、checker及监督子进程均wait/reap、exit0、无信号且无自动重试。
55项32进程CPU专项、零模型准入、全仓Ruff和格式检查通过；没有运行全量测试、
GPU、新WAM、拟合或fresh。

正确参考受载单点和此开发短窗提供了继续研究的局部证据；它们没有通过统一候选
的18条正确参考、12条WAM开发或24条新初态最终验收。下一步需要在固定相同控制
合同下扩展正确参考和其他开发轨迹的完整反馈，检查接触与长期跟踪；不使用最终
留出集合整定，也不因本例改善降低原成功门槛。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/defer_close_task1_feedback_20261111/producer/result.json` | `9db4fe1ba6d7a91b4de54fe727ab7cfe1d987b9aa9ee6acdd1d86029389083d1` |
| `outputs/defer_close_task1_feedback_20261111/checker/result.json` | `188a782b952f561a3a06312b78e7f0c16e72551e9cb21eb537a038eef9cbd894` |
| `outputs/defer_close_task1_feedback_20261111/producer/case_07_deferred/result.json` | `1c1142bdceb6b30c0d2e8a296ac3d956b01f6acab5a9926e399aa81b90aad8a0` |
| `outputs/defer_close_task1_feedback_20261111/checker/case_07_deferred/result.json` | `65fd720ade9dc270cfa9e744f0cf73ce38fd87dda85724f0e3d7c9188fd23bef` |
| `outputs/defer_close_task1_feedback_20261111/producer/workers.json` | `68b4d1ae04b5f74fb4df82335b4fac1d7c992ea5c6c0fe73f39b6f94926500fe` |
| `outputs/defer_close_task1_feedback_20261111/checker/workers.json` | `d8ca26b5e92dd152c384c8ee86512ee0abfd7503c36dfeb979835f523c8d45d1` |
| `outputs/defer_close_task1_feedback_dry_20261111/result.json` | `e25c16e8bd6e2eab9820127ccf6bd7d9eed54c0ce18f35a6c97cba1021cd15e2` |
| `outputs/defer_close_task1_feedback_supervisor_20261111/result.json` | `378032f8467987ba6f8dcecb3e832b13df4b257220304a809106697195485a09` |
| `outputs/qa/defer_close_task1_feedback_20261111/result.json` | `c8aa9e394fdf2a44eedf90101c93347587f90554f9ba2077778489f125c8c15e` |
