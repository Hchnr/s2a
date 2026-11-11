# 双点模型余量的实际反馈对照

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/task2_margin_two_step_feedback_20261111.md)，
同一task2开发初态、seed43下，从raw step32和terminal058 step80运行严格模型余量
反馈，逐周期使用本分支的实测状态、已完成历史及已生成参考。每个候选先通过模型
前两点0.0095 m / 0.047 rad筛选，再通过原双点/八周期安全独立审核与余量复核。

| 开发起点 | 原guarded：执行 / 实际达标 | 已完成计划回退：执行 / 实际达标 | 新余量：执行 / 实际达标 | 新余量实际最坏位置 / SO(3) | 停止 |
| --- | --- | --- | --- | --- | --- |
| raw step32 | 1 / 1/1 | 1 / 1/1 | **0 / 0** | 不适用 | step32前无余量候选 |
| terminal058 step80 | 4 / 4/4 | 5 / 5/5 | **4 / 4/4** | 0.009208 m / 0.046803 rad | step84前无余量候选 |

raw在线首轮搜索消耗58次模型响应，未找到原门或新余量双点候选，因而没有派发；
这比两个已有对照少一个实际达标周期，不能据此断言模型或物理不可达。terminal058
四个实际端点的位置误差依次为0.006307、0.009065、0.009208、0.009074 m，SO(3)
误差依次为0.019680、0.033208、0.038509、0.046803 rad；四周期均满足原实际
0.01 m / 0.05 rad门，100个2 ms内环样本未出现0.05 rad缓冲失守，最小关节余量
0.142473 rad。未出现目标grasp或任务成功，未完成8周期。

terminal058原guarded四周期最坏误差约0.009656 m / 0.049547 rad；新策略的最坏
位置和姿态误差均较低，但首周期位置略高，且两条轨迹从首动作起已不同。这是同一
开发初态的配对对照，不证明泛化或统计收益；新策略持续长度与guarded相同，比原
首点回退分支短一周期。两种起点共享初态和seed，不是独立泛化样本。

terminal058四次选中候选的独立重建均满足额外模型余量；多次第二点贴近0.0095 m /
0.047 rad新门，并非有额外确定性裕度。第五次求解在step84前无原门候选，新策略
按计划停止，没有尝试guarded或后移回退。raw静态对照和在线搜索种子不同，不能将
静态失败当成在线失败的精确重放。当前证据不支持把严格余量作为统一默认；后续若
组合原搜索或已完成计划回退，须另行固定顺序、共用响应预算及独立审核，不能追认
未执行的组合效果。

## 独立重放与费用

checker没有导入候选策略，4条动作、完整状态、100个2 ms内环q/qvel/time、实际TCP
误差、grasp和任务标签逐值一致；两个停止位置及未派发状态一致，但checker未重求
无候选原因。每phase共116控制周期/2900内步、6模型和6个OSMesa context；两phase
合计232周期/5800内步、12模型和12 context。producer有6次原细化/435响应、6次
余量搜索/362响应和6次初值LP；4次候选审核消耗800端点查询/10400模型转换步。
选中决策约2.53–3.28秒，仿真等待时物理暂停，不具备20 Hz实时资格。

四个worker与监督进程均实际wait/reap、exit0、未发信号，输入和源码前后一致。66项
32并发CPU专项、全仓Ruff/格式及零模型dry通过；本轮未重跑全量pytest。无新WAM、
GPU或最终留出样本；`complete/passed`仅表示执行和重放闭合，不表示任务成功或完整
控制资格。原18正确参考、12开发和24最终留出门槛保持不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_margin_two_step_feedback_20261111/producer/result.json` | `3a94fbc681bbc360fe61b9a3b2d217f0e1d5e94a148d3c5d51505e5efbabe41e` |
| `outputs/task2_margin_two_step_feedback_20261111/checker/result.json` | `1b5aae47c8d8be3f52ad2211ab2435c82352227d719a334acda8afce3ccf4af2` |
| `outputs/task2_margin_two_step_feedback_20261111/producer/workers.json` | `48b7c0d66c7a7e5ed645eebfa11cdb913a2bd9fb98a9a4c62b4235344db12f65` |
| `outputs/task2_margin_two_step_feedback_20261111/checker/workers.json` | `ab9dc6056d964fcf70d16d28893062876e5adb5d34002e383ef11224ca177ac5` |
| `outputs/task2_margin_two_step_feedback_dry_20261111/result.json` | `621c750eeb98cfac494083166b3c36bf8ad6b3428278cb3516d6158a7940f0fd` |
| `outputs/qa/task2_margin_two_step_feedback_20261111/result.json` | `5c8ca48fc73c9745cfc8d5414cd0af09023bbf474d4d45a377045df2b64202ec` |
| `outputs/task2_margin_two_step_feedback_supervisor_20261111/producer_exit.json` | `53359f51460a4efeb812d3773cc15d83cf4c0b2d39fd0913a372fba3f0b848c9` |
| `outputs/task2_margin_two_step_feedback_supervisor_20261111/checker_exit.json` | `2474232dcc3ff0986d9d3fce5a6832d34b1235a281cf3b4afcf180e10f782720` |
