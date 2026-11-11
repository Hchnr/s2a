# 正确参考受载点的暂缓闭合对照

2026-11-11。按[原登记](../../../plans/v0.0.1/tracking/gripper/defer_close_correct_feedback_20261111.md)
及运行前的[完整前缀守护补充](../../../plans/v0.0.1/tracking/gripper/defer_close_prefix_guard_20261111.md)，
对已研究correct_08、task2/demo49、seed42，在同一初态及固定正确参考下做两分支
CPU/OSMesa对照。两分支先各重放80个原命令，81个状态端点均与原记录逐字节一致；
step80–87中唯一命令差异是step85夹爪`+1 → 0`，机械臂命令不变。

| 原命令周期 | 原参考opening | 基线实际opening / 有符号误差 | 暂缓后实际opening / 有符号误差 | 两组目标grasp |
| ---: | ---: | ---: | ---: | --- |
| 85 | 0.532448 | 0.529678 / −0.002770 | 0.530900 / −0.001548 | 均有 |
| 86 | 0.535367 | 0.529370 / −0.005997 | 0.530835 / −0.004531 | 均有 |
| 87 | 0.539773 | 0.529281 / −0.010492 | 0.530773 / −0.009000 | 均有 |

干预后这三个端点的平均绝对opening误差为0.006419→0.005026；基线motor缓存
为`[-1, 1]`，暂缓分支保持`[-0.75, 0.75]`。这只是在当前固定后续命令下的局部
比较：step86–87沿用原历史动作，反事实分支并未按其新状态重算策略。八个端点
两组均有grasp、均无任务成功；不能据此推断长期抓取或完整正确参考成功。

干预后三个端点最大TCP位置误差为0.00008540→0.00008901 m，最大SO(3)误差为
0.00035931→0.00036179 rad；两组八端点都满足原0.01 m/0.05 rad位姿门。每组
前缀2000个实际内步最小关节余量均为0.640651 rad，诊断窗口最小分别为
1.04450110/1.04450173 rad；无内步低于0.05 rad缓冲或越界。位姿误差略增，
因此结果不能概括成所有状态分量均改善。

每个分支的独立checker从原初态重放实际命令，比较全部80×25前缀及8×25诊断
内步的q、qvel、时间和诊断状态；最大差为0，grasp及任务标签一致。checker
独立复算step85提案且不导入候选策略。每阶段两分支各执行88控制周期，总计
352周期/8800内步；所有执行、复核及其监督子进程均实际wait/reap、exit0、
未发信号，没有自动重试。32进程CPU专项34项通过、0跳过，全仓Ruff及格式检查
通过；未运行完整测试集、GPU、新WAM或fresh。

此受载单点没有观察到短时grasp退化，可以继续核查已研究task1释放片段；但它不是
18条正确参考的候选资格，也不能替代12条WAM开发或24条最终留出。下一对照须按
已有片段的同一起点固定来源和动作，记录对原WAM参考的连续开度、TCP、关节内环
及接触；不能根据此单点放宽原门槛。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/defer_close_correct_feedback_20261111/producer/result.json` | `70511f80baf5c66f83814f2b1320d6bee9868593f61081d48a9f8291dd19742b` |
| `outputs/defer_close_correct_feedback_20261111/checker/result.json` | `b3601e0fec947f1bf813c5cfb801748875a13b5d0383ca25400c1143539c84e3` |
| `outputs/defer_close_correct_feedback_20261111/producer/case_08_baseline/result.json` | `d0be50983e7f0dcf37c93dd51dec87a8140005fce9a7621a95541e9eebc6a23d` |
| `outputs/defer_close_correct_feedback_20261111/producer/case_08_deferred/result.json` | `2ec12cbd4a8eec26c07535e95c7939dc2ff6c65f9323be0823a9bbc1ad040ba4` |
| `outputs/defer_close_correct_feedback_20261111/checker/case_08_baseline/result.json` | `82029ddf91f491a2718c5a53ee25173002bc702a368b4c4a72a1ffaf519daf77` |
| `outputs/defer_close_correct_feedback_20261111/checker/case_08_deferred/result.json` | `b1e8fcf25e27c618d2382052cd6dff56f9b786c37e489dab23d9d0b21bdd06b5` |
| `outputs/defer_close_correct_feedback_20261111/producer/workers.json` | `1f83d960482bb34e221339ea2c022e5e5737f70807664816f676ef8768e147ed` |
| `outputs/defer_close_correct_feedback_20261111/checker/workers.json` | `ea99da2bbd97a99e5ebb5af5249ccd16f6b6f0f614905d366a8757f25a4b108f` |
| `outputs/defer_close_correct_feedback_dry_20261111/result.json` | `884939143ee9d6e1d6b8134f3030123a4704096a134a560fc9671bff6d08c528` |
| `outputs/defer_close_correct_feedback_supervisor_20261111/result.json` | `ababa57307d336d9e7f0695ea55b807c9198021ac2fcf371fdcaac1289f042a9` |
| `outputs/qa/defer_close_correct_guarded_20261111/result.json` | `e7cb33bba18b04c1a26531dc06472bc84714c9f1e734e2c52ba5dd4ccfda810d` |
