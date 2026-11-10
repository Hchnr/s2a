# 闭合后切回已生成新参考的实际反馈

2026-11-10。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/causal_event_switch_feedback_20261110.md)，
在已研究的task1 seed42 terminal058开发案例，从原始状态及360周期原生命令前缀
出发，先跟踪step344已生成的旧参考8周期，再在端点368切回step360已生成的新
参考索引8。没有重新推理WAM或读取原基线后续状态。

前8个新动作及对应完整状态、200个2 ms内环、双参考评分、夹爪和任务标签均复现
[上一轮旧参考分支](causal_event_long_feedback_20261110.md)。其中端点367和368有
目标grasp，物体在端点368相对本对照起点上升约 `0.00375 m`，8周期均无任务成功，
实际最小关节余量 `0.80234 rad`，没有内环样本跌破0.05 rad缓冲或越关节范围。

端点368切换参考时，手臂创新、已完成周期、上一实际动作、内环误差历史和夹爪
滞回状态均保持；新参考的生成step和cursor对应相同的下一绝对端点369。切换后
首周期的有限非线性质量搜索**未找到witness**，因此第9条新动作未派发，已暂存的
夹爪提案被撤销。该结果不证明全局不可行，也不代表切换后抓取能够保持或失去。

在切换点，以实际TCP与下一端点比较，旧参考位置/姿态误差约
`0.00710 m / 0.01509 rad`，新参考约 `0.01490 m / 0.08421 rad`；实际opening约
`0.25487`，下一旧/新opening目标分别约 `0.23925 / 0.96398`。这些差异表明本次
参考变化较大，但单独不能证明搜索失败的唯一原因。此前固定旧参考继续执行时
曾出现短时抬升，不能把它当作本次未派发动作之后的实际结果。

## 复核与费用

checker从原初态重放8条实际新动作；完整状态、200个内环q/qvel/time、前一周期
25个内环、双参考误差、夹爪和物体位置最大数值差均为0，grasp及任务标签一致。
checker没有重新求解无witness，也没有重新执行切换决策；结果中对此明确标记。

producer和checker各一个CPU worker均实际wait/reap、exit0且未发信号，源码、
参数、输入与资产前后一致。每phase实际368控制周期、9200个2 ms积分步、3个
模型和3个llvmpipe软件context；合计736周期、18400步、6模型和6个context。
producer选择8条新动作、尝试9次臂细化/质量求解，其中质量求解使用749个响应；
夹爪历史及提案共361次planner、2368521条完整候选序列。checker无候选求解。
112项32并发CPU专项、零模型dry和全仓Ruff/2757文件格式检查通过；0新WAM、
GPU和final fresh。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/causal_event_switch_feedback_20261110/producer/result.json` | `8a0a2f95d71df025ffd10901b263e03db1e7d48824554cc92c26f9f6c38ec023` |
| `outputs/causal_event_switch_feedback_20261110/checker/result.json` | `86406034f61f04864475a584ad5959bad87d85401c986cd03ad2ffe32e6daf4f` |
| `outputs/causal_event_switch_feedback_dry_20261110/result.json` | `353d3e73e96ac6fb2a8c7fdd5223fef6f853777618319737a4c677ce18a183b3` |
| `outputs/qa/causal_event_switch_feedback_20261110/result.json` | `fc0a0ce79c5b5cdc0590d1bd2cd19c7d016237a7cd168de339b8175eea86e3a8` |

下一步先以该实际停止状态及当前已生成的两套参考，分析受原约束保护的衔接候选；
任何插值、持续抓取或释放选择都需保持因果输入，并分别报告对最新参考的误差、
安全、接触和任务结果。原18正确参考、12开发和24最终留出门槛不变，总目标尚未达成。
