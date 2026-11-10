# 已生成参考的八周期因果衔接反馈

2026-11-10。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/causal_event_bridge_feedback_20261110.md)，
在已研究的task1 seed42 terminal058开发案例，从同一原初态重放360周期命令后，先
执行8周期已生成旧参考，再将手臂位姿与夹爪opening同步过渡到step360已生成的新
参考，最多继续8周期。两套参考在实验起点均已生成；没有新WAM调用、参数扫描或
读取原基线后续观测。前8周期的命令、完整状态、内环、双参考误差、夹爪、物体位置
和任务标签分别与[旧参考16周期对照](causal_event_long_feedback_20261110.md)及
[直接切换对照](causal_event_switch_feedback_20261110.md)逐项复现。

## 实际结果与限制

| 参考及指标 | 全部16个端点 | 衔接后的8个端点 |
| --- | ---: | ---: |
| 对衔接参考的实际TCP位置和姿态双阈值 | 16/16 | 8/8 |
| 对最新参考的实际TCP位置和姿态双阈值 | 7/16 | 2/8 |
| 对旧参考的实际TCP位置和姿态双阈值 | 10/16 | 2/8 |

双阈值为位置误差≤0.01 m及SO(3)误差≤0.05 rad，均在相同绝对端点计算；它们
不包含opening或任务成功。衔接自身最大位置/旋转误差分别为 `0.004806 m / 0.017398 rad`；
对最新参考的最大误差为 `0.022657 m / 0.103835 rad`。对衔接opening的最大绝对
误差为 `0.382623`，对最新opening为 `0.725564`；均为归一化开度，不是米。末端
实际opening约 `0.615958`，最新目标约 `0.975392`。

本案例使用 `libero_closed_pad_observed_endpoint_eef10_v1` 域；同任务发布标签的
[既有合同核对](../../wam/qualification/wam_tcp_time_alignment_20261020.md)将slot 9
解释为下一端点的实测pad opening，而非原生夹爪命令；它没有证明本案例历史训练
消费的精确字节。这里的参考来自[Action128解码](../../../src/s2a/references/action_reference.py)，
实际opening由[当前夹爪关节](../../../scripts/simulation/runtime/run_causal_event_feedback.py)计算，
两者差是归一化参考跟踪误差；实际发送的第7维控制命令与motor target另行记录。
不能把该LIBERO语义泛化为DROID域：后者在[原计划](../../../plans/v0.0.1/goals/001_state_to_action.md)
中记录的是来源命令转换后的opening。

16条新动作均已执行，无预设停止；其中第9条之后的8周期由固定衔接参考驱动。实际
400个2 ms内环样本的最小关节余量为 `0.764479 rad`，无样本低于0.05 rad缓冲或越
关节范围。该短窗结果不保证后续安全。

原生grasp标签在端点367–371连续为真，从372开始为假；物体相对起点最高上升
`0.013829 m`，最后约 `-0.000013 m`。这是端点标签和物体高度，不说明各周期内接触
持续情况；夹爪逐渐打开与接触丢失同时出现，但不能由此确定唯一原因。16个端点
均无任务成功。直接切换时第9条动作未派发；本次固定衔接可以继续执行，但仅是
同一已研究案例的有限窗口，不能据此声称全局可行、稳定抓取或可靠跟踪最新目标。

固定旧参考的16周期分支此前在10个连续端点grasp并抬升约5.31 cm，却仅5/16点对
最新参考双阈值达标；固定新参考则16/16点对自身达标、无grasp。这些分支在共同
起点之后产生不同状态，后续并非相同状态下的在线反事实，也不是持续重规划的世界
模型闭环。本次臂决策耗时约 `1.03–1.17 s`，超过50 ms周期，尚无实时资格。

## 独立复核与实际费用

独立checker不导入桥接生产实现或候选手臂/夹爪策略，使用SLERP从两套原参考逐点
复算40个桥接端点；最大位姿矩阵元素差 `8.88e-16`，opening差0。checker从原初态
重放实际16条新动作，对完整状态、400个内环q/qvel/time、三套参考误差、开度、
motor目标、物体位置及grasp/任务标签逐项核对；各数值最大差为0。checker没有
重新求解策略或重新执行参考切换决策。producer的16次质量witness审核均通过。

producer和checker各一个CPU worker均实际wait/reap、exit0且未发信号；源码、参数、
输入及资产在各phase前后一致。每phase实际376控制周期、9400个2 ms积分步、3个
物理模型和3个llvmpipe软件context；两phase合计752周期、18800步、6模型和6个
context。producer新选16条动作，16次质量求解使用1287个响应；夹爪历史与提案共
368次planner和2414448条完整候选序列。checker没有候选求解。151项32并发CPU
专项、零模型dry和全仓Ruff/2767文件格式检查通过；0新WAM、GPU和final fresh。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/causal_event_bridge_feedback_20261110/producer/result.json` | `99f30369de8db3a7ca5d7074c21eceb73e199b26698259ef168ea3d31185a53a` |
| `outputs/causal_event_bridge_feedback_20261110/checker/result.json` | `833d0b9331a15f98cadf2cf3e9f878c1ef6bcae4563971bf9f9878dc30465dfa` |
| `outputs/causal_event_bridge_feedback_20261110/producer/reference.json` | `4615ab0f1883c4d627288515bcf11c798053a663c86b5c7e3c9e9b4b35abe90a` |
| `outputs/causal_event_bridge_feedback_20261110/producer/bridge/result.json` | `ecb9d8b44fd9e92ae576f1b6904aad595e71cd8444155a838f4227c18232cb8a` |
| `outputs/causal_event_bridge_feedback_20261110/checker/bridge/result.json` | `65c308d7c1052756464c2defdb3533b7a8ba0419a95c5ec879cd6cd76cbf550d` |
| `outputs/causal_event_bridge_feedback_dry_20261110/result.json` | `9b7e5a33e5f150fe19f01987067fc164984c7019edae1e06e3202b6136da5a32` |
| `outputs/qa/causal_event_bridge_feedback_20261110/result.json` | `c8831e9dccdac2935b9de21357a17c0732623e36645ab451ac72881982cab256` |

下一步需要同时处理新参考要求打开与当前抓取阶段之间的冲突，并继续量化对最新
位姿及opening的实际代价；不能把保留旧抓取或跟踪衔接目标单独等同于任务完成。
原18正确参考、12开发、24最终留出及≥20/24、各任务≥4/8门槛不变，总目标未达成。
