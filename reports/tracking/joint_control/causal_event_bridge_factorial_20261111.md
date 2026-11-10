# 衔接位姿与开度的两因素实际对照

2026-11-11。依据[原方案](../../../plans/v0.0.1/tracking/joint_control/causal_event_bridge_factorial_20261110.md)
和[独占恢复方案](../../../plans/v0.0.1/tracking/joint_control/causal_event_factorial_recovery_20261111.md)，
在已研究的task1 seed42 terminal058案例，以原初态和360周期原命令前缀为共同起点，
比较旧/衔接位姿与旧/衔接opening的四种组合。旧/旧和衔接/衔接复用此前已独立重放
的两组；本次新执行其余两组，各16周期。前8周期四组相同，两条新分支的命令、完整
状态、内环、参考误差、夹爪、物体位置及标签均分别与两条既有基线复现。

## 实际结果

表中grasp和对最新参考的位姿双阈值均统计**切换后的8个端点**369–376；末端高度
为相对共同对照起点的物体z变化。双阈值为TCP位置≤0.01 m且SO(3)误差≤0.05 rad，
不包括opening或任务成功。

| 位姿参考 | opening参考 | grasp端点 | 末端物体高度变化 | 对最新位姿双阈值 | 末端实际opening | 任务成功端点 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 旧 | 旧 | 8/8 | +0.053149 m | 0/8 | 0.239901 | 0/8 |
| 旧 | 衔接 | 3/8 | −0.000030 m | 0/8 | 0.616013 | 0/8 |
| 衔接 | 旧 | 8/8 | +0.001635 m | 2/8 | 0.239750 | 0/8 |
| 衔接 | 衔接 | 3/8 | −0.000013 m | 2/8 | 0.615958 | 0/8 |

两种位姿路径下，旧opening对应的grasp端点为369–376；衔接opening对应369–371，
从372起标签为假。标签只在控制端点读取，不证明各周期内接触持续。旧位姿+旧
opening末端抬升约5.31 cm；衔接位姿+旧opening虽保持端点grasp，末端仅比起点高
约1.63 mm。因此该案例中，改变opening参考与接触保持结果相关，改变位姿路径也
明显改变物体高度；接触动力学和状态随干预共同变化，不能将单例结果推广为唯一
原因或通用门控规则。四组均无任务成功。

末端最新opening目标为 `0.975392`，而两组旧opening的实际值均约 `0.240`；保持
接触伴随显著偏离最新开度目标。新执行的两组切换后对各自选用位姿均8/8满足双
阈值，对最新位姿分别2/8和0/8；跟踪混合参考不等于跟踪世界模型原始输出。opening
为归一化端点状态，实际测量、期望参考、发送的夹爪命令及motor target分别记录。
本次混合仅是因果诊断，不冒充模型直接生成的完整参考或可部署策略。

两条新分支各16周期均执行完毕，无预设停止；各自400个2 ms内环样本没有低于
0.05 rad缓冲或越关节范围，最小关节余量分别为 `0.764475 / 0.633101 rad`。
已执行臂决策约 `1.04–1.17 s`，超过50 ms控制周期，尚无实时资格。分支从共同
起点出发但之后实际状态不同，不把后续轨迹称作相同状态的在线反事实，也没有新
WAM查询或最终留出任务验收。

## 独立复核、恢复与费用

两种混合参考均由checker不导入混合生产器独立逐点核对40个端点；旧参考对齐可用
24点，覆盖本次最多读取的索引22。新旧保存的混合参考文件SHA完全相同。checker
不导入候选手臂/夹爪策略，从原初态重放32条实际新动作；完整状态、800个实际内环
q/qvel/time、四套参考误差、opening、motor目标和物体位置最大数值差均为0，grasp
及任务标签一致。checker没有重新求解策略或重新执行参考切换决策；producer的32次
质量witness审核均通过。

恢复producer与checker各两个CPU worker均实际wait/reap、exit0且未发信号；两个
监督进程也分别等待并回收各自主执行器，源码、输入及资产在各phase前后一致。每
phase实际752控制周期、18800个2 ms内步、6个物理模型和6个llvmpipe软件context；
两phase合计1504周期、37600内步、12模型和12 context。producer选择32条新动作，
32次质量求解使用2611个响应；夹爪历史与提案共729次planner和4782969条完整
候选序列。复用基线未计入这些新增费用。原[中断运行](causal_event_factorial_interruption_20261111.md)
另有已知至少128周期/3200内步及未知额外费用，未把它写成完整结果或精确总费用。

182项32并发CPU专项、零模型dry及全仓Ruff/2779文件格式检查通过；本次未重跑
全量pytest，0新WAM、GPU和final fresh。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/causal_event_bridge_factorial_recovery_20261111/producer/result.json` | `1693239e64068685806efdb46132322c051d2063ece86c9aaed12991c75cbd69` |
| `outputs/causal_event_bridge_factorial_recovery_20261111/checker/result.json` | `ca0cbc4345a8800695fc98c17eac78c170fd44063788f6638b27997686a28406` |
| `outputs/causal_event_bridge_factorial_recovery_20261111/producer/references.json` | `952417fb54b96acd926a7b49c522a84ec3a45f5b7cbe902993660a8930f458c8` |
| `outputs/causal_event_bridge_factorial_recovery_20261111/producer_owner_exit.json` | `f0ba7fb669573d6ea39f613f13e34736a19398cb2c6ec3e60f97039ff02b4503` |
| `outputs/causal_event_bridge_factorial_recovery_20261111/checker_owner_exit.json` | `5dc2bd779be3b3c55cd07799b850b52c1eea9b692c0ec9cd3b122936a0dc3048` |
| `outputs/causal_event_bridge_factorial_recovery_dry_20261111/result.json` | `0f02f928bd58e7b867bb319fb42d12701288e2be7b1102b12034658b25926b3e` |
| `outputs/qa/causal_event_bridge_factorial_recovery_20261111/result.json` | `9af459d59ff5dbfe564c751ab0081b80eed6ea5a37543d83588a8b32250a207f` |

下一步需要研究能在合法释放时撤销保持、同时处理旧/新位姿冲突的因果策略；不能
单凭强制保持夹爪或单例抬升声称任务成功。原18正确参考、12开发和24最终留出
验收，以及≥20/24、各任务≥4/8门槛保持，总目标未达成。
