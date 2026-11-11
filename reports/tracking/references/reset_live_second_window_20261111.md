# 从复位开始的当前图像 WAM 第二窗口

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/references/reset_live_second_window_20261111.md)，
固定 v4 开发 `task2_init0_s42_raw`，从该案例现场 reset 开始使用统一分段策略。
首 8 周期采用经审核的原首个 H40，逐周期核对已审核分段 producer；实际完成后在
**本次 step8 当前双相机图像**上请求一次 8400 masked WAM，seed42、H40、35 次去噪，
随后沿用本次已完成的手臂、夹爪和内环误差历史，执行新参考的前 8 周期。只使用当前
可测状态、公开任务、实际历史和已生成参考；任务/grasp 谓词只在动作完成后评分。
这是一个已研究开发初态的 16 周期结果，不是三任务泛化或完整任务评估。

## 实际结果

| 指标 | 首 8 周期：原保存参考 | 后 8 周期：本次当前图像 WAM |
| --- | ---: | ---: |
| 实际 TCP 位置向量 RMSE | 0.000432 m | 0.003014 m |
| 实际 TCP SO(3) 向量 RMSE | 0.003623 rad | 0.006869 rad |
| 实际 opening 绝对误差均值 | 0.148953 | 0.021061 |
| 位置 ≤0.01 m 且旋转 ≤0.05 rad | 8/8 | 8/8 |
| 最小实际 2 ms 内环关节余量 | 0.597500 rad | 0.596358 rad |
| 目标物体 grasp 标签为真 | 0/8 | 0/8 |
| 原生任务成功标签为真 | 0/8 | 0/8 |

两段参考和物理状态不同，表中差异仅是描述，不构成同参考控制对照；也不能把
短窗位姿合格等同于任务成功。新窗口无预派发停止和关节缓冲损失。请求绑定的
step8 实际状态 SHA256 为 `5645afab9b3a41c11c0c7e0661d5793fcf54bddecf8a37b45a6dc4fdc57699d4`，
当前 PNG SHA256 为 `c4b266e7505b9f118e2a2fc38e537ca31b5647f8ef1cd67125f48720b614591a`；
渲染前后及等待模型期间物理状态和时钟不变。模型实际看到该图像与公开任务
`put the cream cheese in the bowl`，inactive 118 维 mask 检查通过。preflight 核验
1,004 个模型源码文件和 163 个权重文件；实际导入的 217 个文件模块逐 SHA 匹配，
另有三个无文件的 Python 命名空间包。模型进程显式绑定
`GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf`，前后空闲且结束后已释放。

新模型 prepare+推理墙钟约 **13.238 s**，8 个 20 Hz 周期的物理时长为 0.4 s；
模型推理期间仿真物理时钟暂停，符合当前仿真验收口径，但不证明 20 Hz 实时或真机能力。

## checker 失败、修复与独立审计

原独立 checker 实际重放首 8 周期/200 内步后，在 step8 图像比较处 exit1 并清理；
状态 SHA、时间、RGB 和 PNG 哈希均与 producer 一致。失败源于 producer 给渲染器的
endpoint 是 step/time/SHA 三字段，而 checker 传了含额外元数据的完整 endpoint，
旧比较错误地要求两个字典的字段集合完全相同。原失败、worker wait/reap 和成本保留。
按[修复方案](../../../plans/v0.0.1/tracking/references/reset_live_checker_repair_20261111.md)，
新 checker 从自己的完整实际 endpoint 提取同三项供 renderer 使用，完整 endpoint 仍
用于本次状态核验；图像哈希、时钟和物理数值门不变。

修复后的独立 CPU checker 完整重放 16 周期，未加载候选策略或模型；重新渲染 step8
当前图像并从本次保存的真实 WAM 响应独立解码目标。与 producer 比较，两段各 8 条
命令、完整状态、各 200 个内环时间/q/qvel、TCP 及 opening/qpos/qvel/motor 的最大
数值差均为 **0**；任务、grasp、位姿门与端点收据一致。producer/模型及修复 checker
均 exit0、实际 wait/reap、无信号；原失败 checker exit1、实际 wait/reap、无信号。
没有发生策略停止，因此无需声称独立验证无解原因。

只读总审计第一次未处理三个无文件命名空间包，第二次误把从 step8 开始的窗口交给
只接受 step0 编号的旧比较器；均在发布结果前失败，0 新模型/物理开销。修正后先
核对绝对步号，再比较完整数据；旧试验和物理门未改变。

## 预算、验证和下一步

producer 使用 3 模型/context、16 控制周期/400 内步、1 次新模型初始化和请求/35
次去噪；原失败 checker 使用 3 模型/context、8 周期/200 内步；修复 checker 使用
3 模型/context、16 周期/400 内步。三者累计 **9 模型/context、40 周期/1,000 内步、
1 次新 WAM 请求/35 denoise**，无训练或最终留出初态。producer 的两段质量响应为
27/32 次，夹爪规划各 8 次/52,488 候选序列，独立 witness 查询各 1,600 次/20,800
模型转换步。所有预算是这次试验的新增成本，不覆盖此前复位及首窗资格的成本。

运行前 32 并发专项 100 项通过；修复 checker 专项 69 项及最终只读审计专项 39 项
均通过，零跳过、xfail 和收集错误。完整 pytest 未重跑；此前仿真环境全量 4 项失败、
2 项跳过和 5 个文件收集错误仍保留。下一步应逐段扩展当前图像 WAM 的实际闭环长度，
再按预定的三个任务、seed 和参考变体进行开发对照；不得用最终留出集合调参。
原 18/12/24、600 周期和末尾 40 周期稳定成功门不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| [运行前 32 并发专项](../../../outputs/qa/reset_live_second_window_20261111/result.json) | `bca5784373dbc586ee84cf83d919ec788a1a5a8d9c11941d3c7908985bdfde55` |
| [零模型 dry](../../../outputs/reset_live_second_window_dry_20261111/result.json) | `7eaa1c778fe40877d4dcddc56d9407e5c5ebb6e03430df6439abbd854aabda61` |
| [模型和权重 preflight](../../../outputs/reset_live_second_window_20261111/preflight.json) | `09e8eb9e7b33861c053abb78b9d5b87142e6a88b040c1f72ffe47322f33f0ae6` |
| [producer owner](../../../outputs/reset_live_second_window_20261111/producer/result.json) | `a4634066e9a4ed8c7b93eeda89501b8f66b0454b0fe854c5976cf46b2c104d0f` |
| [producer worker 生命周期](../../../outputs/reset_live_second_window_20261111/producer/workers.json) | `f394a6e8578b753b2f751fc97e83446eef231dcd82d64b9c77c04e088588a227` |
| [实际 WAM 记录](../../../outputs/reset_live_second_window_20261111/producer/model/result.json) | `ced2c6970c170a8ed16b30bf26e686ad3d0d281f2a582159a60935d1a4316415` |
| [原 checker 失败](../../../outputs/reset_live_second_window_20261111/checker/result.json) | `bbdcb22927763f9a7cb6bc460155fe70adc073cc7c3da1ed17017e9fad32c93a` |
| [原 checker 部分物理](../../../outputs/reset_live_second_window_20261111/checker/simulation/result.json) | `554e49aa1cdd870b8b2e51f8cddfe23dae54885f3db55d51ad54c7df4527a850` |
| [修复 checker 32 并发专项](../../../outputs/qa/reset_live_checker_repair_20261111/result.json) | `af2f34cb6c690492bfc42342e138ae47ab854b281eecc2ee29b91a42fd7cac7c` |
| [修复 checker 零物理 dry](../../../outputs/reset_live_checker_repair_dry_20261111/result.json) | `2f98c20e3c2ce746c9f2793f91b31ff5a05dd385f212f2694f8561f24196dd2d` |
| [修复 checker](../../../outputs/reset_live_checker_repair_20261111/result.json) | `2efb8543c58de08a038dae31f5f075d84d712cc43f502832480bb51b801208b5` |
| [修复 checker worker 生命周期](../../../outputs/reset_live_checker_repair_20261111/workers.json) | `3805ba9ce1f5840e5b706d21f2f69fe9f542f58b8eb822b8218972c3012b7a8c` |
| [审计 32 并发专项](../../../outputs/qa/reset_live_second_audit_v3_20261111/result.json) | `5dc88fb116c443be2ddaa73badb185fb6fada08831be2d6d8813065c34073bee` |
| [最终只读审计](../../../outputs/reset_live_second_audit_20261111/result.json) | `60e2d1b66403de596387798ea819ad94bc58e44b8b5656f99527e35e9532966c` |

执行和审计 QA 绑定的更新前 `docs/status.md` SHA256 为
`c453f6e2fd60a9c376774d7e2853a415713d63f1715575472c0b434d151c9b9e`；
原字节见[历史副本](../../../docs/history/experiments/status_before_reset_live_second_a718e467.md.txt)。
