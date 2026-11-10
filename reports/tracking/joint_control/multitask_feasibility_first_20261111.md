# 三任务停止/预警状态的可行性优先搜索对照

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/multitask_feasibility_first_20261111.md)，
在三个任务的四个已独立重放开发状态上，比较原质量搜索和可行性优先搜索。
每对求解使用相同的状态、参考、种子、因果历史与接口，逐项核对输入SHA和报告出的
关节、制动、终端方向及首点质量约束。每次至多160次响应和60次迭代。

| 开发状态 | 原质量搜索 | 可行性优先搜索 | 后者最近线性可行尝试的位置 / SO(3)误差 |
| --- | --- | --- | --- |
| task0 raw step48 | 无witness，122响应，60迭代 | 无witness，112响应，60迭代 | 0.011628 m / 0.066683 rad |
| task1 raw step57 | 无witness，154响应，60迭代 | 无witness，91响应，60迭代 | 0.011236 m / 0.059738 rad |
| task2 raw step34 | 无witness，61响应，37迭代 | 无witness，75响应，39迭代 | 0.017676 m / 0.034678 rad |
| task2 terminal058 step82 | 有witness，68响应，38迭代 | 有witness，69响应，41迭代 | 0.005363 m / 0.033202 rad |

首点双阈值仍为位置≤0.01 m、SO(3)≤0.05 rad。最近尝试仅是求解过程中已检查的
线性可行点，不是全局最优、物理可执行命令或质量witness。两种求解器在四个状态
的线性关节/动作/制动约束均有解；前三个状态的有限非线性搜索未找到同时满足首点
质量的计划。task0和task1的可行性优先首阶段虽然返回优化成功，其超限平方目标
分别为0.06636和0.02210，首点仍超限；task2 raw首阶段同样收敛到非零超限目标
0.29464。优化器收敛不等于满足质量约束，也不证明全局无解。

唯一共同witness出现在task2 terminal058的原预警起点，两个候选都经原独立逐内环、
制动和首点审核；最小模型内环余量分别约0.054267 rad，制动余量约0.050000 rad。
此前[实际反馈](task2_warning_feedback_20261111.md)的质量分支从同一预警起点
执行3周期后，在第4周期派发前无witness；不能将本次起点模型witness推广为完整
反馈窗口或任务成功。两个task2状态是同一已有初态、seed43的相关生成变体。

task0、task1各使用自身已完成5周期的历史，重算的最大内环误差分别为0.00084222和
0.00094618 rad；其种子是后移的最后已执行计划，并非当时未保存的原停止决策种子。
两个task2预警起点没有预测—实测误差配对，误差profile为零且范围标为未知。冻结
自由模型不包含接触和完整非线性动力学，有限搜索失败不证明物理不可达。

本次构建4个静态模型、14次`mj_forward`及种子rollout，8次求解共752次响应；两个
模型witness独立审核计费400端点查询和5200模型转换步。4个CPU worker均实际
wait/reap、退出码0且未发信号；监督进程等待并回收总执行器后退出，源码和输入
前后一致。58项32并发CPU专项和
零模型dry通过；本轮未重跑全量pytest。无物理积分、渲染context、新WAM、GPU或
最终留出数据。原18正确参考、12开发和24最终留出门槛不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/multitask_feasibility_first_20261111/result.json` | `4aa9e97a3384d6e88c7f6932a5d4aa7c4120a681a99728bcbb8dc4c831d70d09` |
| `outputs/multitask_feasibility_first_20261111/workers.json` | `fc2864c244db073c0e5db58efc88a2223a22ef9d141d2b698783866877261c7b` |
| `outputs/multitask_feasibility_first_dry_20261111/result.json` | `d0d9958f202cc2a6c410f6b352e13443bacc34c1216d54f3d1181c54ef031b54` |
| `outputs/qa/multitask_feasibility_first_20261111/result.json` | `9805728135c9d6559789e271b4a02076de1dc4cfc709f117b493020d77d4c1d1` |
| `outputs/multitask_feasibility_first_supervisor_20261111/owner_exit.json` | `fedec0b55264ae69b21f802c3373573741ca99483bde60f322829e5fd7d69c7b` |

改变搜索顺序没有在这四个状态中产生新witness。下一步应区分固定接口下首点目标
本身的可达性与有限优化搜索失败，并在更早的因果状态寻找可持续跟踪方案；实际
反馈和独立重放仍是采用候选前的必要证据。
