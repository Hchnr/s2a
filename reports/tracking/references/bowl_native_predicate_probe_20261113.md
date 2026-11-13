# 碗任务第 129 端点的原生谓词诊断

2026-11-13。对[独立重放](adaptive_long_final_endpoint_replay_20261113.md)的
40 个自适应保存状态做静态 MuJoCo 查询：按原生 `On` 的高度、碗盘接触和
水平距离三个条件重建，结果与 40 个已保存原生任务标签全部一致。

| 端点 | 碗高于盘原点 | 水平距离 | 碗盘碰撞 geom 接触对 | 原生任务标签 |
| --- | ---: | ---: | ---: | --- |
| 128 | 21.134 mm | 18.231 mm | 1 | 真 |
| 129 | 21.039 mm | 20.413 mm | 0 | 假 |
| 130 | 20.515 mm | 22.161 mm | 1 | 真 |

第 129 端点的高度条件成立，水平距离也小于原生 30 mm 严格门；静态重建未
检测到登记的碗/盘 collision geom 接触，因此联合谓词为假。其余 39 个端点
三个条件均成立。该证据解释该采样端点的判定，不证明接触中断持续多久、
为何发生，或某种控制修正能够保持稳定。

## 记录缺陷与纠正

原查询程序保存了指向 MuJoCo 可变缓冲区的盘位置向量，输出中 40 行的该向量
均变成最后端点的值；最大分量误差为 `0.0019539464728098865 m`。原结果及
程序保留用于来源追溯，**其中逐行 `plate_world_m` 不可直接使用**。

[只读纠正审计](../../../scripts/simulation/runtime/audit_bowl_predicate_alias_20261113.py)
根据原 XML 的 free-joint 次序和逐端点状态 SHA 恢复盘位置，并将每行的水平
距离、高度标量与原始状态重新计算：40/40 完全一致，条件和原生标签也一致。
上表及纠正后的逐端点盘位置取自审计产物；已保存的接触对来自原静态查询，
审计没有再次调用 MuJoCo。测试还复现了原可变数组问题并验证复制后的值不会
被后续修改。

## 来源与开销

实际查询使用 1 个静态模型/data 和 40 次 `mj_forward`，没有积分步、渲染
context、WAM 请求或新动作；worker 正常退出并回收。6 个可用 TCP 图像锚点的
最大坐标差为 `1.39e-17 m`，状态最大变化为四元数归一化产生的
`2.22e-16`，其余状态逐位不变。

本阶段 32 并发和两次 64 并发 QA 各执行 1 个真实 CPU MuJoCo 合成测试，合计
3 个合成模型/data、3 次非积分 forward，与本次真实保存状态诊断分开记账。
另有此前 2 个静态诊断模型；更早测试的合成模型总数未知。原累计物理范围仍为
2,321–2,436 控制周期、58,025–60,900 内步及各 96–99 个仿真模型/context；
累计 WAM 初始化 11 次、请求尝试/完成 30/27 次、完成去噪 945 次均未增加。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/bowl_native_predicate_probe_20261113/result.json` | `1eb86fc00e6f5d65e727a3aaa24131dad4a3925e0766a0b10dc910a63b562147` |
| `outputs/bowl_native_predicate_probe_20261113/worker/result.json` | `0fa42bbbcbfc9ed8c633507d696c61945fddfea525ce350ade06cfefaaefd9f4` |
| `outputs/audits/bowl_predicate_alias_20261113/result.json` | `9f3b95a67cafbb501a9061d5a2decb1474a2e00b49a1b31c0637f38378c82870` |
| `outputs/qa/bowl_native_predicate_probe_accounted_20261113/result.json` | `38ecd0f0b791e769d61e52bc1b50e293dedb5abd672e6dddd02da173abb733a7` |
| `outputs/qa/bowl_predicate_alias_audit_20261113/result.json` | `ed7ed0e8158fc0f7f57a2d74daca1be74c08c2314f17b50b435dfd4664501681` |

最终 64 并发查询前回归 361 项、32 并发纠正审计回归 56 项通过，均零跳过、
源码一致、worker 已回收。一次旧开发初态的静态解释不替代 600 周期、多任务、
24 条留出初态、实时或真机验收；物体特权信息与任务标签未用于在线控制。
