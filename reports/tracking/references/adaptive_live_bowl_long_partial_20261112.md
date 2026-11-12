# 连续重规划：长窗生产与未闭合的独立重放

2026-11-12。按[预登记](../../../plans/v0.0.1/tracking/references/adaptive_live_bowl_long_20261112.md)，
同一旧开发初态的生产者完成 147 控制周期，其中 107 为前缀、40 为自适应周期；
在已完成的第 107、111、119、120、128、136、144 周期使用当前图像请求模型。
前 115 周期与上一轮生产者的动作、状态、内步、误差及夹爪反馈最大差为 0，
两个共同图像也一致。第 147 周期达到登记的周期上限，没有派发下一动作。

生产者 147/147 位姿端点满足当前约束，全部周期 TCP 位置 / SO(3) 旋转 RMSE
为 0.0029684699840395548 m / 0.01817749342379491 rad，最小内步关节余量
0.3871836971502849 rad。40 个自适应端点有 39 个原生任务成功、第 129 端点
为假，末尾连续成功 18 个；新增窗口目标 grasp 为 1/40。模型初始化 1 次，
7/7 请求及 245/245 次去噪完成，GPU 已释放。成功谓词波动原因尚未确定。

## checker 失败与部分证据

独立 checker 已执行并记账 147 个控制周期、3,675 个内步，但在最后一个动作
完成后的额外端点读取处报 `Replay endpoint observation changed the physical state`；
worker 退出码为 1，已实际 wait/reap，原始失败收据未覆盖。此次没有记录该额外
读取前后的状态数组，因此不能断言变化分量或原因。

对已保存部分作[只读审计](../../../scripts/simulation/runtime/audit_adaptive_long_partial.py)：
40 个自适应动作、1,000 个内步中的 command、state、TCP error、inner q/qvel/time
和 gripper 最大绝对差均为 0，任务与 grasp 标签一致；7 组重采原始相机和拼接
图像文件字节一致，最后动作的端点收据也一致。此审计不改变 checker 的失败状态，
不声称最后额外读取或完整 checker 已通过。过去另一次重复端点读取曾改变物体
四元数浮点表示，但不能将其直接当作本次根因。

## 资源与来源

本轮 producer 与失败 checker 合计 294 控制周期、7,350 内步及各 6 个模型/context。
含更早缺失收据的累计区间为 2,055–2,170 控制周期、51,375–54,250 内步及各
90–93 个模型/context，另有 2 个静态模型；累计 WAM 初始化 11 次、请求
尝试/完成 30/27 次、完成去噪 945 次。只读审计未增加模型请求或物理步。

| 仓库内 `outputs/` 证据 | SHA-256 |
| --- | --- |
| `adaptive_live_bowl_long_20261112/preflight.json` | `f4975485663ae59abbd268229e74b6d8b18643f7d26c9302fad0d6d4b4e7bbeb` |
| `adaptive_live_bowl_long_20261112/producer/result.json` | `afc31cfb643b013c584a95940dba7c4b7fa427273ed53d84362da020d2e204c5` |
| `adaptive_live_bowl_long_20261112/producer/simulation/result.json` | `76f6f55da2a6cdb25508f405f722869d40a37763bd39a6f9c885059ed57b81d6` |
| `adaptive_live_bowl_long_20261112/producer/model/result.json` | `5c4752ee236e6c3022ffbafb88503a034749d29002a8b153e5320a7e54f9b75f` |
| `adaptive_live_bowl_long_20261112/checker/result.json` | `39e80cb7b0861f2fcf8d6c8b350152e2c554d0cff3002e7ada006631c5addd8b` |
| `adaptive_live_bowl_long_20261112/checker/simulation/result.json` | `82003f17ef6452f56d2ebf6d9417616dbd5d42580aa60571e8620c3935ae1abb` |
| `audits/adaptive_live_bowl_long_partial_20261112/result.json` | `acbbeb312be8e18cec2c7c24903985771855db8352957c8d143408c643e58421` |
| `qa/adaptive_live_bowl_long_final_20261112/result.json` | `f11ad146a1cfd4ef93ee364f6877bd5ea553aed8fb30bc863625d87070236a10` |

真实试验前的 64 并发相关回归 325 项通过；只读审计及相关回归 32 并发 67 项
通过，均无跳过、源码一致且 worker 已回收。当前只是单个旧开发初态的短窗，
末尾 18 周期不足原要求的 40 周期，600 周期及 24 条留出验收仍未完成。
