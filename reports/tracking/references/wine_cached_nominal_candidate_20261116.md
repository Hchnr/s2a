# 已计算名义两点候选的 wine 同状态结果

2026-11-16。按[登记方案](../../../plans/v0.0.1/tracking/references/wine_cached_nominal_candidate_20261116.md)，
从同一 wine / seed42 reset 重算并逐值核对原 143 周期，只在 step143 的首次
`quality_margin_consumed` 信号处启用默认关闭的缓存候选。原流程用满 320 次
质量搜索响应、没有选出余量候选；主搜索的已计算缓存中实际有 45 个符合原始
名义两点门限及线性关节域的候选。复用其中原求解器选择的计划无需额外搜索响应，
该计划通过原独立双点、动作、关节、制动和终端审核。

**实际执行未满足原位姿门限。** 额外执行的 step143 结束后，位置误差为
`0.010711236 m`，超过原 `0.01 m` 限值 `0.000711236 m`；旋转误差为
`0.046165967 rad`。运行按 `observed_pose_cap_exceeded` 在 step144 前停止，
没有派发下一动作，也没有目标 grasp 或任务成功。该候选不采纳为默认方案，
本次结果不满足向六例扩展的预登记条件。

| 已审核模型预测与实际 | 位置误差 | 旋转误差 |
| --- | ---: | ---: |
| 第一预测点 | `0.009630988 m` | `0.047343097 rad` |
| 第二预测点 | `0.008593016 m` | `0.047219141 rad` |
| 实际执行后的第一点 | `0.010711236 m` | `0.046165967 rad` |

两点预测满足原名义门限，但没有满足更严格的模型余量；独立 witness 验证的是
该模型与约束，并不保证实际执行误差。第一点预测与实际的位置误差向量之差的
范数为 `0.001094099 m`；该额外周期内步最小关节余量为 `0.691024539 rad`，
没有内步低于原关节缓冲。局部模型候选通过而实际超限，不能等同控制成功。

对已完成的 143 周期做**事后描述**：逐周期比较当时 witness 第一预测点与
实际 TCP 误差向量，全部 143 条可比较；最近 8 条的位置向量差范数中位数 / 95
分位 / 最大值为 `1.6205 / 4.2628 / 4.5930 mm`。这只是同一旧开发轨迹上的
有限历史，不是统计保证，也没有用于本次动作选择；可作为后续研究预测残差与
控制余量的线索。

## 独立重放与成本

checker 从 reset 重放实际 144 个周期，不加载候选、不重新判断 producer 的
停止原因。命令、状态、TCP、全部内步 q / qvel / 时间与夹爪反馈相比 producer
最大差均为 0；与原基线的前 143 周期也逐值一致。两阶段各使用 144 控制周期 /
3,600 内步、3 模型及 3 个 OSMesa context；两个 worker 实际 wait/reap 并清理。
没有新 WAM、GPU 或 fresh 初态。

producer / checker `passed=true` 表示登记的执行与重放证据完整，**不表示实际
位姿门限或任务通过**。64 并发相关 CPU 回归 181 项通过、0 skip；零物理预检、
全仓 Ruff 和格式检查通过。原正确参考、WAM 开发及最终留出验收保持不变。

| 本仓库 `outputs/` 相对路径 | SHA256 |
| --- | --- |
| `wine_cached_nominal_candidate_20261116/dry.json` | `323b636ec03812bff092beda767004381dd79beffca5f3ebd5b1bd80107080b8` |
| `wine_cached_nominal_candidate_20261116/producer/result.json` | `dee615df0d4bf496bd4aa76c1bd6c1f24f4028ccfaf30c43bf929a6342bb1c42` |
| `wine_cached_nominal_candidate_20261116/checker/result.json` | `eec38ff3bab25b791ccdb302b395b44552bcab25a9f1751670ef13dfa9957fa0` |
| `wine_cached_nominal_candidate_20261116/producer/case04/worker/result.json` | `15655ae1d65f854159d22ac776a2063b76f22bc645ce52f89acd7e3f9114374f` |
| `wine_cached_nominal_candidate_20261116/checker/case04/worker/result.json` | `ff03bb71b16beb196ae48f7b97313a066422b748114daf83d5cbbdbe05c5d46d` |
| `qa/wine_cached_nominal_candidate_final_20261116/result.json` | `34118d1d16882412ce9885f9c5f7d0fe23a725ea3a495aae36617c260939ace1` |

输出不随 Git 提交；完整收据保留来源、实际轨迹和进程身份。
