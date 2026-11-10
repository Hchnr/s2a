# 自由关节预览的批量展开对照

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/batched_joint_rollout_20261111.md)，
在此前剖析过的同一task1 seed42开发记录step352、355、359上，对固定8周期自由
模型比较原逐内步展开与可选批量实现。每点每种实现预热1次、交替计时50次，并对
每一对完整输出逐字段核对；没有调用优化求解器、推进物理或查询WAM。

| 原记录step | 原展开中位墙钟 | 批量展开中位墙钟 | 两者中位数之比 | 全字段最大绝对差 |
| --- | ---: | ---: | ---: | ---: |
| 352 | 4.462 ms | 2.507 ms | 1.780× | `2.67e-15` |
| 355 | 4.396 ms | 2.483 ms | 1.770× | `6.22e-15` |
| 359 | 4.406 ms | 2.511 ms | 1.755× | `7.11e-15` |

这里的比值是两种中位耗时之比，批量实现的准备开销约 `0.268–0.270 ms`；计时
不包含逐字段对比。三点共153对结果、306次展开全部通过原绝对容差 `1e-10`：
包括每个2 ms关节位置及雅可比、周期末q/qvel、TCP、twist、各雅可比和离散投影
计数。合成测试另外覆盖不同horizon、近关节边界、原不支持的零旋转、独立逐步
动力学及有限差分雅可比；批量候选拒绝 `enforce_limits=true`，不能替代受投影或
接触的路径。

该局部展开基准表明存在可测加速，不证明质量求解器会选择相同命令或满足实际
安全约束。原求解每次约1秒且SLSQP本身也有显著开销；不能把子模块的1.75–1.78倍
外推为完整闭环加速或20 Hz资格。后续若要接入，须固定相同求解域、原阈值与预算，
逐项比较witness、优化输出和独立审核，并在独立物理对照中验证。

## 验证与费用

三个CPU worker均实际wait/reap、exit0且未发信号；监督进程等待并回收主执行器。
源码、输入及参数前后一致。共153次原展开和153次批量展开；求解器调用、新物理步、
WAM、GPU和final fresh均为0。63项32并发CPU专项、零模型dry及全仓Ruff/2789文件
格式检查通过；没有重跑全量pytest。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/batched_joint_rollout_20261111/result.json` | `456d35ea3312bcc94336a6af0cae98572256a11c763fa4c261a30c7c5a6eb981` |
| `outputs/batched_joint_rollout_20261111/step_352/result.json` | `0a1f80e42fd3f10fe29dd6dee905d0fd2a59e5068975ba132ef3b617e2d4b90d` |
| `outputs/batched_joint_rollout_20261111/step_355/result.json` | `910bf46c9f731de6d9c8993c347f71e75f8e7f2fc949ee5b39a2036710571a0f` |
| `outputs/batched_joint_rollout_20261111/step_359/result.json` | `1ac2105c5efb0451b0167c10138316a640589da71e69e991672880108262638a` |
| `outputs/batched_joint_rollout_dry_20261111/result.json` | `65dc37c19aa5cf76b5a7f4b429482eda53661ecf0275bfbc676e68a1e28cf764` |
| `outputs/qa/batched_joint_rollout_20261111/result.json` | `f3c10128c2383e053472ea091fb23bd5f2b417b917430dac431629c3e5ca5107` |
| `outputs/batched_joint_rollout_supervisor_20261111/owner_exit.json` | `c7eca6e87787cb00c05372303a71d3c4ca69ecafc90863ecf7bf32b0f098983b` |
