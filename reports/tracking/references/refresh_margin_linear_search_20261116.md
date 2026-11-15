# 刷新余量单次线性化搜索结果

2026-11-16。按[预登记方案](../../../plans/v0.0.1/tracking/references/refresh_margin_linear_search_20261116.md)，
在[此前额外 320 次响应未找到候选](refresh_margin_targeted_search_20261116.md)的
同一 bowl / seed 43 step 16 状态，围绕已选八周期基线计划，构造一次局部
线性化问题。原动作、关节、制动和终端条件保持；第一点模型余量、第二点刷新
余量及后六点基线 slack 加 `0.02` 同时作为约束，各编码动作分量的改变量限制
在 `±0.1`。

基线的八点预测与原独立 witness 对齐，原动作和线性安全域检查通过。HiGHS 在
239 次迭代后返回 **status 2 / infeasible**；没有计划可供第二次局部模型响应
或独立 witness 审核。本次未独立验证不可行性证书，且此结果受基线线性化和固定
信赖域限制：不能声称非线性问题、其他信赖域或其他状态整体不可行，也不能把
它作为实际控制或任务成功的证据。

本次静态 worker 完成 1 次 MuJoCo 模型构造、1 次预览模型构造、1 次静态
forward、1 次模型响应和 1 次线性规划；0 物理步、0 渲染、0 WAM、0 GPU、
0 fresh。worker 正常退出并已 wait/reap；来源和源码前后一致。Ruff 与格式
检查通过；相关 CPU 回归实际使用 64 个 worker，67 项通过、无 skip，均已
wait/reap。此前独立发现的[历史源码身份测试失败](refresh_margin_targeted_search_20261116.md)
仍未修复，未包含在本次所选回归中。

| 本地证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `refresh_margin_linear_search_dry_20261116.json` | `eb66e26548033ac606064321f47646362b31f1cc2c386f0bdf16c2536f3be23c` |
| `refresh_margin_linear_search_20261116/result.json` | `645de77b607f7fe92b04414a4dd9ab79e48f496716355d094934918bbc5acbfe` |
| `refresh_margin_linear_search_20261116/worker/result.json` | `e427beeb45c02d9362c152d95fc77fa6658a7b9dee911d7fc4489dc3b3eecd0d` |
| `qa/refresh_margin_linear_final_20261116/result.json` | `b33f16ef627f25721f5df2a48358a0781daa9e6df5dc0d83dd591b68a7c19db9` |

结果保存在本仓库 `outputs/`，不随 Git 提交；原 18 条正确参考、12 条 WAM
开发和 24 条最终留出验收保持不变。
