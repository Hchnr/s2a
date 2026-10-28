# 固定保存方向回缩检验（2026-10-28，CPU）

总目标仍是：固定机器人和控制接口下，将世界模型给出的期望状态轨迹，结合当前实测状态与因果历史，转换为可执行action，并在留出新初态上稳定完成任务。当前实际仍止于原cursor8、task/grasp0；本轮是局部数值响应检验，不是新物理控制或任务验收。GPU继续禁用。

## 固定问题与执行准入

[执行前计划](../plans/v0.0.1/saved_direction_backtracking_20261028.md)冻结全部16条原cursor9主分析方向和六个比例`0,.0625,.125,.25,.5,1`。使用[保存差分诊断](saved_trajectory_differences_20261028.md)原结果SHA`f918aaf2709fd3bce09ef90d32e92a5f7c255bb753e502c9caea3fb6d954e5db`，不依结果增删方向、加点、改caps或提前停止。

H12、early、late完整独立旧审计均已actual wait/reap0且无信号；1921份来源/输入引用及三组共同actual368、completed367输入通过准入。批内及跨case96条命令精确bytes互异，与旧trace交集恰为32端点，新增64内部点。每条重新从actual368 q/v和actual367旋转latch开始，禁止读actual369及之后反馈。端点重算也完整计费。

实现分三次提交：`761ca65`固定网格与完整评分；`80590c1`重置、计账、证据链及有界CPU owner；`7e17f25`独立96条重构。新执行源码、测试及计划在正式运行前冻结。

- 原生Python3.11.15、NumPy1.26.4、SciPy1.15.3，统一72项合成测试通过（4.97s），真实child wait0，源码前后SHA一致。
- QA：`outputs/backtracking_native_qa_20261028_t9zepx64/result.json`，SHA`b8b4af8eee7f49cd039713d9d21e1d8a34f7cc37d535a20622a7c981b05bfe65`。
- 全部8个新Python文件Ruff/format及producer、checker、owner实际CLI dry通过，无新响应；准入记录`outputs/backtracking_native_qa_20261028_t9zepx64/admission.json`，SHA`bcfb512c6e0bb32ef8d4b5600e1c204d873bf3c472f0f88e7380c4e2b827a0df`。
- 测试覆盖端点bytes/负零、三组固定请求和64/32计数、H12/H13各候选重置、轨迹内部因果递推、每条26H查询、保留违规候选、strict caps/buffer、实际wait/reap及来源变化拒绝。独立checker核心8个数值函数与冻结H13 checker AST相同，仅HORIZON按case绑定。
- 空CUDA/LD，数值库单线程，seed42；没有随机采样、模型请求或新初态。

## 正式运行状态

唯一producer owner已启动，尚未据文件存在宣称完成。其输出为`outputs/tracking_v5_saved_direction_backtracking_execution_20261028/`；完整producer结果将写`outputs/tracking_v5_saved_direction_backtracking_20261028/verified.json`。独立checker只在producer实际成功关闭之后启动，输出另存`outputs/tracking_v5_saved_direction_backtracking_check_20261028/`及对应`_check_execution_`目录。

| 每侧固定预算 | H12 | early | late | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 方向 | 10 | 5 | 1 | 16 |
| 响应 | 60 | 30 | 6 | 96 |
| 新独特命令 | 40 | 20 | 4 | 64 |
| 数值内步 | 18000 | 9750 | 1950 | 29700 |
| queries（含各case一个current） | 18721 | 10141 | 2029 | 30891 |

producer与checker各完整复算，计划总计192响应、59400内步、61782查询。完整保存九组约束slacks、inner/endpoint q/v、FK、目标递推及力矩。旧trace端点仅可比较已有commands/TCP errors/safety minimum/prefix/joint/feasible六字段，不声称旧档保存每条完整q/v或slacks。

正式数值结论及全方向表等待实际结果与独立关闭证据。局部下降、target达标、完整H witness及物理/任务资格分别报告，不能互相替代。即使出现完整局部模型witness，也需另行登记物理pilot及独立重放。原18条正确参考、12条WAM开发、最终24条留出门槛保持。
