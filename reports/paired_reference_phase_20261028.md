# 提前与临近参考相位分配（2026-10-28，CPU）

总目标仍是固定机器人/控制接口下的实际轨迹可靠跟踪和留出新初态任务稳定成功。GPU继续禁用。本对照按[执行前登记](../plans/v0.0.1/paired_reference_phase_20261028.md)在同actual368、已生成chunk46和650ms总时长下，仅比较额外50ms放在5→6或8→9段；保留全部12个原节点、原空间caps、安全约束和未消费后缀。原时序cursor0..12误差与新13相位误差分别记录，不将调速后的预测合格当成原时序控制通过。

## 实现及执行准入

- `1034daf`：参考合同、几何中点、两种时序及原节点评分、严格warm准入和30项合成测试。
- `9406fbd`：独立case入口、唯一响应缓存、两组同源owner及8项计账测试。
- `d3d70df`：完整独立H13重构、严格分类、原时序/新相位/原节点三套误差及10项合成测试。
- `68c6515`：最终执行合同；原prefix全量复核已在`6093134`关闭。

最终正式Python3.11.15/NumPy1.26.4/SciPy1.15.3环境的三个新测试文件及原prefix solver/ledger回归共81项通过（pytest 3.43s、实际exit0）。[持久QA](../outputs/paired_phase_native_qa_20261028_6l6nkn_q/result.json) SHA `e47c7c908b70812a1aa8ab7778a4f31d06224288eb4b0cd587b811a11e63e61a`，包含命令、日志SHA、运行环境和最终源码/计划/测试前后指纹，来源未变；owner对该记录的准入实际通过。新7个Python文件Ruff/format通过，三个CLI help通过。该范围不是全仓库检查，也不是实际控制性能证据。

执行前另行只读核对1864个前置来源，以及实际chunk46两组21字段：目标差异0、warm/used_start字节一致、两组命令准入通过、后缀12..39保留；该准备过程未调用MuJoCo或新增真实case响应。独立核心query/recurrence与已关闭prefix checker数学AST一致。两组动态warm随后均在正式唯一预检中通过；最小nonlinear safety slack分别为0.00735900207和0.04034866571，均未触发准入拒绝。

## 当前状态与结果解释

两个producer均已正常结束，各4000唯一完整H13响应、50共享迭代、1300000数值内步、1352001次query，均因共享响应预算耗尽而停止，完整witness为0。预检和solver初值恰好复用一次：每组4001次predict调用、1次cache hit；合计8000响应/100迭代/2600000内步/2704002次query。early搜索耗时733.140833s，late721.805304s；两组进程部分并行，不用壁钟差异比较方法效率。以下性能数字仍是生产归档及保存数据算术，完整独立动力学复算正在进行。

| 指标 | early：5→6加50ms | late：8→9加50ms |
| --- | --- | --- |
| 动作/关节合格候选 | 1294/4000 | 678/4000 |
| selected / closest trace（从0计数） | 3440 / 2705 | 2396 / 2316 |
| 新13相位连续预测达标端点 | 6→10 | 6→10 |
| 原节点连续预测达标数 | 9/12 | 9/12 |
| 原时序连续预测达标端点 | 8/13 | 9/13 |
| 原cursor9位置/旋转误差 | 0.011150609m / 0.05575181rad | 0.011183500m / 0.05652764rad |
| 完整新路径witness | 0 | 0 |

10个新端点包含一个插入中点，不能写成原参考推进到cursor9：两组仍仅原cursor0..8达标，原cursor9在物理endpoint index10失败。原H12 prefix selected在同节点的误差为0.011516282m/0.05758048rad；多给50ms并重新优化仅略缩小该局部缺口，两组依旧同时超过原0.01m/0.05rad。early原时序前缀还从原H12的9降为8，不能省略这种退化或把新相位评分替代原时序。有限局部搜索未找到解，不构成物理不可行证明。

两组终止阶段均尝试修复新endpoint index10（原cursor9）：early从响应1107至4000、36迭代，late从1028至4000、37迭代，均未增加原节点连续合格数。不根据预算耗尽宣称优化收敛。

## 正常退出及独立复算

两个producer owner的1876项来源SHA表完全相同，结束前来源检查通过。root工具会话99422和83708均实际取得exit0；下表child的actual wait/reap0、无信号、无原始错误，PID/start_ticks与launch一致。没有重试或覆盖。

| 组别 | owner工具会话 | child PID / start_ticks / parent | producer证据 |
| --- | --- | --- | --- |
| early | 99422 | 3225627 / 613724158 / 3225434 | [registration](../outputs/tracking_v5_paired_reference_phase_early_execution_20261028/registration.json)、[launch](../outputs/tracking_v5_paired_reference_phase_early_execution_20261028/capture/launch.json) |
| late | 83708 | 3226886 / 613730067 / 3226782 | [registration](../outputs/tracking_v5_paired_reference_phase_late_execution_20261028/registration.json)、[launch](../outputs/tracking_v5_paired_reference_phase_late_execution_20261028/capture/launch.json) |

| 归档 | SHA256 |
| --- | --- |
| [early verified](../outputs/tracking_v5_paired_reference_phase_early_20261028/verified.json) | `9f751f169790027391cfc9317e71a4687245b9b0688e01d8c0c83967b84f5101` |
| [early registration](../outputs/tracking_v5_paired_reference_phase_early_20261028/registration.json) | `d38a7164c73e3904549acd8dfb1d124468ea9ac99c8d1c87522c92cdd40c3b43` |
| [early owner result](../outputs/tracking_v5_paired_reference_phase_early_execution_20261028/result.json) | `d73b33655dde4d9bc3dea51c8acb574e6121de5a2e06f94c00a5bd6082b76050` |
| [late verified](../outputs/tracking_v5_paired_reference_phase_late_20261028/verified.json) | `8c817b99c99405ce2b5c29e8e0221279b461190b4a929a742ca9500c01d8be1e` |
| [late registration](../outputs/tracking_v5_paired_reference_phase_late_20261028/registration.json) | `ccf7cdf6ec3a295c7d0dac732379b63616a2ce5620f183d993b64ead93462149` |
| [late owner result](../outputs/tracking_v5_paired_reference_phase_late_execution_20261028/result.json) | `02cafc5370235432d9905dd06968cbb3fb95b1714d6fe073c6589fa5d3f35bfe` |

两个完整独立checker已分别由新owner唯一启动，时限各5400s、空CUDA/LD和数值库单线程；首次核查同PID/start_ticks/parent实际状态R。early工具会话81399，child3240665/start613804281/parent3240423；late会话15578，child3241251/start613807509/parent3241135。输出分别为`outputs/tracking_v5_paired_reference_phase_{case}_check_20261028`，不覆盖生产结果。每组必须核对全部4000响应/1300000内步/1352001查询、严格双合同分类、原节点映射和阶段账本，并取得owner实际wait/reap；当前尚无完整独立资格。

登记的命令拒绝和动态warm拒绝分支本次都未触发。两组完整搜索仍无witness，故不进入13周期物理pilot；先关闭全量独立复算，再依据完整约束/误差证据决定下一干预，不追加当前预算或悄悄增加起点。

当前新增物理、世界模型调用和fresh案例均为0，实际仍止于cursor8，task/grasp0。若找到完整模型witness，仍须另登记13周期原生物理及独立重放；本计划不自动派发动作。原18条正确参考、12条WAM开发、最终24条新初态和全部任务门槛保持，总目标未完成。
