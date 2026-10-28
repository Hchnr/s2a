# 末周期平移恢复终端余量

按[执行前计划](../plans/v0.0.1/tail_translation_reserve_20261029.md)，本阶段固定前11周期及全部旋转命令，只改变第12周期一个平移轴。检验是否能在保持原安全约束、达标prefix9及因果响应不变的条件下恢复终端余量。该恢复不会推进cursor9，也不替代完整H12 witness或任务资格。

实现已完成：固定12候选加首末anchor共14完整调用/13唯一命令；候选生成后、模型构造前独立重建和审查全部命令。每调用重置actual368状态与completed367 latch，不使用后续反馈。生产和独立复算分别固定4200数值内步/4369查询；所有候选均保留，科学失败不提前停止。

选择沿用原九组审计，并额外要求全部非线性原始slack >=1e-8、前9点原caps以及历史/重复/因果exact。按终端值、实际扰动L∞、请求序号严格排序，只有严格改善才有selected。足量恢复使用前项全15原模型残差独立重算的E17，按原式 `terminal_selected - 1e-8 >= 10*E17` 判断；10倍经验预留不构成72维鲁棒安全保证。

独立checker完整复算全部14响应、独立重建计账和选择，并核验复算路线内部的完整前11周期、全部旋转goal及首末重复。安全buffer、prefix和足量阈值的严格标签不允许被跨实现浮点容差掩盖；两边一致的科学false可正常闭合。

## 当前验证状态

- 分项native CPU测试：core22项、独立audit20项、owner/执行接线69项通过，共111项；Ruff/format通过。owner最终scope句柄83672实际exit0，69 passed in17.30s。
- 执行测试包含逐调用初态重置、强制完整末repeat、第15调用拒绝、原E17全15点来源、1ULP安全buffer变化、低于数值比较容差的因果变化，以及命令仍合法但扰动周期错误时0query拒绝。
- 新95源/29模块统一native CPU QA已完成：766 passed、122.69秒；两个警告来自既有SLSQP合成用例。执行句柄72990实际exit0，child实际wait/reap，无信号、无重试；新95源及driver保持原字节，旧85源SHA为同字节子集。
- QA：`outputs/tail_translation_reserve_native_qa_20261029_lkpp55kq/result.json`，SHA `e421b3860faa93407194fad3cc57eabd56d16d6eea58bb46998a90dc62dde663`；日志SHA `02c23ad7dea004b1a499a6c49a398d831b433a815dee252e55f77b23b6bd73f6`。QA driver SHA `efc7ba7d10a9471b3f5c91b4fcfae6077133a6df7083ecaa362e8e67beefc08d`。
- 95源Ruff/format及producer/checker/diagnose owner的三条实际CLI dry均正常完成。五个子进程均实际wait0/reap，无信号、无重试；句柄64989实际exit0。准入记录 `outputs/tail_translation_reserve_admission_20261029/result.json`，SHA `a02ef18417130cdd2aa74a299e6e9245def56c3627a388495a066626ce3c8e44`；driver SHA `f34537226acb2f50553cf27a2e322fd7e3e670d85557b7fa2e836818e2e0bb00`。producer/checker各核2435来源，新增响应/查询0。
- 实现提交 `0503033`。14次真实局部数值响应现已完整完成；独立复算14条后发生元数据比较异常，完整来源闭合尚未完成。组件测试不作为控制效果证据。

旧85源及所有已执行证据保持原字节。GPU继续禁用，新增物理/world-model/fresh/参考消费为0；实际仍cursor8、task/grasp0，原18正确参考/12WAM开发/24最终留出门槛保持，总目标尚未达成。

## 生产响应已关闭，独立核验失败记录保留

producer及owner已正常关闭：固定14调用、13唯一命令、4200数值内步/4369查询，优化迭代0；句柄15941实际exit0。child553959/start616882219，actual wait0/reap、无信号、无重试。

| 证据 | 相对路径 | SHA256 |
|---|---|---|
| producer | `outputs/tracking_v5_tail_translation_reserve_20261029/verified.json` | `0305dc39480abf4512205f8628729fd789595d40aeca189343bc19779f029d70` |
| producer owner | `outputs/tracking_v5_tail_translation_reserve_execution_20261029/result.json` | `f604b9e31b1a4aa00a576b099065334fd04aae8722ac5710165b080c2c76b6cb` |
| check owner实际dry | `outputs/tail_translation_reserve_check_admission_20261029/result.json` | `40db64dbca0b784f2b890a91fa0e7b5e3211f4048fe4dae13494d938ec555908` |

check owner实际dry句柄68393已exit0，新增响应/查询0。独立复算句柄58216实际exit1，完成14条数值复算后在元数据审计失败，详情如下。

生产侧12候选中6条通过恢复资格，按原排序选择请求5（末周期y +.001）。其terminal为7.734855175988188e-5，扣除1e-8后的余量7.733855175988188e-5，超过固定10E17目标2.3626357467448553e-5。其他原安全slack最小3.267282409069594e-7；历史anchor、首末repeat及全部14条因果检查均exact。生产侧improvement_found与sufficient_reserve_restored均true；这些科学标签仍待独立复核，完整H12 witness仍0，不能计作cursor9推进或任务成功。

## 独立检查器失败与隔离修复登记

v1 checker完整复算14条/4200内步/4369查询，原数值比较及strict分类通过，随后在完整historical entry的`clipped`布尔数组比较时发生TypeError。原独立audit默认比较器会转float，实际checker传入的比较器直接相减；组件QA未覆盖该实际接线。ledger及fresh独立因果/恢复核验尚未完成，不能宣称原生产结论已完整独立通过。

失败checker SHA `f7582560bc9dce92f56632b8c7688c9b4403b50ce23be4dc1d68b818d3600751`；失败owner SHA `22a0be80635c2023c2cc95b2a088eb0c9b51aa5e2c362c7b4a52c97eae88da8f`；完整traceback日志 `outputs/tracking_v5_tail_translation_reserve_check_execution_20261029/capture/command.log` SHA `876e35928e2245bf60f093cf22cd2e1abd24af0e23506bafb1ad234b6d3df7a9`。child569716/start616909863实际wait1/reap，无信号、无自动重试；原95源和产物全部保留。

[check-only v2计划](../plans/v0.0.1/tail_translation_check_v2_20261029.md)新增类型明确的比较及真实保存answer回归。仅布尔/整数标签改用exact类型、shape和值比较，浮点沿原冻结比较器和容差；不重跑producer、不改候选/选择/E17/控制参数。新5源、100源/30模块QA及两实际CLI dry通过后，在独占v2目录再完整复算同14条。包括失败成本共42完整响应/12600内步/13107查询；此处为登记预算，v2尚未执行。


v2修复提交 `5496c42`，前置计划/失败记录提交 `3ee6142`。针对性native CPU测试30项通过；其中原真实producer完整answer与前项17响应档案经实际custom比较函数完成metadata ledger，新增响应/查询0。新100源/30模块正式QA为796 passed、120.35秒，句柄8283实际exit0，actual wait/reap且无信号/重试，旧95源逐项摘要保持。

QA：`outputs/tail_translation_check_v2_native_qa_20261029_y5rxr7gl/result.json`，SHA `fe36dfd12c29f6a55fd73d789cc91132e3fb726069100cf3a729743e421590d8`；日志SHA `dff58e3b438223defe52717ceecf45d754dca5cbf095a3529804710d608bc090`。QA driver SHA `865bebb41c55f5e948e815ffe8ca22015c5b42b0aa45e4559618ec50a6354fd6`。

新4条实际准入（Ruff/format/checker dry/owner dry）均actual wait0/reap、无信号/重试，句柄74506实际exit0。准入 `outputs/tail_translation_check_v2_admission_20261029/result.json` SHA `159cae7a80c7907e934869a388aab26a7269a02b44ebb105940f54be9dc6729f`；driver SHA `ca1d8218f7f7ed215033a7f3be409ae1e899d3d803a64810970a093855dac494`。v2独立完整复算现已启动；原生产未重跑，完整资格仍待正常闭合。
