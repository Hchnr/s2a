# 保存H12候选的前缀线性化诊断

36个非零callback的保存差分线性预测均把已达前缀推到约束边界，完整保存响应却全部失去该前缀。这定位了当前有限搜索拒绝整步的一项机制，支持下一步检验显式完整响应回缩；它不证明目标不可达或回缩必定成功。原完整H12 witness仍0，实际仍cursor8、task/grasp0，总目标及18正确参考/12WAM开发/24最终留出门槛保持，GPU禁用。

## 范围与执行

按[保存诊断登记](../plans/v0.0.1/prefix_trust_saved_linearization_20261029.md)，仅读取已正常关闭的candidate producer。登记发生在只读探索之后，未将已观察到的现象冒称事前预测。正式脚本未导入求解器或MuJoCo；仅从保存commands、TCP errors及request ledger重建差分模型。当前完整v2独立动力学复算仍在执行，本文算术结果不授予数值或控制资格。

实现提交`8c61f8f`。`scripts/diagnose_prefix_trust_saved_linearization.py` SHA`7fe43ba58ef93a3849c89aa19b7318d42abf0c21173c1821c26ce75032038ef0`；相应测试SHA`6c125c3e09eae61861cbc6a92cbb6de74ea57d89132ae25f63b28ea724542809`。统一native QA为`outputs/prefix_trust_linearization_native_qa_20261029_hryeqybr/result.json` SHA`7d8db6e091e9cf75637e8dbd6b1b375839d71a5790e9ec8bdbb2dba5030cffe3`，11 passed/3.96s、actual wait0、三份源码/测试/计划不变；Ruff检查与格式通过。Python3.11.15/NumPy1.26.4，空CUDA/LD、数值库单线程。

正式结果：`outputs/tracking_v5_prefix_trust_saved_linearization_20261029/result.json` SHA`0fe281c9c67dfbbc0441586ce1f680baa24539c08a22438f63d0947307f97394`。执行owner：`outputs/tracking_v5_prefix_trust_saved_linearization_execution_20261029/result.json` SHA`fcc29ced5b904a12a5d610512e35bc4ae5a456b86b178316a8a2dadf6a55b369`，actual wait0/reap0，无signal/retry，来源前后不变。绑定原producer SHA`7338118bcad2cc80644555fcee42f9257b411cf9fa6234444fe1790601386d34`及其成功owner；driver、QA日志及逐命令启动/退出记录保留。新增响应、查询、数值内步、物理步、世界模型调用和fresh案例均0。

独立算术交叉核验未导入本诊断函数，按单遍ledger另行重建45个24×72 Jacobian（77760分量）及45个callback的两套模型，共360个完整24分量向量，索引、严格标签、目标和汇总逐项一致，最大数值差和J差均0。结果`outputs/prefix_trust_linearization_crosscheck_20261029/result.json` SHA`08b6ff1f0ffdf061789389402b9918093ad85998cb832a4b5fbd7a1afcb8cfd0`；同目录`exit.json` SHA`acc64e3622fa83ddac246ac7159396c3404934d803eef08765ee7e7d2e770305`记录独立child3626309实际wait/reap0。12份输入/代码SHA前后不变、无信号/自动重试；脚本和launcher均保留。这是保存算术的一致性证明，仍不替代真实动力学或任务验证。

## 精确匹配与结果

45组完整差分、45个callback全部映射，无TCP差分来源缺口。每组核对72个轴及原±1e-4 nominal delta、command bytes、cache/new response索引和同stage请求先后。callback仅配其之前最新的完整差分组；另一套模型使用该stage已可用的anchor首组。不能用“前一个callback”猜测差分基点，例如响应2413实际配基点2340，前callback是2339。

以`q=[||e_pos||²/.01², ||e_rot||²/.05²]`逐节点排列，按保存差分计算`J`，再计算`q_linear=q_base+J@(u_callback-u_base)`。输出每条的全部24分量、J、原始符号、实际值和残差；不截断负线性预测、不用诊断容差改写原`q<=1`分类。

- 9个末callback与其最新FD基点数值相同，残差为0，单独记录；不能将它们算作9次独立的非零步验证。
- 36个非零步的线性前缀最大超限范围为`[-5.1070259133e-15, 1.9317880628e-14]`。严格符号有6个不超限、30个略超限，但正值只有舍入量级，不能宣称实质线性不可行。
- 同36步的实际前缀最大超限均为正，范围`[2.1683697230e-5, 4.9685511255e-4]`；实际严格前缀通过0/36。这里是归一化平方误差，不能读作米或弧度。
- 后期stage6/7/8的anchor均为响应1876，首组FD基点及72个响应索引2194..2265相同，三个半径可直接比较。但实际超限不随半径减半而逐步单调下降。

下表每行列出该stage四个非零callback的实际最大前缀超限，单位为`1e-5`归一化平方误差：

| stage | radius | callback response indices | 实际超限 |
| --- | --- | --- | --- |
| 0 | .01 | 73,146,219,293 | 49.686,14.250,8.220,2.341 |
| 1 | .01 | 437,510,584,658 | 11.145,3.548,6.343,12.385 |
| 2 | .01 | 802,875,949,1023 | 11.085,2.733,6.251,12.065 |
| 3 | .01 | 1168,1241,1315,1389 | 9.939,4.926,6.012,13.513 |
| 4 | .01 | 1534,1607,1681,1755 | 9.920,5.450,5.744,12.748 |
| 5 | .01 | 1900,1973,2047,2121 | 8.936,3.950,5.522,12.311 |
| 6 | .01 | 2266,2339,2413,2487 | 8.016,4.087,5.696,12.208 |
| 7 | .005 | 2560,2633,2706,2780 | 11.466,4.853,4.426,2.168 |
| 8 | .0025 | 2854,2927,3000,3074 | 8.665,3.711,5.193,4.268 |

stage anchor模型与最新模型必须区分：后续callback相对anchor的线性预测本身可能超前缀约束；这不等同于它在最新模型下也实质超限。最后三个阶段的12个非零callback相对同一anchor都改善cursor9，但全部失去前缀；相对各自最新FD基点，部分callback的cursor9实际误差反升。所有两类比较均在JSON逐条保留。

## 下一步与证据限制

现有SLSQP已经包含其自身线搜索，callback也不等于已接受迭代，不能据此称原实现“没有线搜索”。保存证据没有每次epigraph、QP乘子或merit状态；多数候选也仅保存标量safety_min，无法恢复完整安全Jacobian或定位所有callback失败关节。现象与非线性曲率、单侧FD误差和数值效应相容，本诊断不区分各自贡献。

当前候选在已失去前缀的中间状态继续完成5迭代，最后可接纳的新anchor都来自差分探针。更直接的下一候选是捕获首个非零major callback，显式沿安全anchor到该候选的方向做有限几何回缩；每点重新计算完整H12响应和九组约束，仅接受原前缀、安全和目标下降同时合格的点。保持同warm、全轨迹目标和原caps，先不同时加入额外pose margin或二阶校正，避免混淆机制。

下一候选仍须另行冻结缓存/所有响应计费、4000响应/60迭代/60阶段预算、回缩顺序、普通anchor来源与全cache witness规则，经过真实合成例、正式执行及完整独立复算。它比较截断、回缩和anchor选择的整体变化，不预先宣称只改变单个因素。当前v2未关闭且仍无完整witness，本阶段不启动新搜索或物理pilot。
