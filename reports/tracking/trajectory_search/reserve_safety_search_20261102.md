# 仅Js安全反馈的实际有限搜索

2026-11-02。预登记2cc4617、实现6adf41f；全部新完整响应和两次独立ledger审核
通过。反馈带来额外局部下降，但未改善完整caps，仍不能作为最终控制方案。

## 同模型/root域的结果

固定anchor85、f4bc1804模型、原root global/pose ceiling、已对齐floor及原实际
reserve接纳，60stage/122response最大预算不变。仅新组实际运行；对照aligned
来自已关闭68a177b2，未重跑，不把历史数据计成本轮新执行。

| 指标 | 历史aligned | 新safety feedback |
| --- | ---: | ---: |
| 初始完整cap loss | 2.0100673083 | 2.0100673083 |
| 最终完整cap loss | 2.0091872116 | 2.0088178441 |
| 最终sum q | 30.03698312 | 30.03114724 |
| caps / prefix | 12/24 / 2 | 12/24 / 2 |
| 完整witness | 0 | 0 |
| 最终reserve | .911682 | .903607 |
| stage / 接纳 | 9 / 4 | 12 / 6 |
| restore stage | 0 | 0 |
| 安全only反馈 | 0 | 2 |
| 生产 / 独立响应 | 22 / 22 | 20 / 20 |
| 停止 | radius下限 | radius下限 |

新组最终joint/buffer安全，min raw slack9.05962027e-6（row4323），加权reserve
由全部原行（含terminal换算）计算，不能仅凭min raw行推断。末点旋转误差
.1310848548rad仍远高于原.05rad。相对旧aligned更多接纳、更小loss，但完整caps/
prefix未改善，且loss仍不及历史baseline2.0076879552；不能据此宣称任务收益。

## 实际反馈与末段停止

按固定来源规则实际更新2次：stage5 response13 alpha .25，stage7 response17
alpha .5，均kind=reserve_safety_only。旧pose ceiling来源仍优先，本轮未触发。
反馈本身只更新Js，Jq/Je与anchor不动；原接受步仍按原规则更新全部3模型。
资格、来源、clip/因果更新与实际计费均由独立metric再次审核。

原因计数：accepted_reduction6、actual_reduction_insufficient5、
pose_reserve_below_entry7、predicted_reduction_insufficient21。
stage8接纳后，stage9–11全部15个候选都因预测下降不足而未调用响应，
全步预测下降分别8.355619e-5、5.539340e-5、4.087523e-5，低于原1e-4门，
QP均qualified。原流程继续缩半radius至下限，预算剩余不自动挪用或续跑。

因此末段不是已证明无安全下降，也不是物理不可达证书：这些新候选没有完整
响应证据。不能直接调低门或把预测小幅下降算成功。下一步应固定验证未调用
方向，区分停止策略与实际前向限制，并重新核对其对完整可靠跟踪的意义，
避免继续仅按微小标量loss改善挑选方法。

## 新执行费用和验证

实际12QP/264迭代/12NNLS，20生产+20独立响应，共40/12000内步/12482查询、
2model；在登记上限内，0新stencil拟合。6个accepted更新及2个安全only更新
分别记录，不隐含新QP/响应。0物理/GPU/WAM/fresh/最终留出/实际参考消费，
任何候选均未下发，不放宽原24caps、原实际安全或reserve门。

生产完整ledger/GN/KKT/scaling/recovery先独立重建，随后每响应完整native复算；
原1e-10/torque1e-8布尔审核保持，response/decision metric逐值exact，以native
metric替换后再次审核全部决策、来源及安全only更新。历史anchor/repeat exact，
源码/输入前后SHA不变。所有新native轨迹与metric独立保存。

71项jobs32专项全部通过，无skip/collection错误，worker回收、来源前后不变。
Ruff/format与Git空白检查通过，未运行全仓回归。旧aligned9QP/22响应软件回放
完整答案SHA保持a8a5dca55e3b6bd068c12dcb1e040abde99bb2ac89b30411005bd61cf5022692，
独立ledger通过；回放没有新增QP或响应费用。

CPU空CUDA/LD、数值库线程1。tool session86436退出0并实际wait/reap，PID240355
确认不存在，1800秒TERM/10秒KILL和phase alarm未触发，无重试或自动恢复。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 完整新结果 | reserve_safety_search_20261102/safety_feedback/result.json | 93deb627f6bd51f5830255a05aaab58d835c58492cb9f10a84af35a62490b9ef |
| 新旧比较分析 | reserve_safety_search_20261102/comparison_analysis.json | 522fcbb9a9d416b7bd48b04412ebebe17c387dd5e80579c4e32f817a126bead2 |
| owner终态 | reserve_safety_search_owner_20261102/observation.json | 4ea7ec0254ca176dec3e31ebecbad78f242e6ebe5e07c40a4b5fd655b9021565 |
| 71项QA | qa/reserve_safety_search_20261102/result.json | ed7bc984c0409d80b4149a3eb8e2eefe9eee28cf72bcc36354a9796b869ca68e |
| 旧aligned回放 | reserve_safety_search_implementation_20261102/aligned_replay.json | c79db194037fbd691fb339c3bb227eb7967b45d62eded9a2694d32dd3b2b67dc |

本轮重新核对036总目标及完整验收要求：局部H12数值、测试或软件回放不替代
固定控制器后的可靠跟踪、独立物理及18/12/24稳定验收。总目标仍未达成，
最终留出关闭，GPU继续禁用。
