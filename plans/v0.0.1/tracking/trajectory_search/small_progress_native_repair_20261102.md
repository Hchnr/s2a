# small-progress已执行响应的独立审核补齐

2026-11-02。原新搜索生产36响应已完成，但独立配置校验仍固定1e-4，未显式接入
登记的两字段1e-5变体，结果844c4e05保持complete=False/passed=False，不覆盖。
这属于审核接线缺陷，不证明算法性能。本补齐不重新执行生产搜索或挪用剩余预算。

在独立generic audit增加显式progress_reference：先用原冻结validator审核基准配置，
再只允许两项minimum reduction准确为1e-5，其余key/value/type逐值等于基准。
默认无reference继续执行原固定校验并拒绝新值；不修改stage_budget_search_audit。
所有实际joint/buffer/reserve/caps、原1e-10/torque1e-8审核及原最大预算保持。
生产入口同步显式传原已绑定cap config，未来错误路径也从保留answer登记真实QP/
NNLS费用。原失败头中的0QP/NNLS是未走完成分支的计数缺口，实际12QP/12NNLS，
必须从完整producer ledger计入，不将其解释为未执行。

只读原producer_final，先以修复后的独立ledger审核，再新建1个native model完整
复算原36响应，替换为native metric后再次审核决策/状态/更新。新增36响应/
10800内步/11233查询/1model，0新QP/NNLS。加原生产36/10800/11233/1model，
累计72响应/21600内步/22466查询/2model、原12QP/12NNLS，仍在原独立预算内。
0物理/GPU/WAM/fresh/最终留出/实际参考消费，候选不下发，不自动重试。

新独占root及新QA/source/map/driver绑定；比较原执行snapshot与当前源码，只允许
显式审核接线、失败计费及新repair配置/计划/脚本/测试变化，原数值控制/模型源码
均须SHA相同，变化清单保存。CPU空CUDA/LD、数值库线程1，1200秒TERM/10秒KILL，
实际wait/reap。完整response/metric、历史anchor/repeat exact，来源SHA前后保持。

补齐报告同时链接旧失败和新审核，不能把旧result改成通过或将新审核追记为旧
进程正常完成。只有新审核全通过才能评价这36个原执行响应的科学结果；即使如此
仍仅局部数值证据，原完整24caps及18/12/24稳定验收保持未完成。
