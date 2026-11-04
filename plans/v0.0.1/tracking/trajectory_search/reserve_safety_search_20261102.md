# reserve拒绝仅Js反馈的有限新执行

2026-11-02，seed42，按66cd45e设计、2883a39实现/f92cbb5闭合验证执行。GPU禁用。
总目标与原18/12/24稳定验收保持，本地数值结果不授予物理或任务资格。

固定anchor85、f4bc1804 Js/Jq/Je、原root域和pose ceiling，原search config全字典
保持（60stage/122response）。继续使用已对齐reserve floor和原实际接纳门，唯一
新增显式ReserveSafetyFeedback：原pose ceiling来源先选；无原来源且无接纳的
pose stage才选择已计费安全reserve拒绝，预测≥entry、实际<entry，按最小alpha。
只更新Js，Jq/Je及anchor不动；stage仍按原失败缩步，不补采样，不降低实际门。

预算独立于旧组：生产≤122/独立≤122，共≤244响应/73200内步/76130查询、2model，
≤60QP/6000迭代/60NNLS。拒绝、initial/repeat全部计费。accepted/原rejected三模型
更新与新增safety-only更新分别记录，0新stencil拟合，0物理/GPU/WAM/fresh/最终
留出/实际参考消费，任何候选不下发。不追加或恢复任何旧预算。

复用共享有限协议，通过显式路径/策略参数启用，不全局替换变量。QA/source/map/
driver绑定，源码与数据SHA前后保持，CPU空CUDA/LD、数值库线程1。preflight/init
300秒、执行及独立1200秒，owner1800秒TERM/10秒KILL、实际wait/reap，无重试。

生产完整独立ledger/GN/KKT/scaling/recovery审核，逐响应完整native复算，原1e-10/
torque1e-8及布尔审核保持，全部响应/decision metric逐值exact；将native metric
替换后再次审核同一决策、来源、Js-only更新及Jq/Je不变，历史anchor/repeat exact。
新kind必须显式启用独立审核，旧默认不冒充支持。

主要比较已关闭aligned68a177b2（9stage/4接纳、loss2.009187、caps12/24、停滞），
既有baseline/reserve只作明确标记的历史上下文，不声称重跑。先软件回放核对
既有aligned完整答案，避免默认漂移。新结果报告完整loss/caps/prefix/witness、
实际reserve、停止、反馈次数/来源、QP资格及费用；多更新或多接纳不等于改善。
最终完整跟踪、独立物理与留出稳定验收仍须分别证明，不能降低门以完成本轮。
