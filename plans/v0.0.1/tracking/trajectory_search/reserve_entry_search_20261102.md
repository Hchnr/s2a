# 完整刷新模型下的reserve接纳有限对照

2026-11-02，seed42，接续2aea4da/5fe09cf/53c4ad8显式设计与闭合软件验证。
本轮才执行新的连续响应，不覆盖旧结果，不追加旧60stage搜索预算。GPU禁用。

两组使用完全相同anchor85、f4bc1804新Js/Jq/Je、完整cap-distance目标和原search
config（各60stage/122response），由原cap结果的root entry继承原global lower/upper
及pose ceiling。入口必须逐值核对继承域与原结果，不能重心平移、改参考或放宽门。
唯一差别：baseline使用原接纳；reserve启用ReserveEntryAcceptance。两组独立模型、
状态和输出，允许并行运行，每组有状态依赖的搜索内部按原顺序，不跨组共享更新。

每组生产≤122/独立≤122，合计每组≤244响应/73200内步/76130查询、2model，
≤60QP/6000迭代/60NNLS。两组合计≤488响应/146400内步/152260查询、4model。
实际费用包括拒绝响应和initial/repeat，所有已用的新模型采集成本仍单列，不隐藏。
0新stencil拟合；搜索实际accepted/rejected causal secant更新照原规则登记。
0物理/GPU/WAM/fresh/最终留出/实际参考消费，任何候选均不下发。

QA/source/map/driver绑定，CPU空CUDA/LD、数值库线程1。每组模型和控制历史独立，
初态/因果历史每响应重置为同一实际状态。源/数据SHA前后不变；原完整响应与
独立逐值exact、1e-10数值/1e-8 torque、布尔audit和严格repeat保持。

先完整独立ledger/GN/KKT/scaling/recovery审核生产答案，再对每响应独立重算全量
轨迹/metric/audit，并用独立metric重建答案重新审核同一决策、拒绝反馈和state更新。
独立新决策审核显式启用原root域与reserve策略，旧fresh checker不冒充支持新原因。
生产答案与完整native响应及指标必须exact，不依赖生产metrics推导新决策。

每组preflight/init300秒、执行/独立阶段1200秒，外部owner1800秒TERM/10秒KILL，
实际wait/reap，无自动重试。异常/不完整/独立审核失败按失败保留，不改容差。

报告各组实际cost、loss/caps/prefix、witness、接受/拒绝原因、restore次数、reserve
及停止原因。减少restore但停滞或整体跟踪不改善必须如实记录。最终成功仍按全部
24caps与原安全门，物理及18/12/24留出稳定验收另行执行，总目标未完成。
