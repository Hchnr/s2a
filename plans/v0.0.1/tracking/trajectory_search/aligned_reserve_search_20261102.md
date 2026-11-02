# pose下限对齐的有限连续迭代

2026-11-02，seed42，按8358769设计、996a90a实现及78a4225完整新方向证据执行。
GPU禁用，CPU only。总目标仍为真实可靠跟踪和原18/12/24稳定验收。

单个aligned组使用同一anchor85、f4bc1804刷新Js/Jq/Je、原root global/pose ceiling、
完整cap-distance目标及原search config（60stage/122response）。显式启用pose
reserve floor对齐与原实际reserve接纳门，restore和反馈资格保持原样。
所有旧默认及已关闭失败结果保持，不追加任何旧搜索预算。

沿用已验证有限搜索/完整native复算入口，仅新增显式路径与策略注入，不复制另一套
搜索或审核实现，不全局替换变量。QP≤60、迭代≤6000、NNLS≤60；生产≤122、独立
≤122，合计≤244响应/73200内步/76130查询、2model。拒绝和initial/repeat全部计费。
0新stencil拟合；原accepted/rejected causal secant更新如实记录。
0物理/GPU/WAM/fresh/最终留出/实际参考消费，候选不下发。

QA/source/map/driver绑定，空CUDA/LD、数值库线程1。先独立重建生产ledger/GN/
KKT/scaling/recovery；每响应完整native重算，原1e-10/torque1e-8布尔审核及全部
响应/metric逐值exact，使用native metric再次独立审核全部决策/反馈/状态更新。
历史anchor/repeat exact，来源/源码前后SHA。preflight/init300秒、执行/独立1200秒，
owner1800秒TERM/10秒KILL，实际wait/reap，禁止自动重试、补选或放宽门。

比较已关闭匹配baseline58b65f7f及reserve5777638f：同起点/模型/root域/预算、
控制数学保持；它们是明确标记的既有实际数据，不伪称本轮重跑。软件回放核对旧
baseline完整答案不变，另报告本轮全部新费用及旧模型/对照费用，不隐藏计费。

报告完整loss/caps/prefix/witness、stage/接纳/restore、reserve、停止与费用；
若仅保持reserve却停滞或完整跟踪不改善，必须如实记录。物理与最终留出未执行
不能由本轮局部响应证据代替，也不因单组优于旧组就提升为默认控制器。
