# 两项最小下降改为1e-5的有限新对照

2026-11-02，seed42，按80c3792设计与固定未调用方向实际证据登记。GPU禁用。
同一anchor85、f4bc1804模型、原root域/pose ceiling、aligned reserve floor及仅Js反馈。
唯一search config变化为minimum_predicted_reduction和minimum_actual_reduction，
从旧1e-4准确改为预先固定1e-5，显式progress_policy，不改旧默认或旧结果。

这两项是算法进展/停止参数，不是放宽原实际joint/buffer、.9 reserve、ratio、
全部24个.01m/.05rad跟踪caps、终态/稳定20/24且每task4/8资格。原已拒绝方向不
追认为通过，原预算不恢复。特别是固定方向中前两个仍实际reserve失败，不能因
进展门改变而接纳。

新独立预算仍60stage/122生产+122独立，共≤244响应/73200内步/76130查询、
2model，≤60QP/6000迭代/60NNLS；拒绝与initial/repeat全部计费。原accepted及
安全only反馈分别记录，0新stencil拟合、0物理/GPU/WAM/fresh/最终留出/实际参考
消费，候选不下发。源入口必须精确核查两字段差异白名单及其余字典全等。

复用共享有限协议，显式路径和progress参数，不全局替换。QA/source/map/driver
绑定、源码/输入SHA前后不变，CPU空CUDA/LD、数值库线程1。阶段preflight/init
300秒、执行/独立1200秒，owner1800秒TERM/10秒KILL并实际wait/reap，无重试。
原1e-10/torque1e-8布尔审核、所有完整响应和decision metric逐值exact，历史/
initial repeat exact；producer ledger与native metric替换后的完整独立ledger均审核。

先回放旧safety_feedback12QP/20响应完整答案核对默认不变；对照为已关闭93deb627
数据，不伪称重跑。报告完整caps/prefix/witness、误差、reserve、接纳/拒绝/更新、
停止及费用；不以更多接纳或小幅loss下降宣称完成。若仍不改善完整跟踪，应转向
完整瓶颈/控制资格审查，不继续仅按标量loss新增进展参数变体。最终留出关闭。
