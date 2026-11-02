# 终点局部下界对算法搜索域的依赖

2026-11-02。前序固定方向在完整数值响应中安全失败，不能执行。原root±.1
限制是算法域，不能把其下界当作机器人全输入域的不可达证明。

固定已审核small-progress最终anchor23、terminal rotation group23和保存lp_48的
A/c、anchor误差法向。只在诊断中把72条command_lower/upper offset替换为原固定
控制接口的完整input_min/max；保留全部4369线性安全行、.25 rate及原模型。
两组floor分别为原buffer和reserve .9，原terminal .1权重保持。原root结果作为
历史对照不重跑。生产算法、原域、实际安全和caps均不改；新LP点不下发。

预算2新LP，每次10秒/maxiter10000，原单位primal/dual/gap容差1e-8，使用有限
全输入box对残余stationarity做保守修正。0新响应/model/查询/内步/QP/NNLS/
拟合/物理/GPU/WAM/fresh/最终留出/实际参考消费。独占
outputs/input_domain_support_20261102，owner300秒TERM/10秒KILL，实际wait/reap，
不重试；输入及源码前后SHA保持，先32并发专项验证并绑定来源。

若完整input域支持≤1，只能说明旧局部排除结论依赖算法域，不证明共同caps
或非线性安全可行；若仍>1，仍仅该局部模型的排除证据。两种结果都必须保留
前序大步非线性安全失败。后续算法要同时处理域探索和可行性恢复，不能只改域
大小或直接采用线性最优点。总目标与原完整验收保持，GPU继续禁用。
