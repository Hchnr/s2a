# 从线性可行点启动长预览搜索（CPU）

同一实际368状态长预览12/16未找到模型witness；16点连已检查线性可行尝试都未保存，而LP已找到可行点。本候选检验初值不满足新增长窗口线性约束导致的优化困难，不把数值失败写成物理不可行。

新模块`feasible_start_trajectory_joint_plan.py`保留完整原约束、端点精度、160响应/60迭代及30迭代可行性阶段。先独立检查原warm start的box/rate、全内环、历史收紧、制动和terminal线性约束；若全部满足则不改变初值，否则选择已求得并逐项重查的LP witness，再进入非线性搜索。LP初值同样计入160响应，不因为线性可行而接收pose；原完整缓存witness检查不变。若LP恰处于未支持的全零rotation latch，明确记录并沿用原搜索起点，不伪装为线性可行启动、不放宽约束。

新模块与原长窗口共6测试通过（4.56s）：已验证warm计划结果保持一致，关节违规但box/rate合格的warm plan确实改用重查过的LP点，预算不超限；另覆盖8/12/16点及17点拒绝。原已执行冻结模块不改。

固定数值比较同一实际368状态的8/12/16预览，原warm plan、参考、实测动力学与历史均与上一阶段逐项一致，只有非线性起点选择改变。3次共最多480响应，不追加初值、不改物理状态、不新增模型调用/fresh；新outputs目录保留历史。只有完整模型witness才考虑固定接口原生前向验证。GPU继续禁用，原全部控制/任务门槛保持。

入口`execute_tracking_v5_feasible_start_trajectory_numeric.py --launch`，当前结果待实际执行关闭。

首版实际结果未完整关闭：8点witness（2响应）、12点无witness（90响应，原warm线性不合格，改用复核LP启动）；16点LP success点未通过原1e-10线性接收复核，probe/owner实际wait1/reap1，保留partial结果。独立重新rollout该LP命令得到全内环最小slack −1.38364821822e-8rad，box合格、rate超限仅1.11e-16、制动slack−1.33e-15、terminal超限4.13e-15；明确拒绝，不放宽1e-10门槛。

v2独立新模块/入口将HiGHS primal/dual求解容差均收紧至1e-10，记录LP返回残差并维持原box/rate/关节接收条件；未通过残差复核时返回明确linear_witness_not_certified且不进入非线性搜索，不把success标记当作资格。其余参考、历史、初值策略与预算不变，新outputs不覆盖。v2与原长窗口6测试通过（实际结果见运行输出），下一次固定8/12/16数值比较仍不增加物理/模型/fresh。

v2三项已actual wait0/reap0完整关闭：8点witness（2响应），12/16从认证LP起点仍无witness（122/103响应），合计227，verified SHA 063ca5370c69c77f1ac53f548219cc63a69192b502b7d5655626e665f07aa575。[结果](../../reports/feasible_start_trajectory_20261028.md)。新模块及原长窗口6测试passed in4.83s，原门槛未放宽。进一步独立XML检查发现被接收8点模型解也有joint/TCP几何不一致，后续优先修正预测几何而非继续放大窗口。
