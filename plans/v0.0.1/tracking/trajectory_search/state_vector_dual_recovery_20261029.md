# pose乘子恢复活动子集的有限对照

本项在新实现、拟合、QP和模型响应前登记。前项整体目标缩放10QP均status0，
但原互补门拒绝7个，仅产生1个pose数值响应；5响应生产及5响应完整独立复算
已关闭，完整witness0。纯保存乘子算术诊断没有新拟合，发现7个最大互补项均在
x方向rate行，原mu给约1.77e-6..4.43e-6的正slack非零权重。直接裁剪破坏
stationarity，没有证明另一合格dual存在，更不能据此认定primal最优。

本次仅改变pose的dual recovery：原canonical slack由原A/b/S及SLSQP返回的原空间点(d,t)计算，
原完整active_indices仍按`slack <= 10.0*certificate_tolerance`记录；新增
recovery_indices为其中满足`slack <= .01*certificate_tolerance`的子集。
tol仍原1e-6，因此新阈值约1e-8；直接保存两种表达式的实际值，不以显示舍入
数替换原表达式。只对canonical A[recovery_indices]的转置（NNLS设计矩阵的列）和返回点处的
**原未缩放gradient=[g+B@d,1]**做一次NNLS，
把结果放回完整mu，其余行mu=0。不裁剪旧mu、不搜索阈值、不weighted-fit、不LP、
不fallback或第二次拟合。maxiter仍为3*max(原完整active数,1)，不缩改原预算语义。
recovery子集为空时mu=0、记录原gradient残差，不制造成功拟合调用。

restore的NNLS集合和算法保持原样；整次QP仍为前项的f/C、grad/C，原C规则、
原S行缩放、初点、100迭代/ftol及SLSQP成功要求保持。新增明确恢复policy和索引，
mu/C仅标注为该次原单位NNLS证书的代数对应，不冒称SLSQP回传dual。

所有证书始终由原B/g/A/b/S、原gradient和新完整mu重算，原primal、stationarity、
complementarity各1e-6与非负乘子门均保持，不以较小活动集合替代完整约束核验。
NNLS失败或原KKT任一失败仍阻止模型响应；原7个失败、完整旧mu和数据保持不变。
更紧子集不保证恢复成功，也可能失去stationarity；不预设胜者。

独立审计应从原stencil和本run的三模型更新/原QP点重建两个索引集合，核非子集
mu exact0、原NNLS预算/原单位残差、C及dual映射，复用原完整状态机/KKT/计费。
单元测试须包含“正slack多余列能使NNLS的stationarity拟合却损坏互补”的真实
需要，以及紧子集不足、空集合、NNLS失败、原primal失败、solver失败仍拒绝，
restore原字段exact、单次QP/单次NNLS与完整执行门不变。不得只凭更少失败标志
或低动作RMSE认定新动作有效。

独占新namespace，从原3557和72条signed差分重建，原actual368/因果历史与12参考
保持；不预载旧final或追加旧50阶段/117响应。numeric config逐字节同前项，
seed42、60stage/122完整响应，所有原接受/反馈/半径/停止/repeat保持。至多一次
有预算生产；先专项CPU QA、Ruff/format及两dry，通过后及时提交，再运行生产
和全部响应独立复算。所有fit/QP/模型与查询显式计费；失败保留，不自动重试。

报告同时列原证书、NNLS完整/恢复集合、预算、F/总loss、12点q/位置/旋转、安全、
prefix/full witness及总成本。数值witness仍需独立实际前向控制，不能授予在线
多任务或留出资格。原18正确参考/12WAM开发/24最终留出门槛保持，实际仍cursor8/
task0/grasp0。全部在当前仓库维护，旧文件冻结，不访问旧清理根或用户未跟踪文件。
显式空CUDA/LD、单线程CPU；GPU继续禁用，等待用户明确恢复。总目标保持。
