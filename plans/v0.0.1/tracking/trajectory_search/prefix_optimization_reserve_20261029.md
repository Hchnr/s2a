# 原H12前缀优化余量对照（CPU、执行前登记）

总目标、固定控制器、18条正确参考/12条WAM开发/24条最终留出门槛保持；GPU禁用。本轮检验显式优化余量能否改善原H12完整模型解搜索，不能用合成例、局部下降或模型解替代实际控制与留出任务成功。

## 已有依据与前置

前一callback回缩producer已实际wait/reap0关闭，结果SHA`81051b71f0822b56e52492f87ce5032623212453d228f4e5952514bef036ebbe`，owner SHA`a233f5c141bf52f1bc0cf26f8c6e0426cf6ff2d4f1e3bf5689c0c3ac9defa427`。1012响应/15迭代/15阶段，12个ray下降被接受，但prefix始终9、完整witness0；最后三个半径的27个ray全丢失原prefix，其中22个仅cursor4旋转严格超原.05rad。最后anchor距该边界约1.74e-9rad，最小alpha仍超限；不能据小量或不单调性认定仅是浮点误差，更不能放宽实际caps。

登记时完整独立checker仍运行（原tool50104，child3756445/start615217558）；其owner登记SHA`3bc501a4855135fdca55a7d4106489692dadf0c06be7eb9d8a2cbaa4826db5b4`。当前只允许新实现和合成测试，正式搜索必须等待1012响应/303600内步/315745查询全部独立核验、75份完整ray与13份anchor审计通过、严格标签无分歧、完整witness0及原owner实际wait/reap0无信号且来源未变。准入绑定已关闭结果的实际SHA、精确producer和登记；如检查失败，不启动本候选。

保留同一actual368 q/v、completed367命令与旋转latch、固定手指状态、chunk46原时序H12参考。共同warm仍为已核验回缩entry55，bytes SHA`16bdfbb266ef13d2588c8116e13511d93df936d72ae8a57d19df0b01ed3acee0`，不以本轮失败的最后anchor换warm。完整96资产快照、原模型/接口/目标/时间/增益/力矩和rate约束均不变。旧已执行源码、测试、计划及产物冻结。

## 唯一修改：优化约束保留固定余量

新增`solve_prefix_optimization_reserve_response`，以冻结callback求解器为基准。归一化平方误差仍为`q_p=||e_p||²/.01²`及`q_R=||e_R||²/.05²`。仅SLSQP阶段内原已达标prefix的约束从`1-q>=0`改为`1-1e-4-q>=0`；`PREFIX_OPTIMIZATION_MARGIN=1e-4`固定，不搜索或按结果调整。共同warm已达prefix的最小余量约8.46e-4，大于该值；准入检查此事实。

此余量仅作用于优化器的前缀约束。目标cursor的epigraph约束、目标函数及其梯度、全部安全约束/差分数学保持原样；prefix约束Jacobian不变，因为仅减去常数。没有给未达目标新增余量，也不改变真实状态误差定义。

普通ray接纳仍按原完整审计、原prefix严格caps、非线性minimum>=1e-8以及原target下降>1e-12或prefix增长判断；不要求接纳点达到优化余量。这样优化器在下一阶段仍有向内恢复余量的要求，不将接纳边界整体替换成更紧边界。完整witness仍按全部原H12 caps和完整审计判定，不要求额外余量；任何完成cache中的witness（含差分点）均保留。

其余控制流完全沿用callback回缩：原三半径、首个非零callback截断、九个固定alpha、普通anchor仅首个合格ray点、精确全局缓存、每阶段最多5次实际迭代、全局4000响应/60迭代/60阶段，以及全部原排序和预算中断行为。不加入二阶校正、更多alpha、额外响应或失败后的自动候选。

新增source AST核对需反向还原函数名、说明文档、唯一prefix常数修改和结果中的余量字段后，与冻结原函数完全一致；证明作用域明确为该solver差异，不替代数值复算或证明新main等同旧main。

## 验证、预算和执行

独立checker沿用冻结的完整数值重构、严格分类和callback账本审计，不导入producer solver判断资格。额外验证余量元数据和上述源码差异；每条唯一响应、initial/selected/closest、全部完整ray与接受anchor均重新核验。优化余量不改变任何实际caps或checker数值容差。

CPU合成测试验证仅prefix约束减去固定常数、Jacobian/目标/安全不变、原caps内但未达到优化余量的完整witness仍被接受、原ray接纳语义不变、实际SLSQP表现与预算，以及AST变造拒绝。保持旧测试，不按新困难例结果改生产参数或原验收；若新困难例失败，如实保留其已知可行和有限搜索结果。

统一native QA绑定旧28份来源及全部新source/test/plan，Python3.11.15/NumPy1.26.4/SciPy1.15.3，空CUDA/LD、单线程；完成Ruff/format、实际producer/checker dry、owner dry与来源核验后提交，再启动唯一候选。

新独占输出为`outputs/tracking_v5_prefix_optimization_reserve_20261029`及相应`_check_20261029`；owner为相应`_execution_20261029`。producer1200秒、checker5400秒；每响应300数值内步/312查询，另一个current query。所有计费、真实wait/reap及异常保留规则保持；无自动重试，不重跑旧producer。

比较完整witness、原prefix、cursor9位置/旋转误差、全部接纳历史及真实响应/迭代成本。局部改善不代表真实跟踪成功；没有完整独立H12解时不进入物理pilot。后续物理环境的LIBERO/OSMesa/配置及bootstrap资产路径还需单独完整迁移和核验，不能据现有96个数值模型资产快照声称已解决。
