# 096 因果局部响应适应对照

095证实接触改变瞬时响应，但完整环境真值不能在线使用。本候选以冻结090预测为先验，仅用最近20个已完成周期（20Hz约1秒）的当前可测SITE运动/速度、已发translation goal及当时先验预测，估计4×12局部仿射修正。特征=[1,原090 fit-only标准化translation goal3]，平均历史平方损失+单位ridge向零修正收缩；20、ridge1在运行前固定，无扫描。所有系数含bias均正则化。每episode独立清零；当前候选预测不更新历史；实际端点抵达后才允许更新；重规划不清实际物理历史。

位置/速度修正加法；旋转观测创新=world Log(R_actual_increment R_prior_increment^T)，预测旋转=Exp(correction) Exp(prior_increment)，不得减Euler或将一般rotvec差当作几何误差。离线预顺序评估时周期t只能使用已完成0..t-1的结果，不能将本行标签提前加入。保留原6holdout×4预测门槛及6anchor×4响应门槛，另与090比较；只测holdout不能授予完整资格。probe anchor共用原native前缀历史，不能把该probe终点用于适应或让±分支相互更新。

先实现核心和因果性、episode隔离、独立增广SVD、SO3修正测试，再做真实6holdout预顺序对照，随后接入固定六anchor响应与持久化完整审查。在线更新/小线性解次数单独记录，不声称“完全零拟合”；不重新拟合090全局先验。物理/世界模型/fresh均0。无控制资格；090原失败保留，原18/18及开发/最终24条门槛不变。

首轮预顺序与六anchor检查已完成：holdout24/24、局部21/24，仍未资格。anchor检查首版在结果汇总处将adapter变量与矩阵dict同名导致AttributeError，修复变量名后完成，未更改候选/窗口/ridge/门槛。持久化复核使用独立owned/results目录重新计算同样统计，不覆盖首轮数据；重复在线更新及独立数值解另计，不作为额外候选拟合。两个检查stage及owner均要求实际wait/reap和完整来源不变，随后关闭。失败性能正常退出，不自动重试或换参数。
