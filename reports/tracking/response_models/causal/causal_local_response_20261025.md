# 096 因果局部响应：完整离线结果

核心估计器已实现：冻结090全局先验，用最近20个已完成控制周期的standardized translation goal和实际响应估计4×12仿射修正，平均平方损失+固定单位ridge，所有系数含bias均向零收缩。位置/速度加法修正，旋转使用world Log(Ractual Rprior^T)创新及Exp(correction)Exp(prior)复合。当前predict不更新历史，episode重新实例化隔离；无全局参数重拟合，无参数扫描。

4项核心检查通过：预测只读与episode隔离、20周期窗口和独立增广SVD一致、非交换SO3创新/复合、未来标签扰动不改变当前/更早预顺序预测。真实6holdout共1315行的每个历史标签均核对为下一当前可用SITE pose/twist，周期t只更新至t-1，再预测t。1309次因果在线更新及1309次独立SVD复算一致至1e-10，未称为“完全零拟合”。

| 6轨迹等权宏平均误差 | 因果局部估计 | 冻结090 | 名义 |
| --- | --- | --- | --- |
| position m | .000460216 | .000478945 | .001116753 |
| rotation rad | .000792711 | .000824822 | .002611390 |
| linear velocity m/s | .015011283 | .015382920 | .035992640 |
| angular velocity rad/s | .023829575 | .024273836 | .079883800 |

6holdout×4相对名义的原预测门槛通过24/24；宏平均优于090不代表每条均改善，也不证明局部action响应或控制收益。部分结果及来源SHA在`outputs/causal_local_response_20261025/holdout_prequential.json`。核心`s2a/causal_local_response.py`、实际检查`scripts/check_causal_local_prequential.py`均静态检查通过，全部945个源码文件Ruff检查通过。

固定六anchor响应验证现已完成，仍21/24：group2位置误差0.00178993（名义0.00123477）、group4旋转增量0.01311296（名义0.00820191）、角速度0.41105636（名义0.26934131）失败。训练12/12、未训练9/12。每个anchor从已完成native前缀当前端点计算观测，七个probe共享冻结adapter，未用probe endpoint适应；独立增广SVD及预测对照通过。首版汇总变量与矩阵dict重名导致AttributeError，改名后完成，参数和门槛未变。

新增物理积分/世界模型/fresh均0，无全局参数重拟合。090失败证据及原18/18正确参考、WAM开发、最终24条留出新初态稳定门槛保持，总目标未完成。

持久化完整复核使用独立owned/results目录，不覆盖首轮部分数据；holdout24/24、局部21/24复现。两个stage和execution owner均实际wait0/reap、无自动重试、来源冻结不变，关闭证据`outputs/causal_local_response_20261025/owned/closure_verified.json`。本次执行1780次因果更新、1315次独立数值解；加首轮检查，共3560次更新、2630次独立数值解，不计为新全局候选拟合。离线forward资格失败，不进入逆控制。

改善历史轨迹上的预测未修复局部action导数。下一步需要审查训练干预的状态/动作覆盖，判断如何补充独立训练响应证据；继续保持现有未训练anchor检查，不用其标签反选衰减或正则化。
