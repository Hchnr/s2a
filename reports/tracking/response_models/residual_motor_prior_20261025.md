# 073 固定保持先验的残差对照：离线通过，物理未执行

按073预登记仅进行一次CPU拟合。复用072独立重建过的6180行、原24fit/6内部留出、40维特征、轨迹等权及ridge0.001；唯一变化是学习next_motor-current_motor，再加回当前motor。没有扫描参数或按案例选择模型。

六条内部留出预测motor和原生可达motor RMSE均不劣保持当前motor基线，offline_performance_eligible=True。wine/demo39的预测/可达误差为0.06167/0.05988，保持基线0.07157；072直接目标为0.10232/0.08002。其余五条可达误差0.01705–0.02712。不是所有误差均优于072，例如bowl/demo40可达误差由0.01917增至0.02712；本次通过的是原保持基线门槛。

独立验证器不调用producer特征和量化函数，重建fit标准化与40列，使用增广最小二乘核对正规方程解；系数最大差5.66e-14。逐轨迹预测/量化/保持误差复算通过，来源与参数SHA读取前后不变。先前072的原始数据逐行核验仍为依赖证据，本次未重复宣称新原始数据审计。Ruff检查及格式检查通过（验证器首次import排序失败，修复后再执行成功）。

产物位于outputs/residual_motor_prior_20261025：parameters.json、fit_result.json、independent_verified.json。新fit1；模型调用、物理步、fresh均0，physical_qualification=False。六条内部留出已用于072诊断，不构成新泛化证据。下一步需另登记统一残差模型的正确参考物理回归，覆盖此前wine、cream退化及原控制资格；不得直接进入WAM或fresh。原开发5/6且各任务1/2、fresh20/24且各任务4/8门槛不变，总目标未完成。
