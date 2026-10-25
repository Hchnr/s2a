# 079 当前速度先验的前向响应残差：唯一fit完成，验证未执行

在查看新干预模型误差前固定plans/v0.0.1/079_velocity_residual_response_validation_20261025.md。原24fit总体权重0.5；新open/close六分支case01/03/05/07/09/11仅anchor后24周期，共144周期，总权重0.5。每组内逐轨迹/分支等权，再周期等权；baseline与共享前缀不重复加权。原6内部留出和新retain02/06/10均未进入拟合、标准化或特征选择，新验证分支共享训练初态不宣称新初态泛化。

输入当前实测q、真实已发送motor cache、从已知candidate命令推算的next_motor、当前瞬时速度V（0.05*(finger_qvel1-finger_qvel2)/0.079）。15维标准化二阶特征，常速度q_next=q+V、V_next=V先验；学习[(q_next-q)-V,V_next-V]两列残差。FP64 ridge0.001、bias不罚，单一fit，无超参数扫描。在线纯模型拒绝非有限rollout，不临时clip V；q按既定[0,1]观测范围clip。

响应模型9项实际检查通过：常速度先验、慢速度不从motor gap臆测、两输出/参数副本、native motor子步饱和、无未来输入、递推自身状态、非有限拒绝；Ruff通过。fit入口在读取训练数据前强制078 closure_qualified与owner正常wait，实际唯一fit完成5009行（原native4865+新144），保存fit_cache、行组/行范围/权重/参数及来源SHA，读取前后不变。所有产物outputs/velocity_residual_response_20261025，validation_executed=False、physical_qualification=False，0新物理/模型/fresh。

尚需独立重建全部训练行和速度索引、增广最小二乘核对，以及冻结参数后的原6+新3retain一步/8步响应验证；当前没有新误差或任务收益结论，不把拟合完成当作性能通过。原074/075失败和最终WAM/fresh门槛不变。

## 独立训练行与求解核验通过

verify_velocity_residual_response_fit.py不导入producer模型/特征/子步函数，从原XML和完整sim state独立重建当前finger qpos/qvel索引及V；重建24native+6干预全部5009行、两列残差目标、组号、源周期范围与权重，并核对0.5/0.5混合比例。增广最小二乘与原正规方程系数最大差6.46e-14，梯度最大1.31e-16；输入/fit/参数/cache SHA前后不变。independent_fit_verified.json numerically_verified=True。

## 冻结参数后的9条响应验证通过

validate_velocity_residual_response.py不拟合参数，评价原6内部留出及隔离新retain02/06/10。一周期与8周期rollout均使用当前q/V/m和已知命令；8步递推不注入未来实测q/V，尾部只算实际存在端点。保持q、常速度先验和冻结075均独立记录。

|响应轨迹|079一步q RMSE|保持q一步|079八步q RMSE|保持q八步|
|---|---:|---:|---:|---:|
|bowl/demo40|0.00705|0.02951|0.03621|0.12216|
|bowl/demo48|0.00532|0.03234|0.04213|0.14331|
|wine/demo39|0.01511|0.02294|0.07365|0.09796|
|wine/demo46|0.00530|0.01632|0.01952|0.06261|
|cream/demo40|0.00678|0.03509|0.04444|0.15395|
|cream/demo49|0.00652|0.02128|0.01680|0.08210|
|新retain case02|0.00510|0.06124|0.02563|0.30200|
|新retain case06|0.00211|0.04371|0.00693|0.20616|
|新retain case10|0.00392|0.03533|0.01372|0.15906|

9条均满足原两个q门槛，非有限rollout0，offline_performance_eligible=True。V误差单列保存，候选8步V RMSE约0.00576–0.01732；未clip V。verify_velocity_response_validation.py不导入producer模型/特征/验证函数，再从原始state的独立速度索引起步，以scalar/math.fsum复算全部9条、四方法的一步/8步q、V误差及有效计数，与保存结果逐项1e-10内一致，来源前后不变。independent_validation_verified.json verified_cases=9、numerically_verified=True、offline_performance_eligible=True。Ruff和实际执行均通过；0新fit/物理/模型/fresh。

079并未全面胜过其他模型：例如wine/demo39八步0.07365略差于常速度0.07139及冻结075的0.07317，bowl两条及cream/demo40八步也差于075。通过的是预登记保持基线门槛，不是所有基线全面胜出。新数据、当前速度状态和常速度先验同时变化，没有单因素消融，不把收益唯一归因于某个因素。

新retain分支共享原训练初态与前缀，只是未拟合命令响应，不是新初态泛化；原6内部留出仍带诊断历史。本结果仅允许另行登记局部模型逆控制实验，physical_qualification=False，尚无新action控制或任务成功。下一步080固定有限原生命令规划、正确参考物理回归，原WAM开发和fresh门槛不变。
