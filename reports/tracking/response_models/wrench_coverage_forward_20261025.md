# 107 新增干预覆盖forward准备

本阶段尚未拟合。106全147条/294阶段及owner正常wait/reap和来源资格通过后，才允许生成绑定配置。现有104资格失败结论保持；新增训练样本不能直接证明闭环有效。

新增生产端与独立组合分别保留086/097全部72条方程并追加106的63条，合计135条干预/5000总方程。dataset及equation_group分离避免身份碰撞。原89维当前因果条件、872列设计、原24fit统计、Fourier基、ridge=.001和两类各.5总权重保持。生产端及独立复核都精确比较104实际参数保存的state/translation均值、尺度和固定基，禁止以重新估计近似相同代替固定参数。

已准备资格绑定、全数据来源/方程对比、唯一fit、独立增广SVD、原holdout/局部响应验证、持久监督及完整关闭入口。新增脚本ruff检查与格式检查通过，完整模块导入通过。这些仅是入口准入，尚未运行5000方程对比、参数求解或性能验证；未声明任何候选资格或任务收益。

入口：scripts/prepare_wrench_coverage_forward.py；scripts/run_wrench_coverage_forward.py。登记计划：plans/v0.0.1/107_wrench_coverage_forward_20261025.md。输出预定outputs/wrench_coverage_forward_20261025，当前无候选拟合产物。新增fit/物理/世界模型/fresh均0。

入口审查修正：launch现在对106资格内容逐项检查（complete/data_qualified、147案例、13209周期及owner正常wait），逐一校验并冻结其来源及summary；104原参数引用与配置绑定来源一并冻结。独立数值记录补充所有新增组合模块指纹。修正后ruff与格式检查通过，尚未执行fit。

## 107 完整结果：资格失败

106全量数据资格通过后，5000方程（4865自然+135干预）唯一fit、独立增广SVD和原性能验证均执行完成。15项基础测试、全部自然特征/135干预方程独立一致性准入通过。独立系数最大差9.022134168229501e-15，最大正则梯度2.79128788983569e-16。完整关闭审查确认三个stage及owner实际wait0/reap/来源不变。

holdout通过23/24通道，局部通过22/24，offline_forward_eligible=false，control_qualification=false。未进入逆控制、WAM或最终fresh。相比104的21/24与22/24，新增覆盖修复两条留出线速度退化，但原逐通道门槛仍失败。

剩余失败：holdout index28线速度RMSE=0.03777410356275418，nominal=0.031242752751587693；group4局部SO(3)增量矩阵误差=0.010768153112973702，nominal=0.008201913715536734；角速度矩阵误差=0.32403079587547007，nominal=0.269341306907388。部分误差改善不能替代全部门槛。当前wrench以线性条件特征进入模型，未纳入原Fourier非线性条件基；覆盖扩展本身不足以解决接触姿态响应，下一步需区分特征表达和局部响应辨识误差，禁止直接绕过资格部署。

权威产物：outputs/wrench_coverage_forward_20261025/fit_result.json、independent_fit_verified.json、validation_result.json和closure_verified.json。新增全局fit1，新增物理/世界模型/fresh均0。总目标未完成。
