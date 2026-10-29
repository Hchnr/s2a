# 088 条件化、平移action仿射的arm前向模型

087数据资格完整通过：24fit/4865行、6内部holdout/1315行，实际SITE状态、解析/数值速度、实际API目标与native尾段latch均核验。086六anchor成对响应资格通过。085预测改善但闭环退化，因此本阶段同时验证预测和局部命令响应，尚不部署新逆控制。

只fit087原24轨迹，4865行，每轨迹总权重1/24、其内各行均分；6holdout及086探针均不参与fit、均值/尺度或参数选择。固定FP64 ridge=0.001，只有全局常数项不惩罚，不扫描参数或特征。seed42，所有来源/行/权重/参数指纹保存。

71维原始条件：current q7、sin(q)7、cos(q)7、qd7、site position3和SO3前两列6、site twist6、finger q/qd各2、known motor before/本周期末目标各2、previous actual native7 command、previous controller goal相对current SITE的position/世界SO3误差各3、当前已知applied rotation goal的世界rotvec3及sin(angle)*axis3、previous-command availability1。末motor只由current known cache和本周期计划夹爪命令按固定25子步/.01/限幅计算；不直接用未来实测motor。当前rotation goal及夹爪命令是forward条件中的当前action，不是世界模型desired state；后续部署必须由真实规划器提供。禁止使用未来观测、task/demo/time/phase/object truth作为条件。

原始条件和current applied translation goal delta3的中心/尺度只按fit权重计算，std<1e-8设1。令phi=[1,standardized_condition]为72维，a为标准化translation delta，288维设计矩阵=[phi,a_x*phi,a_y*phi,a_z*phi]。保持对translation action仿射，不在平移命令中引入平方项。参数288×12，不把小模型形式当作性能保证。

输出12维=[实际下一position差3, world SO3增量3, dt*下一site twist6]，dt=.05。学习相对原名义PD exp(A*dt)预测的残差；nominal仍用实际applied controller goal delta6及当前site twist，明确它是局部模型而非真实动力学。对已知rotation goal条件可非线性，但本阶段不求逆或改变rotation策略。

执行次序：先实现并检查特征因果性、motor规则、平移action仿射性和零残差名义基线，再做唯一fit；独立从原087 cache重建全部特征/标签/权重并用另一最小二乘分解复核，固定系数绝对/相对容差1e-8和加权正规方程梯度绝对1e-9。原失败和所有比较保留，不自动重fit。

离线性能门槛：6条holdout分别报告position增量、SO3增量、下一线速度、下一角速度四类RMSE，全部24项不劣名义PD才通过该项；同时报告保持和常速度基线，不预设胜者。086六组以相同实测初态、known action条件，在baseline及±实际translation命令下预测末位置/旋转/twist；旋转先Exp预测增量再取world Log差，不能直接相减Euler/rotvec替代组差。对position、线速度、SO3耦合、角速度四类3×3双侧响应矩阵，与真实差分比较；全部24个组/通道Frobenius误差不劣名义模型才通过局部响应项。086只作独立off-action方向验证，不能混入fit。

数值、预测、局部响应三项全通过也只授予离线forward资格；新增bounded inverse和原18/18正确参考回归须另登记，再依次WAM开发与最终24条新初态。任何一项失败不得据局部优秀项进入控制或挑另一个ridge。本阶段0物理/世界模型/fresh，唯一fit次数显式记录。当前仅登记，尚未fit。

实现补充：state_conditioned_arm_forward.py已实现71维当前条件、288维平移action仿射设计、原名义PD与12维预测。4项CPU检查通过：未来label/task/phase/time不进入条件、平移action仿射性、真实25子步motor规则、零残差回到独立闭式临界阻尼基线；Ruff通过。验证中的SO3误差必须比较Exp(predicted increment)与实际下一旋转的群差，而非直接相减rotvec；保持基线为零motion/零下一twist，常速度基线为当前twist持续一个dt。尚未fit、独立数值或留出/梯度验证，模型资格未取得。

执行实现补充：fit/verify/validate/run/close_state_conditioned_arm_forward.py与独立arm_forward_independent.py已实现。生产端加权正规方程Cholesky，独立端逐行重建condition/target/weight并以增广SVD复算；验证同时比较独立/生产预测，再评估6holdout×4门槛及6anchor×4成对响应门槛。性能失败仍正常完成并保留False，不自动重fit；参数、cache与fit_attempt独占写入。真实首轨迹164行的独立条件/设计/闭式名义响应对照通过，尚未开始fit。接下来全部准入通过后，唯一CPU owner依次执行三阶段和实际wait/reap审查，不能以数值合格替代性能。
