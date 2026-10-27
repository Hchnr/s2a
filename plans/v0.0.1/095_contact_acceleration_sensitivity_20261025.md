# 095 接触求解下的瞬时加速度灵敏度

094已排除group4原七分支目标周期的限幅及robot接触body对切换。比较六baseline anchor处相同M/J下，无约束与实际接触求解的关节力矩→SITE加速度局部导数；不把瞬时导数等同0.05s端点响应或控制收益。

复用093目标周期首个实际arm ctrl与uncoupling实测结果。用086 anchor当前完整qpos/qvel及motor缓存，加本周期已知gripper命令一次更新重建首内环gripper ctrl；所有输入SHA固定。只做mj_forward，保持qpos/qvel/time不变，对7个arm广义力施加中心扰动±1e-4 Nm，另用±5e-5检查数值一致性，不择优挑步长。两者不稳定时显式标记不可靠，不调整到“通过”。warmstart统一清零，承认历史warmstart未保存；不把恢复求解力说成历史实测。

输出6×7 d(SITE加速度)/d(arm力矩)、无约束J M^-1、相对差和translation goal瞬时响应差；全状态/接触属于离线真值，不允许作在线S2A输入。该检查只定位模型遗漏，后续若建可用模型需要机器人可观测信息及完整一步/控制验证。物理积分/模型/fresh/fit均0。不修改现有模型、控制器和原验收门槛。结果新namespace outputs/contact_acceleration_sensitivity_20261025。
