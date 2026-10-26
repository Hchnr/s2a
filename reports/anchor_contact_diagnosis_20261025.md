# 092 六anchor接触与命令力矩灵敏度：结果

已完成086六个baseline anchor的只读状态重建。每个实际XML和原始状态均核验SHA，mj_forward后的SITE位置/朝向与原日志一致至1e-9，J*qvel与原twist一致至1e-8，arm M通过正定检查。原086数据完整资格和owner退出证据已核对。不运行mj_step，也没有参数fit。

| group | 全部接触数 | 涉及robot/gripper的接触数 | 机器人接触对象 |
| --- | --- | --- | --- |
| 0 | 37 | 0 | 无 |
| 1 | 26 | 7 | 碗 |
| 2 | 37 | 0 | 无 |
| 3 | 41 | 8 | 酒瓶 |
| 4 | 22 | 2 | 桌面、碗 |
| 5 | 41 | 8 | 奶油奶酪 |

group4的两处几何接触分别为table–gripper0_leftfinger（dist=-0.000698869m）和akita_black_bowl_1_main–gripper0_right_gripper（dist=-0.000750424m）。六anchor仅此处出现机器人接触桌面。这个事实为091无接触冻结M/J模型在group4的旋转/角速度失败提供明确的遗漏机制，但不是唯一因果归因：group1/3/5也有物体接触，且其他非线性因素仍存在。

同时保存每关节normalized translation命令到力矩的7×3局部导数，以及实际XML中的actuator限幅与gear。六处最大绝对导数分别88.2933、69.6145、72.5136、66.2246、86.0972、79.1167 Nm/normalized command。XML七arm actuator ctrl范围前五个±80、后两个±12；不能直接比较导数的单位值与限幅来认定饱和，必须结合实际命令跨度、基线力矩、补偿及空域项。历史ctrl和控制器缓存尚未恢复，因此actual_saturation_verified=False。

接触力未宣称为实测：mj_forward在当前重建ctrl下的求解力并不是历史实际控制周期接触力，结果仅记录几何/约束行，不以其数值作因果判断。完整object/contact真值只用于离线诊断，不得作为在线S2A输入。

结果与来源SHA保存在`outputs/anchor_contact_diagnosis_20261025/result.json`。源实现为`scripts/diagnose_anchor_contacts.py`，静态检查通过，实际六anchor重建检查通过。新增物理积分/世界模型/fresh/fit均0；原正确参考、WAM开发和最终24条留出新初态门槛均未改变，尚无新控制资格。

源码路径追踪表明make_environment不传controller覆盖，LIBERO env_wrapper.py:47读取默认OSC_POSE配置；该环境osc_pose.json的uncouple_pos_ori为true。这是源码/配置路径证据，尚未替代实际实例uncoupling字段读取。下一步须在独立实例核实该字段，并获取与历史endpoint一致的控制器状态/实际力矩，随后评估仅用当前机器人状态与因果历史的约束响应估计；不直接将离线桌面接触真值接入控制。
