# Franka Panda 跨数据集解析验证

没有执行真机，也没有在独立模拟器执行候选命令。本报告评估只读日志的解析几何与已保存 action 字段。

代码 `f3f7214577546b211a33f48eef6ef1d513d36a29`；dirty=`False`；seed=42。抽样名单在数值评估前冻结，不训练或调参。

| Cohort | 数据证据 | 执行/选择 episode | 几何通过 | 转移数 | 单步最大 m/rad | 累计最大 m/rad | 原始观测最大 m/rad |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| robomind_robotiq | real_robot_log | 20/20 | 20 | 2428 | 6.01e-08 / 5.7e-08 | 1.19e-07 / 4.46e-08 | 4.15e-08 / 4.43e-08 |
| m1_robotiq | simulation_log | 20/20 | 20 | 7614 | 6.34e-08 / 7.05e-08 | 3.06e-07 / 6.52e-08 | 3.37e-08 / 4.15e-08 |
| robomind_panda_tool | real_robot_log_different_gripper | 10/10 | 10 | 1823 | 7.19e-08 / 5.43e-08 | 1.21e-07 / 3.24e-07 | 4.17e-08 / 2.98e-08 |
| a1_robotiq_sim | simulation_log | 10/10 | 10 | 7456 | 5.98e-08 / 7.81e-08 | 4.63e-08 / 5.06e-08 | 3.43e-08 / 4.26e-08 |

## 独立保存的动作字段与运动学代理

| Cohort | 对比对象 | 位置 RMSE m | 姿态 RMSE rad | puppet FK 观测位置/姿态 RMSE |
| --- | --- | ---: | ---: | --- |
| robomind_robotiq | recorded master joint target 的名义 FK | 0.0307754 | 0.0605377 | 2.22e-08 m / 9.93e-09 rad |
| m1_robotiq | 保存的 Cartesian action 字段 | 0.00287213 | 0.00443367 | 不适用 |
| robomind_panda_tool | recorded master joint target 的名义 FK | 0.0215378 | 0.0450447 | 2.11e-08 m / 2.4e-08 rad |
| a1_robotiq_sim | 保存的 Cartesian action 字段 | 1.46063e-08 | 7.18403e-09 | 不适用 |

容差为 1e-5 m / 1e-4 rad，同时检查候选 native pose 与下一原始观测；完整逐 episode 结果、masks、媒体时间及 action 字段差异见 JSON/NPZ。

## 这些结果能说明什么

解析式为 T_BC_next=T_BC_now@Delta_T_C；候选原生位姿为 T_BC_next@inverse(H_native_tcp)。每个数据集单独固定 H，未使用 DROID 的 .075/.15 缩放。M1 另有已记录的 effector→link8 z+45°，A1 原生四元数为 wxyz。

几何通过验证了发布表示、原始观测与坐标转换的一致性。目标来自日志未来观测，且发布状态本身源自这些原始观测，因此不能把低误差解释为控制器逆动力学正确、机器人执行成功或跨机器人控制泛化。

## 未执行项及原因

- 所有 cohort：没有真实硬件执行；序列号、历史控制配置及物理反馈时间未确认。
- 不以媒体 fps 代替物理 dt，未跨数据集转移 DROID PID/B/C/归一化 velocity 接口。
- M1 的夹爪缺少标定反馈，保留无效 mask，不生成夹爪命令。
- RoboMIND-v1 的夹爪 target 为下一观测，未当作独立绝对命令；原生 master joint 只通过名义 Panda URDF 做 FK 诊断，不冒充已确认的 Cartesian action；另报 puppet FK 与原始 EEF 一致性。
- A1 对应分支为仿真日志；与真实机器人日志分开，未据此声称真机或独立仿真通过。
- RoboMIND2.0 涉及 Panda/FR3 和名义 FK，不作为已确认同型号比较。

## 依据

- 型号来源：[RoboMIND 官方项目](https://x-humanoid-robomind.github.io/)、[A1 官方项目](https://internrobotics.github.io/interndata-a1.github.io/)。M1/A1 是 Panda/Franka 仿真分支；真实机器人日志来自 RoboMIND-v1。
- M1 官方数据卡明确为仿真：[InternData-M1](https://huggingface.co/datasets/InternRobotics/InternData-M1/blob/main/README.md)。M1/A1 均未计入真实机器人日志。
- 原始 native pose 约定：外部只读 eef_state_sources.py:89、:119、interndata_mobile_pose_probe.py:11，SHA256 存入 manifest；S2A 不导入发布器。
- 运行：LD_LIBRARY_PATH='' .venv/bin/python scripts/run_cross_dataset.py --seed 42。
