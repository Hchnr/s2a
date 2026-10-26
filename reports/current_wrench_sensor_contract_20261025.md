# 102 当前力/力矩反馈合同：完整结果

实际固定模型定义 gripper0_force_ee / gripper0_torque_ee，传感器site为gripper0_ft_frame，位于gripper0_right_gripper子体，其父体为robot0_right_hand。实际运行传感器type为4/5、地址0/3、各维度3；robot.ee_force/ee_torque与直接sensordata逐元素一致。

MuJoCo 2.3.7定义读数在传感器局部坐标系，表示子体向父体的相互作用力/力矩，包含接触和其他动力学作用，不能称为纯外部接触力。[官方传感器定义](https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html#sensor-force)

转换使用Fworld=Rws Flocal，Tworld_at_control_site=Rws Tlocal+(p_sensor−p_control_site)×Fworld，保持符号，仅转换方向与moment参考点。固定机器人接口按N/Nm记录，未做重力/载荷/惯性补偿。保留原local6、sensor世界pose、实际control SITE位置、名称、type、地址及time。此为仿真传感器合同，不代表真机F/T标定。

六条原baseline按原初态/seed42/固定控制器/命令重放477显式控制周期；483个endpoint均在原endpoint刷新后、下一命令前读取。读函数本身无forward/step，完整state/ctrl/sensordata读取前后逐元素不变。全部prefix state/SITE pose/twist与086匹配；独立逐端点核对buffer/getter合同、SO3矩阵、旋转和moment平移、时间与有限值通过。

| group | endpoint数 | anchor局部force范数 N | anchor局部torque范数 Nm |
| --- | --- | --- | --- |
| 0 | 35 | 5.3811 | .0460 |
| 1 | 80 | 5.5508 | .6858 |
| 2 | 78 | 5.1895 | .6228 |
| 3 | 159 | 5.4025 | .7079 |
| 4 | 44 | 88.3047 | 8.1388 |
| 5 | 87 | 5.5311 | .0938 |

表中anchor是当前周期起点（不是新命令执行后的末端）。group4读数较大，与先前已确认触桌的场景相对应，但不据此选择接触阈值/证明唯一原因，也不能把这6个点当作泛化分类结果。

3项变换/杠杆臂/非法输入检查及既有观察器测试通过，6真实CLI dry-run通过；12个采集/检查阶段及owner实际wait0/reap、冻结来源不变。完整证据 outputs/current_wrench_sensor_20261025/closure_verified.json，各group下wrench_endpoints.json含实际读数。源入口src/s2a/current_wrench.py、scripts/verify_current_wrench_sensor.py，994个源码文件静态检查通过。

新增477显式控制周期/11925积分子步，6次初始化内部步骤另列；世界模型/fresh/fit均0。没有改变命令或模型，没有新控制/任务收益。后续若使用wrench条件，需先将同样读取合同覆盖原30轨迹，保持24fit/6holdout并核验原各anchor的共同因果前缀，再登记唯一拟合。不能从不含历史ctrl/warmstart的旧snapshot重算sensor力充当原测量，不能读取probe未来wrench作当前条件。原18/18正确参考、WAM开发及最终24新初态验收保持，总目标未完成。
