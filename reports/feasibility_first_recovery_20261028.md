# 首周期可行性优先搜索的恢复检查（2026-10-28，CPU）

[候选计划](../plans/v0.0.1/feasibility_first_joint_plan_20261028.md)保持原控制接口、参考和全部约束，仅改变搜索顺序。在同一实际held endpoint378、同一真实q/qvel和自身10周期因果历史上，分别检查hold8和原resume9，不添加重启或物理动作。

| 固定目标 | 模型witness | 响应评估 | 迭代 | 首周期可行性阶段 |
| --- | --- | ---: | ---: | --- |
| hold8 | 有 | 83 | 55 | 1次迭代、超限平方目标0 |
| resume9 | 无 | 155 | 52 | 7次迭代正常收敛、超限平方目标0.12822907265 |

hold8最终完整预览优化正常收敛，首周期误差约.00339m/.03518rad，保留原约束。resume9随后完整预览搜索45次迭代退出Positive directional derivative for linesearch。其全部已检查线性可行尝试中，最小最大归一化超限为.50641696782，对应位置.00588297705m、旋转.07532084839rad；旋转超过原.05rad，不接收。这里的最好已检查尝试不是全局下界，不证明目标在物理或非线性模型中不可达，也不因第一阶段optimizer success而判定精度合格。

两项均在共享每项160响应/60迭代内关闭，合计238响应；原质量搜索和三个初值结果仍保留，不修改。owner实际wait=true/returncode0及reaped=true/returncode0。产物`outputs/tracking_v5_feasibility_first_numeric_20261028/verified.json` SHA `fddcff00e253ad7a6ec9b335bf192f0c20088bf42b548a3a1973390c6f5b21e2`。本轮新增物理、模型调用与fresh均0，不取得新的action、原时序控制或任务资格。

新候选/原质量规划/有界IK/XML FK相关10项CPU测试通过（5.38s）；新源码Ruff及格式检查通过。这是对应功能检查，不宣称全量回归重跑通过。GPU持续禁用，原18/12WAM开发/final24任务门槛不变。

下一步已登记[同一失败目标的静态有界IK](../plans/v0.0.1/recovery_pose_ik_20261028.md)，区分单点姿态存在性与固定控制器下的动态/状态约束。静态解即使存在也不代替实际可执行action或成功任务；若不能构造同一轨迹的合法前进行为，不能以继续hold取代总目标。
