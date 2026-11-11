# 双点模型余量的实际反馈对照

2026-11-11。[匹配静态对照](../../../../reports/tracking/joint_control/margin_two_step_search_20261111.md)
在task2 terminal058 step80找到满足新模型余量的双点候选，但raw step32没有找到；
在线初值由分支自己的原细化器生成，不等于该静态对照保存的已派发计划，不能假定
静态结果会复现。现以同一task2开发初态、已完成前缀与已生成chunk，从raw step32和
terminal058 step80各运行一个新CPU/OSMesa反馈分支，最多各8个原时钟周期。

每周期使用[余量求解器](../../../../src/s2a/research/trajectory_search/margin_two_step_search.py)，
前两点模型内分别留0.0005 m / 0.003 rad，只有该分支已评估、安全且符合余量的计划
才能进入原独立双点及全时域安全审核；独立重建的前两点还须使用相同`1e-10`容差
再次满足模型余量。无候选或审核失败则派发前停止；不自动改回原搜索，也不增加
回退rollout。原0.01 m / 0.05 rad实际误差门、关节缓冲、动作、终端及0.1 s制动
条件和每决策320响应、最多60及120迭代均不变。完成周期后若实际2 ms关节缓冲
失守则不再派发；初始内环误差未知置零，夹爪固定0。

分别与同一起点的[原guarded双点反馈](../../../../reports/tracking/joint_control/task2_guarded_two_step_feedback_20261111.md)
及[已完成计划回退反馈](../../../../reports/tracking/joint_control/task2_causal_shift_fallback_20261111.md)
并列实际达标周期（此前raw均1/1，terminal分别4/4与5/5）、位置/SO(3)误差、2 ms
关节余量、grasp、任务标签、停止及墙钟。三个策略不同，路径分歧后各用自身实际
状态与已完成历史；两个开发变体共享初态和seed43，不视为独立泛化样本。固定已生成
chunk可能跨原重规划时刻，不视为新WAM闭环；只有完整8周期满足原条件才称短窗通过。

producer/checker各2个worker，每phase最多128控制周期/3200内步、6模型/6软件context、
16次原细化/7680响应、16次余量双点搜索/5120响应、32次初值LP、3200独立审核端点
查询/41600模型转换步。checker不得导入候选策略，只重放已派发动作，不重求无候选
原因。先验证余量独立审核和失效关闭、32并发专项及零模型dry，固定源码和输入SHA；
监督及worker实际wait/reap，不自动重试。0新WAM、GPU或最终留出样本，原18正确参考、
12开发和24最终留出门槛不变。
