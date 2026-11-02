# 087 原生arm前向响应数据资格完成

旧30轨迹保持24fit/6内部holdout，分别4865/1315行，共6180个一周期样本；未加入086探针或最终新初态。实际controller SITE pose/twist、7关节q/qd、finger状态、motor cache、上一实际命令/goal与当前applied command/goal分别保存，下一真实motion/SO3增量/twist/关节状态独立作标签。

全部6210个端点从raw time/qpos/qvel和原record XML重建pose/twist；解析速度与epsilon=1e-6秒的FK数值导数独立一致，最大绝对差5.95934e-10，低于固定1e-7容差。pose与原实际site矩阵最大绝对差2.88658e-15。全部普通API目标、native-end measured SITE latch、当前命令与goal日志、标签/历史初始mask、来源指纹复核通过。当前action目标与未来观测明确分开，native action不能伪装成在线state输入。

三个准入进程通过（3项XML运动学测试、首真实164行检查、独立脚本导入）；全仓Ruff/格式检查通过。重建、独立复核及execution owner实际wait0/reap、无信号与全部来源不变通过。输出outputs/native_arm_response_data_20261025中的reconstruction_result.json、independent_verified.json、closure_verified.json为权威依据。

0新fit、物理积分、模型请求或fresh。仅取得数据资格，尚无条件化前向模型或新控制收益；原18/18、WAM开发及24条最终留出门槛保持，总目标未完成。
