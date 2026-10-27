# 097 独立训练轨迹的首次闭合响应覆盖

096再次出现holdout预测通过而未训练局部响应失败。原9条差分监督来自3个anchor，其中未包含其他21条fit轨迹的受控动作变化。将原24fit中除086三条profile以外的全部21条纳入，任务各7条；不根据接触真值、性能或目标方向筛选。每条取原native命令首次gripper>0周期，固定baseline和x/y/z各±0.1归一化平移探针，裁剪规则与086一致，旋转/夹爪与前缀命令不变。

显式共21anchor/147trial/8281控制周期；baseline须先精确匹配native再运行本组探针。新采样全部归fit，原6holdout和086六anchor验证保留、[0,3,5]/[1,2,4]身份不变。原已看过的anchor是开发验证，不升级为最终新初态证据。不使用新数据挑选超参数；任何新fit须另外登记、唯一执行并独立验证。此次只采集，不启动fit或改变控制器。

同086保存全部endpoint实际state/命令/goal/图像/标签、gripper缓存和速度，独立XML FK/数值twist、前缀一致性及probe条件审查，持久owner逐trial采集/检查实际wait/reap并冻结来源。不覆盖086或其他产物。新namespace outputs/training_response_coverage_20261025。环境初始化内部步骤另列，世界模型/fresh/fit均0。CPU/OSMesa显式空CUDA/LD。全部数据资格和生命周期完成前不得用于训练，不授予控制资格，原18/18和WAM开发/最终24条目标保持。
