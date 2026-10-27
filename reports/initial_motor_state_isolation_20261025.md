# 069 当前电机目标初态隔离：单例独立通过

同wine训练demo_11初态、seed42、228步正确世界参考，command preview0，保持原OSC、position_hold及全部参考。只将开度对齐初始化替换为保留实际当前motor target，长度1按Panda原broadcast语义复制，不读取历史专家cache作为在线输入。

八项广播等价/控制复用测试、Ruff、真实Python3.11 dry通过；独立checker另有AST复用测试通过，确认除当前motor目标准入与实际fork身份外，原机械检查保持。owner3466452/birth582909051、client3466460/birth582909081实际启动，正常wait0/reap，无自动重试，0模型/fit/fresh。

完整独立控制/初始motor保持、229实际端点标签/时间/hash、三当前图像、normal收据及来源核验通过，independent_verified.json complete=True、performance_qualified=True。native成功82步、尾连续78，末40稳定；原068 baseline/candidate同参考均仅2成功步且终点失败。首步gripper command为-1/open，motor target保留[0,0]，不再发生原+1/close方向反转。

这支持该失败案例受初始控制状态干预影响，不证明所有参考或WAM改善。原068性能失败不改判；必须在更广固定正确参考集合上验证无退化，再登记模型开发。原总验收/fresh门槛不变。
