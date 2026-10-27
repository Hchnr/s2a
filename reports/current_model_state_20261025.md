# 151 当前测量输入适配实现

新增src/s2a/current_model_state.py，从已刷新端点只读读取实际SITE pose/world twist、全局索引下的arm/finger q/qvel、真实motor cache、controller上一goal与当前wrench。显式上一实际command/availability；候选command与实际OSC解码goal_delta同步，再附加150已完成历史。拒绝非fixed delta接口和未建模goal限幅，读取前后完整state/ctrl/sensordata保持一致。

新增必要索引测试与150前缀测试共2项CPU通过，覆盖qpos/qvel非相同索引、标量motor缓存广播、候选实际解码、返回数组副本及历史parity；静态/格式检查通过。固定环境源码确认controller限幅属性和robot索引字段存在。尚未运行实际sim前缀parity，不把mock测试当真实传感器合同验证。

下一步实际固定环境读取并与原087/103前缀输入逐项对齐，再接149及原旋转/夹爪策略。当前无新增物理/fit/模型/fresh或控制资格，原18条/4907步与全部后续门槛保持，总目标未完成。
