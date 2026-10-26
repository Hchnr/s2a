# 151 当前仿真端点到合格模型输入

在调用方已刷新当前端点后只读读取SITE pose/world twist、实际关节q/qvel、finger q/qvel、gripper motor cache、controller上一goal与当前wrench。显式上一实际command/available，候选command经实际controller input/output bounds解码为goal_delta，不改变仿真/controller。要求固定delta OSC且无goal位置/旋转额外限幅，否则拒绝。历史由150已完成状态附加，不读未来标签。

当前仅输入适配与索引/只读合同测试；后续必须实际sim前缀与训练输入parity，再绑定149逆解、083旋转/081夹爪和原18条控制回归。无控制性能声明，原门槛保持。
