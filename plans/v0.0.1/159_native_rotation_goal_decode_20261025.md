# 159 固定控制器数值旋转goal解码修正

158已证实quat2mat的float32路径。新模块包装151当前状态读取，非零旋转命令用实际robosuite axisangle2quat/quat2mat纯函数构造goal=current增量左乘SITE矩阵，再按Log(goal*currentR^T)提供模型输入；零旋转保持上一goal规则不变。不调用set_goal/forward/step，不改变控制器缓存。旧157/151冻结源码不修改。

原157全部164日志用新解码对实际controller goal重算，严格1e-9保持，独立实际goal检查不放宽。当前仅数值接口修复，无新物理/fit/模型/fresh或控制资格；后续新namespace重新准入、完整18条回归，原门槛保持。
