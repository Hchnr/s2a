# 三任务夹爪反馈的独立重放修复

原[成对实验](cached_calibration_gripper_feedback_20261111.md)生产阶段六分支正常
关闭，20个新周期、148总周期/3700内步。原checker六worker在进入物理运行前均
因候选模块隔离检查失败退出1，已真实wait/reap、未发信号；源入口在模块级导入
夹爪历史辅助函数，进而导入了`loaded_velocity_control`。原checker结果
SHA `2d62c48d37634dc659b0ad71ba655851c6484c0c72e49e8a11a5bc5cd869fcf3`，
原记录保持失败，不能称其已独立重放。

本项登记一次新独立重放：不修改已被生产来源SHA绑定的原代码，不复制或重做
生产、不选择新动作；建立不导入任何臂/夹爪候选或校准生产函数的新checker，
从原初态分别重放生产六分支实际命令，独立重算raw/calibrated参考及两套误差。
逐周期核对完整状态、全部25个q/qvel/time、开度/电机、目标接触和任务标签；
记录quality停止原因由生产结果提供而非重新求解。无动作分支仍复现完整前缀。

固定生产结果SHA `da29aa0fade42b572c427cedad831f2dcc8a3f3de54da900af9b983fa874f55b`；
沿用三任务相同输入和原最大预算，实际应为148周期/3700内步、0新决策。六个
CPU/OSMesa worker各自实际wait/reap；失败保留，不覆盖或自动重试。零新WAM、
GPU及fresh；重放一致不授予完整控制/任务资格。
