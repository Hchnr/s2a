# 多周期保持后恢复：原生反馈诊断（CPU）

从独立复核的实际held endpoint378重放378条真实命令，初始最后消费cursor8、下一cursor9。最多16新20Hz控制周期，总预算394周期；model/fresh均0。每个新周期先解原下一8点，若无witness再解保持上一点；仅一套已声明的shifted warm start、每项最多160响应。保留原box/rate、内环裕量、制动和首周期.01m/.05rad精度约束，双无解则不发送动作；执行后若实际内环裕量低于.05rad或TCP误差超过.01m/.05rad，则停止下一动作。记录每周期参考相位、搜索、25内环q/qvel/time、模型预测、实际误差及自身历史更新。

历史只来自旧实际368..376与本分支实际held377；继续执行后只用自身新实际反馈更新，不借旧378+。resume只消费下一点，hold不消费，不跳点/插值，原始未消费点完整记录。参考来自已生成chunk46，不调用模型、不读取对象真值规划；对象和任务谓词仅在执行后记录评价。

夹爪保持最后真实命令不变，是明确有界的arm-only机制诊断，不能作为完整夹爪/姿态协调的重定时方案。即使16周期全部成功也不代表原timed控制合格、完整WAM成功或fresh稳定成功。原18/WAM开发/final24门槛不变，GPU持续禁用。

入口`execute_tracking_v5_reference_recovery_native.py --launch`，探针`probe_tracking_v5_reference_recovery_native.py`；使用独立新outputs目录，不覆盖历史。延用上次明确的两份测试指纹9db6efa→2f09739迁移，其余冻结原v5源码仍按SHA验证。后续需独立保存命令重放，逐端点及25内环状态核验，不能用producer自报替代物理复核。

CPU producer已实际启动，结果待owner正常关闭。独立检查入口为`execute_tracking_v5_reference_recovery_native_check.py --launch`，必须先确认producer实际wait0/reap0；重放每条保存命令，逐端点与25内环q/qvel/time比较，检查参考消费、因果历史更新、原动作box/rate、实际误差/裕量及停止后无派发。当前不能宣称恢复或任务成功。
