# 084 native夹爪单通道消融：执行中

固定083 arm、参考与原初态，三条原失败case03/10/12仅替换native夹爪命令，共566步。该命令来自同轨迹sent_commands最后一维，是已知未来专家命令的离线oracle诊断，不是可部署S2A或WAM成功。原18/18及最终新初态门槛保持。

完整执行与checker AST还原检查证明arm/端点/非策略机械审查保持；无未执行shadow规划器干扰。5个准入进程全部通过（14项CPU检查、3条实际3.11 CLI dry及独立checker导入），Ruff通过。来源与config冻结在outputs/native_gripper_ablation_20261025/source_sha256.json，已核实进程身份：

[
  {
    "stage": "owner_wait/launch.json",
    "pid": 3672255,
    "start_ticks": 584099985,
    "live": true
  },
  {
    "stage": "orchestration/case00/launch.json",
    "pid": 3672263,
    "start_ticks": 584100019,
    "live": true
  }
]

CPU/OSMesa空CUDA/LD，无模型请求、拟合或fresh。逐条实际wait/reap和独立命令/motor/arm/标签/图像审查后才能解释物理结果；当前尚无完整结果。

首条（映射原083 case03）独立机械/命令/标签/图像核验完成，任务ever/terminal/stable均False，0成功步。该固定arm下替换native夹爪命令不足以修复失败，不能据此断言唯一arm原因。原owner进入第二条。新增close_native_gripper_ablation.py核验6个阶段及owner实际wait/reap、全部来源不变、566步和原083对照，仅产出oracle消融结论，deployable_s2a_qualification始终False；Ruff/导入通过，待全3条实际完成后执行。

## 完整关闭：oracle消融2/3稳定

三条566步全部采集和独立机械/命令/标签/图像审查完成，6个阶段及execution owner实际wait0/reap、无信号、来源指纹不变审查通过。原输出目录ablation_summary.json与closure_verified.json保存完整依据。映射原case03失败、case10和12稳定，后两条原083均不稳定。仅说明固定083 arm算法下这两条存在可行native夹爪命令，尚无可部署的state到夹爪恢复方案；case03仅替换夹爪命令仍不足，不能唯一归因为arm。

case10只读命令对照保存case01_command_comparison.json：083/084首次关闭周期36/38、首次后续打开82/80，实际motor首次差异周期36；末端碗到盘水平距离33.985/28.047mm。对象位置仅离线诊断，不能作为在线输入或唯一根因。0新模型/fit/fresh；原18/18及最终留出门槛不变，总目标未完成。
