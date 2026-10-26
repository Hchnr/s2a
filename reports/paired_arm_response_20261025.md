# 086 成对arm响应：已通过准入并启动

仅旧24fit中的demo8/1/2，close/open两个anchor共42条3339步。每组native baseline完整核验后，允许该anchor六个translation±0.1探针；旋转/夹爪不变。0fit/模型/fresh，不授予任务或控制候选资格。

44个准入进程全部通过（21项CPU检查、全部42条实际3.11 CLI dry、独立checker导入）；全仓Ruff检查/格式检查通过。固定来源/config/计划与持久化owner在outputs/paired_arm_response_20261025；实际进程身份已核实：

[
  {
    "stage": "owner_wait/launch.json",
    "pid": 3712697,
    "start_ticks": 584325992,
    "live": true
  },
  {
    "stage": "orchestration/case00/launch.json",
    "pid": 3712705,
    "start_ticks": 584326024,
    "live": true
  }
]

CPU/OSMesa显式空CUDA/LD；无自动重试。独立核验完整native前缀、实际命令差值、原始qpos/qvel的XML site运动学、两端点图像/标签、motor及实际wait/reap。当前尚无完整新数据资格或局部响应结论。

首anchor baseline（case00，task0/demo8首次close前33周期前缀+1周期）实际完成并通过独立核验：完整native前缀及一步端点、XML site pose/twist、当前两端点图像、标签、motor和采集/审查实际wait0均通过。原owner PID3712697/start_ticks584325992仍存活，已允许该anchor探针继续。尚未取得全42条/3339步数据资格，不作响应或控制收益结论。

首个anchor七条完整通过独立核验，另第二组baseline和首探针亦通过，最近快照9/42核验完成，原owner仍存活。新增paired_response_analysis.py及analyze_paired_arm_response.py已实现独立单侧/双侧差分、真实裁剪步长、SO3 world Log及物理单位换算，4项算法检查和Ruff通过；汇总再核验相同anchor原始状态/twist、固定旋转/夹爪与实际controller目标差值。首组局部同轴位置响应/名义比值0.8722/0.8980/1.0154，旋转耦合Frobenius 0.012248rad/归一化translation输入。结果保存outputs/paired_arm_response_analysis_20261025/verified_groups_01.json，属于完整anchor prefix诊断，非全42条数据资格，不证明全局动力学或闭环稳定。0额外物理/fit/模型/fresh。

前14/42条独立核验完成，task0/demo8两个anchor组完整。只读汇总outputs/paired_arm_response_analysis_20261025/verified_groups_02.json显示close/open两阶段同轴位置响应/名义比值分别[0.8722,0.8980,1.0154]与[0.4071,0.5442,0.8016]，该局部响应随所测状态/阶段显著变化；两阶段旋转耦合Frobenius分别0.012248/0.010606rad每归一化输入。不能由这两个anchor挑选全局补偿系数或唯一归因接触。原owner仍存活，进入task1 baseline；全42条及实际owner关闭尚未完成。

前21/42条独立核验完成，三个anchor组完整。第二任务首次close前anchor同轴位置响应/名义比值为[0.9237,0.9380,0.9548]，旋转耦合Frobenius0.014431rad/归一化输入；前三完整组记录在outputs/paired_arm_response_analysis_20261025/verified_groups_03.json。原owner身份保持并继续其后首次open anchor。半程数据不等于全42条或新控制资格，无fit/模型/fresh。

前28/42条及四个完整anchor组独立核验通过。第二任务open阶段同轴位置响应/名义比值[0.8923,0.0873,0.2469]，与其close阶段[0.9237,0.9380,0.9548]不同，存在很弱的局部y/z命令响应；旋转耦合Frobenius0.020623rad/归一化输入。四组结果保存outputs/paired_arm_response_analysis_20261025/verified_groups_04.json。该数据只说明这些初态/命令附近的一周期差分，不证明唯一接触机制、全局模型或稳定逆。原owner仍存活继续第三任务，尚无全量资格/fit/新控制/模型/fresh。

前36/42条独立核验完成，五个完整anchor组可分析。第三任务close阶段同轴位置响应/名义比值[0.0714,0.1239,0.1389]，旋转耦合Frobenius0.008202rad/归一化输入；弱响应不只出现于open阶段，因此不能仅按close/open标签选择固定增益。五组数据保存outputs/paired_arm_response_analysis_20261025/verified_groups_05.json。原owner继续最后一组，尚未完成全量数据/退出资格，0fit/新控制/模型/fresh。

## 完整关闭与六组响应

42条3339步、6个原native baseline、全部独立命令/前缀/XML运动学/图像/标签审查及84个阶段和execution owner实际wait0/reap/来源不变通过。data_summary.json与closure_verified.json保存原输出目录，授予数据资格，不授予任务或控制资格。完整六组响应位于outputs/paired_arm_response_analysis_20261025/complete_06.json。末组同轴位置响应/名义比值[0.8308,0.9169,1.0478]；六组总范围约0.0714–1.0478，均有非零旋转耦合。0fit/模型/fresh，尚无新逆模型或稳定控制收益。下一步先重建旧24fit/6holdout原生轨迹的关节状态、实际控制目标与一周期响应数据，不能直接将本表比例当作全局逆增益。
