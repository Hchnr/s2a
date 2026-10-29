# WAM recovery 开发对照：2026-10-21

完整执行与独立审计了 **24 条 WAM 闭环开发评测（14,400 个 20 Hz 端点）**，以及独立的 6 条 oracle 资格正控（3,600 端点）。本批四个方案均未取得末尾连续40端点的稳定成功，不能提升为已完成稳定任务能力。

本报告只统计两个 checkpoint、共同 guidance=1.0、3 个旧开发场景 init0/seed42、4 个方案、每条600步/30秒。无 governor 的方案执行40步前缀；governor 使用恢复桥及模型时间重定时，最迟80物理步重新查询。推理等待不推进物理时间。oracle按当前物理步选取特权 fixed-world chunks，仅用于执行资格，不能计入 WAM 分母。8400/8600 的附加评测、后继 v3 oracle 资格以及 fresh reservation 均不属于本报告。


本批服务实际加载的权重路径（均non-EMA，guidance1.0，seed42）：

- 8800_masked：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_21src_20261012_warm_start_run01/cosmos3_action/action128/action128_wam_nano_1b_22src_20261012_warm_start_videocap30_static_vae_aot_run1_from6k/checkpoints/iter_000008800/model`；saved config：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_21src_20261012_warm_start_run01/cosmos3_action/action128/action128_wam_nano_1b_22src_20261012_warm_start_videocap30_static_vae_aot_run1_from6k/config.yaml`。
- 15400_masked：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_23src_20261015_warm_start_from22src_i8800_run01/cosmos3_action/action128/action128_wam_nano_1b_23src_20261015_warm_start_from22src_20261012_i8800_from12src_e2aux345_i6200_videocap30_static_vae_aot_run1/checkpoints/iter_000015400/model`；saved config：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_23src_20261015_warm_start_from22src_i8800_run01/cosmos3_action/action128/action128_wam_nano_1b_23src_20261015_warm_start_from22src_20261012_i8800_from12src_e2aux345_i6200_videocap30_static_vae_aot_run1/config.yaml`。

## 完整任务结果

`stable` = 最后40个端点全部native success；`terminal` = 最后一个端点success；`ever` = 任一端点success。计分来自模拟器原生谓词，目标双指contact只作观测指标，不能替代任务谓词。

| checkpoint | 方案 | complete | ever | terminal | stable |
|---|---|---:|---:|---:|---:|
| 15400_masked | intent_hold | 3/3 | 0/3 | 0/3 | 0/3 |
| 15400_masked | recovering_intent_hold | 3/3 | 0/3 | 0/3 | 0/3 |
| 15400_masked | recovering_intent_hold_alpha05 | 3/3 | 0/3 | 0/3 | 0/3 |
| 15400_masked | recovering_intent_hold_alpha0 | 3/3 | 1/3 | 0/3 | 0/3 |
| 8800_masked | intent_hold | 3/3 | 0/3 | 0/3 | 0/3 |
| 8800_masked | recovering_intent_hold | 3/3 | 0/3 | 0/3 | 0/3 |
| 8800_masked | recovering_intent_hold_alpha05 | 3/3 | 1/3 | 0/3 | 0/3 |
| 8800_masked | recovering_intent_hold_alpha0 | 3/3 | 0/3 | 0/3 | 0/3 |

完整观察到的 2 个 ever 案例（仍以末尾native谓词单独验收）：

- 8800_masked / recovering_intent_hold_alpha05 / task0_init0：native成功step 154、157（非连续，共2端点），末尾连续成功0；目标双指contact 48端点。
- 15400_masked / recovering_intent_hold_alpha0 / task0_init0：native成功step 199–426，共228端点，末尾连续成功0；目标双指contact 48端点。

15400 alpha0 的连续成功199–426共228端点，即11.4秒；随后427–599全部失败，不能把中途维持成功视为末尾稳定完成。8800 alpha0.5仅step154和157非连续通过，不能写成连续0.1秒成功。成功后仍继续查询/执行是冻结协议的一部分；本批没有用特权成功谓词触发停机。

## 参考、控制及误差的不同含义

四方案均使用已资格验证的 task-agnostic GripperIntentHold 和同一 response arm inverse/OSC 接口。adapter只使用当前/历史开度、真实motor目标和已生成参考，不输入对象、接触、任务标签或未来真值。motor达到闭合限位后command=0保持原闭合motor目标；观测开度在物体/机械阻塞下可以明显大于motor目标，不能将0命令解释为零握力，也不能把闭合目标当作实际夹爪开度/力。

alpha1保留每个raw WAM旋转增量；alpha0.5把每个增量的SO(3)对数乘0.5后重新积分，改变了模型参考。alpha0将每个旋转增量置为单位旋转：**只在该次query的实测TCP anchor朝向保持**，下一次query重新取当前实测anchor，绝非全episode固定朝向。平移与开度slot仍来自WAM且数值不变，但local平移增量通过修改后的朝向积分，world位置轨迹也可以改变；不是在同一world位置轨迹上只换朝向。alpha0仅为部分输出分量消融，不能用于认证原checkpoint完整输出。raw/effective128及各query anchor均已保留并独立核验。

`admitted`：实际状态相对governor本步允许执行的目标；`raw/effective model-clock`：相对当前已消费模型时间的原始/修改参考；`raw/effective nominal-clock`：相对该query后按物理20Hz推进、耗尽后夹到40端点末端的原始/修改参考。重定时会暂停/桥接/丢弃剩余suffix，因此admitted误差下降不能证明同一条名义WAM轨迹获得更好执行，也不能单靠低RMSE证明任务成功。不同variant闭环观测变化后会生成不同后续参考，并非全episode固定同一轨迹。

下表对每个3-case组等长端点合并计算RMSE，位置单位mm，朝向rad；R=raw、E=effective。

| checkpoint / 方案 | admitted TCP / SO3 | R模型时钟 TCP / SO3 | E模型时钟 TCP / SO3 | R名义时钟 TCP / SO3 | E名义时钟 TCP / SO3 |
|---|---:|---:|---:|---:|---:|
| 15400_masked / intent_hold | 70.051 / 0.255 | 70.051 / 0.255 | 70.051 / 0.255 | 70.051 / 0.255 | 70.051 / 0.255 |
| 15400_masked / recovering_intent_hold | 30.200 / 0.047 | 30.006 / 0.047 | 30.006 / 0.047 | 72.992 / 0.317 | 72.992 / 0.317 |
| 15400_masked / recovering_intent_hold_alpha05 | 29.790 / 0.100 | 31.080 / 0.184 | 29.470 / 0.098 | 96.635 / 0.402 | 96.284 / 0.190 |
| 15400_masked / recovering_intent_hold_alpha0 | 22.594 / 0.088 | 27.275 / 0.339 | 22.244 / 0.082 | 35.873 / 0.409 | 30.514 / 0.082 |
| 8800_masked / intent_hold | 116.658 / 0.330 | 116.658 / 0.330 | 116.658 / 0.330 | 116.658 / 0.330 | 116.658 / 0.330 |
| 8800_masked / recovering_intent_hold | 29.744 / 0.118 | 29.581 / 0.118 | 29.581 / 0.118 | 70.756 / 0.367 | 70.756 / 0.367 |
| 8800_masked / recovering_intent_hold_alpha05 | 25.926 / 0.045 | 25.935 / 0.181 | 25.776 / 0.045 | 60.061 / 0.359 | 60.291 / 0.144 |
| 8800_masked / recovering_intent_hold_alpha0 | 42.467 / 0.263 | 42.457 / 0.433 | 41.546 / 0.249 | 76.932 / 0.642 | 74.869 / 0.249 |

## 模型时间消费与恢复桥

每个case以30物理秒为分母；discard仅统计重query时尚未消费的40端点执行前缀尾部，剩余预算末尾未消费量另列。桥恢复是独立复算的两类误差不退化gate，不是对象接触或任务成功gate。

| checkpoint / 方案 / 场景 | 消费模型秒 / 30s | generated fraction | discard秒 | 预算末剩余秒 | bridge / advance / pause端点 | 目标双指contact |
|---|---:|---:|---:|---:|---:|---:|
| 8800_masked / intent_hold / task0_init0 | 30.000 | 1.000 | 0.000 | 0.000 | 0 / 0 / 0 | 0 |
| 8800_masked / intent_hold / task1_init0 | 30.000 | 1.000 | 0.000 | 0.000 | 0 / 0 / 0 | 0 |
| 8800_masked / intent_hold / task2_init0 | 30.000 | 1.000 | 0.000 | 0.000 | 0 / 0 / 0 | 0 |
| 8800_masked / recovering_intent_hold / task0_init0 | 12.161 | 0.676 | 4.353 | 1.485 | 31 / 22 / 53 | 207 |
| 8800_masked / recovering_intent_hold / task1_init0 | 11.974 | 0.665 | 5.282 | 0.744 | 20 / 16 / 182 | 0 |
| 8800_masked / recovering_intent_hold / task2_init0 | 14.009 | 0.778 | 3.213 | 0.778 | 28 / 18 / 147 | 0 |
| 8800_masked / recovering_intent_hold_alpha05 / task0_init0 | 17.245 | 0.784 | 3.079 | 1.676 | 16 / 15 / 60 | 48 |
| 8800_masked / recovering_intent_hold_alpha05 / task1_init0 | 12.830 | 0.713 | 3.867 | 1.303 | 17 / 15 / 160 | 0 |
| 8800_masked / recovering_intent_hold_alpha05 / task2_init0 | 18.928 | 0.860 | 2.272 | 0.799 | 15 / 13 / 35 | 0 |
| 8800_masked / recovering_intent_hold_alpha0 / task0_init0 | 8.101 | 0.450 | 8.258 | 1.642 | 47 / 41 / 91 | 25 |
| 8800_masked / recovering_intent_hold_alpha0 / task1_init0 | 12.921 | 0.718 | 3.904 | 1.175 | 31 / 23 / 181 | 198 |
| 8800_masked / recovering_intent_hold_alpha0 / task2_init0 | 12.044 | 0.669 | 4.799 | 1.157 | 34 / 33 / 47 | 0 |
| 15400_masked / intent_hold / task0_init0 | 30.000 | 1.000 | 0.000 | 0.000 | 0 / 0 / 0 | 0 |
| 15400_masked / intent_hold / task1_init0 | 30.000 | 1.000 | 0.000 | 0.000 | 0 / 0 / 0 | 0 |
| 15400_masked / intent_hold / task2_init0 | 30.000 | 1.000 | 0.000 | 0.000 | 0 / 0 / 0 | 0 |
| 15400_masked / recovering_intent_hold / task0_init0 | 10.953 | 0.608 | 5.597 | 1.450 | 21 / 18 / 180 | 0 |
| 15400_masked / recovering_intent_hold / task1_init0 | 19.301 | 0.804 | 2.699 | 2.000 | 14 / 11 / 41 | 0 |
| 15400_masked / recovering_intent_hold / task2_init0 | 11.063 | 0.615 | 5.337 | 1.599 | 22 / 19 / 222 | 0 |
| 15400_masked / recovering_intent_hold_alpha05 / task0_init0 | 7.407 | 0.412 | 9.094 | 1.499 | 29 / 27 / 95 | 0 |
| 15400_masked / recovering_intent_hold_alpha05 / task1_init0 | 16.144 | 0.807 | 2.598 | 1.257 | 12 / 13 / 120 | 35 |
| 15400_masked / recovering_intent_hold_alpha05 / task2_init0 | 15.414 | 0.771 | 4.061 | 0.525 | 13 / 13 / 58 | 0 |
| 15400_masked / recovering_intent_hold_alpha0 / task0_init0 | 18.969 | 0.862 | 1.324 | 1.706 | 26 / 26 / 29 | 48 |
| 15400_masked / recovering_intent_hold_alpha0 / task1_init0 | 26.902 | 0.961 | 0.143 | 0.955 | 5 / 5 / 0 | 0 |
| 15400_masked / recovering_intent_hold_alpha0 / task2_init0 | 13.308 | 0.665 | 5.154 | 1.538 | 22 / 20 / 86 | 464 |

## 正控、复核与边界

- chunked oracle资格为intent_hold 3/3、recovering+intent_hold 3/3稳定成功；原native成功参考也经审计。正控选择是按当前物理step索引的特权真值，不代表WAM在线能力，也不证明所有大角度/阻塞响应范围已合格。
- 全30个case均600步、20Hz，WAM24与oracle6分开计数。冻结auditor重算因果响应、夹爪命令、motor仿射目标、gripper substep、任务谓词、raw/effective积分、10-slot mask、query身份/seed/固定anchor、查询前后physics SHA，以及独立v2 governor桥gate/时钟/速度界。
- 仅controller goal旋转重建使用统一1e-6数值容差，原因是vendor quat2mat的float32路径及NumPy标量转换；body quaternion重排本身未作量化修正。其余状态/误差复核保持原1e-10、实际dt1e-9，任务stable阈值没有放宽。
- 报告生成前及生成后重新核验完整audit snapshot列出的所有输入SHA；图只读取这些既有日志，0新推理、0新物理。物理仿真为CPU/OSMesa，模型仅登记物理GPU5/7，属于用户允许的4–7；无真机。
- 3个旧开发初态和一个simulation seed不足以证明新初态、多任务或训练未见泛化。未把Cosmos只读来源合同与实际历史训练消费/图像-pose物理同步混为一谈；本批也不认证checkpoint权重损坏或唯一归因WAM/S2A。

## 图与可恢复证据

每个paired图包含三种governor alpha方案与无governor基线的实际TCP XY、admitted/raw-model-clock误差、实测/参考/motor开度、累计模型/重定时时钟与native成功谓词。不同线型表示不同物理量；motor目标并非测得开度或力。

![8800_masked_task0_init0](../outputs/wam_stability_20261021/recovery_report_final_20261021/8800_masked_task0_init0.png)
![8800_masked_task1_init0](../outputs/wam_stability_20261021/recovery_report_final_20261021/8800_masked_task1_init0.png)
![8800_masked_task2_init0](../outputs/wam_stability_20261021/recovery_report_final_20261021/8800_masked_task2_init0.png)
![15400_masked_task0_init0](../outputs/wam_stability_20261021/recovery_report_final_20261021/15400_masked_task0_init0.png)
![15400_masked_task1_init0](../outputs/wam_stability_20261021/recovery_report_final_20261021/15400_masked_task1_init0.png)
![15400_masked_task2_init0](../outputs/wam_stability_20261021/recovery_report_final_20261021/15400_masked_task2_init0.png)

- 完整独立snapshot：`outputs/wam_stability_20261021/recovery_full_complete_20261021.json`，SHA256 `b5ad882e7e696ec7d57a05987585c7ca5eea2ffc4c957c48ac00459c99a30eef`。
- 六页PDF：`outputs/wam_stability_20261021/recovery_report_final_20261021/paired_recovery.pdf`。
- 逐case报告指标、输出指纹与全部输入复核：`outputs/wam_stability_20261021/recovery_report_final_20261021/report_manifest.json`。

下一阶段先解决中途完成后再失效的稳定性，并以不使用对象标签/接触/成功谓词的在线规则单独资格验收；保留统一预算和原分母。源数据排除链尚未闭合的fresh reservation不能提前宣称已独立验收。本批结果支持继续诊断参考旋转、开度与执行时间的组合，尚不能宣称已达到用户的稳定成功目标。
