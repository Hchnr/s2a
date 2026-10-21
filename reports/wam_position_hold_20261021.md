# 连续开度映射下的 WAM 开发评测 — 2026-10-21

005固定24条WAM闭环均完成600步/20Hz并独立核验，原生任务 ever **2/24**、terminal **0/24**、末尾连续40端点 stable **0/24**。004的9条oracle资格全部通过，单独计数，不能计入WAM分母。

本报告固定两个checkpoint（8800/15400）、共同guidance1.0、旧task0/1/2 init0、sampling seed42、四方案，共14,400个WAM物理端点；oracle为9条/5,400端点。H40、执行前缀40、最长80物理步重观测、35采样步/shift5、response inverse .5和arm command rate5/s与003相同，推理等待不推进物理时间。006/其他checkpoint、新初态验收及真机均不属于本批分母。

权重完整路径与saved config沿用[003报告](wam_recovery_20261021.md)中列出的8800/15400 non-EMA模型；本批每query另核checkpoint身份、seed、10-slot mask及物理单位。

## 完整矩阵与003对照

`ever`为任一端点原生成功，`terminal`为最后端点成功，`stable`为最后40端点全部成功。各组3条全数计分，未按成功病例拼接。

| checkpoint / 方案 | 005 ever / terminal / stable | 003对应方案 ever / terminal / stable |
|---|---:|---:|
| 15400_masked / position_hold | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 |
| 15400_masked / recovering_position_hold | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 |
| 15400_masked / recovering_position_hold_alpha05 | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 |
| 15400_masked / recovering_position_hold_alpha0 | 2/3 / 0/3 / 0/3 | 1/3 / 0/3 / 0/3 |
| 8800_masked / position_hold | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 |
| 8800_masked / recovering_position_hold | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 |
| 8800_masked / recovering_position_hold_alpha05 | 0/3 / 0/3 / 0/3 | 1/3 / 0/3 / 0/3 |
| 8800_masked / recovering_position_hold_alpha0 | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 |

原生成功端点的完整范围（零基step，仍继续执行预定600步）：

- 15400_masked / recovering_position_hold_alpha0 / task0_init0：273–337, 341–488, 501–513，共226端点；末尾连续0端点。
- 15400_masked / recovering_position_hold_alpha0 / task2_init0：180，共1端点；末尾连续0端点。

## 同一首query内，执行映射改了什么

24条的第一张输入PNG、完整40×128 raw/normalized预测、模型条件及初态physics/anchor与003对应方案逐值相同。首query后的夹爪命令会改变机器人/对象状态，后续观测和WAM预测随之分叉，因此全600步对照不代表始终执行同一参考。

旧v2把超过.02的下降趋势直接锁存为close-to-limit。新v3在未锁存时按absolute reference与已知motor目标之差作三值位置跟踪（deadband .07）；仅reference≤.60且关闭方向成立时进入close-to-limit。释放仍需当前/历史开度余量、参考回升及完整.05s证据，之后恢复部分开度位置跟踪。参数是当前Panda/20Hz接口的固定开发候选，不是握力估计或通用抓取保证。

下表每行限制在两次执行都尚未发出第二次query的共同物理区间；阈值为描述性reference>.8、observed_after<.2，不用于策略或验收。`高ref锁存`为在该区间新发生的关闭锁存；高reference下的单次位置减量命令不等于full-close锁存。

| checkpoint / 方案 / 场景 | 共同步数 | 高ref低actual 003→005 | 高ref新锁存 003→005 | 首gripper / arm命令分叉step |
|---|---:|---:|---:|---:|
| 8800_masked / position_hold / task0_init0 | 40 | 0→0 | 0→0 | 1 / 2 |
| 8800_masked / position_hold / task1_init0 | 40 | 0→0 | 1→0 | 0 / 1 |
| 8800_masked / position_hold / task2_init0 | 40 | 0→0 | 2→0 | 1 / 2 |
| 8800_masked / recovering_position_hold / task0_init0 | 41 | 0→0 | 0→0 | 1 / 2 |
| 8800_masked / recovering_position_hold / task1_init0 | 41 | 0→0 | 1→0 | 0 / 1 |
| 8800_masked / recovering_position_hold / task2_init0 | 44 | 0→0 | 0→0 | 1 / 2 |
| 8800_masked / recovering_position_hold_alpha05 / task0_init0 | 40 | 0→0 | 0→0 | 1 / 2 |
| 8800_masked / recovering_position_hold_alpha05 / task1_init0 | 41 | 0→0 | 1→0 | 0 / 1 |
| 8800_masked / recovering_position_hold_alpha05 / task2_init0 | 44 | 0→0 | 0→0 | 1 / 2 |
| 8800_masked / recovering_position_hold_alpha0 / task0_init0 | 40 | 0→0 | 0→0 | 1 / 2 |
| 8800_masked / recovering_position_hold_alpha0 / task1_init0 | 41 | 0→0 | 1→0 | 0 / 1 |
| 8800_masked / recovering_position_hold_alpha0 / task2_init0 | 44 | 0→0 | 0→0 | 1 / 2 |
| 15400_masked / position_hold / task0_init0 | 40 | 0→0 | 1→0 | 1 / 2 |
| 15400_masked / position_hold / task1_init0 | 40 | 0→0 | 0→0 | 1 / 2 |
| 15400_masked / position_hold / task2_init0 | 40 | 0→0 | 2→0 | 1 / 2 |
| 15400_masked / recovering_position_hold / task0_init0 | 44 | 0→0 | 1→0 | 1 / 2 |
| 15400_masked / recovering_position_hold / task1_init0 | 41 | 0→0 | 0→0 | 1 / 2 |
| 15400_masked / recovering_position_hold / task2_init0 | 43 | 0→0 | 1→0 | 1 / 2 |
| 15400_masked / recovering_position_hold_alpha05 / task0_init0 | 41 | 0→0 | 0→0 | 1 / 2 |
| 15400_masked / recovering_position_hold_alpha05 / task1_init0 | 41 | 0→0 | 0→0 | 1 / 2 |
| 15400_masked / recovering_position_hold_alpha05 / task2_init0 | 41 | 0→0 | 0→0 | 1 / 2 |
| 15400_masked / recovering_position_hold_alpha0 / task0_init0 | 41 | 0→0 | 0→0 | 1 / 2 |
| 15400_masked / recovering_position_hold_alpha0 / task1_init0 | 41 | 0→0 | 0→0 | 1 / 2 |
| 15400_masked / recovering_position_hold_alpha0 / task2_init0 | 41 | 0→0 | 0→0 | 1 / 2 |

分叉按精确数值非相等记录，不设置拟合后的显著性阈值；极小数值残差本身不代表实际运动差异。该极端错配计数为零也不代表所有开度都正确；高ref/低actual可能发生在先前低ref合法锁存后的释放等待。motor target、实际开度和握力是不同量。更长600步的同类计数、首次reason/command/motor明细保存在配对诊断JSON，只作后续输入不同条件下的描述，不能唯一归因模型或控制。

![首query开度对照](../outputs/wam_stability_20261021/position_report_final_20261021/first_query_gripper_pairs.png)

## 生成参考与实际允许目标的跟踪

alpha1保留raw WAM增量；alpha.5逐增量在SO(3)上缩放旋转，alpha0令每个旋转增量为单位，因此只保持本次query实测anchor朝向，后续query重新取anchor。local平移slot数值不变也会因修改后的朝向积分改变world路径；alpha.5/0都不是纯执行补偿，alpha0不能认证原checkpoint完整输出。

admitted表示实际状态相对本步允许目标；R/E模型时钟表示raw/effective参考按已消费模型时间采样；R/E名义时钟按每query后的物理20Hz推进，耗尽后夹到执行前缀末端。低admitted误差可能来自暂停/桥接/丢弃后缀，不能单独证明同一名义轨迹跟踪改善或任务成功。位置单位mm，SO3单位rad。

| checkpoint / 方案 | admitted TCP / SO3 | R模型时钟 TCP / SO3 | E模型时钟 TCP / SO3 | R名义时钟 TCP / SO3 | E名义时钟 TCP / SO3 |
|---|---:|---:|---:|---:|---:|
| 15400_masked / position_hold | 83.383 / 0.312 | 83.383 / 0.312 | 83.383 / 0.312 | 83.383 / 0.312 | 83.383 / 0.312 |
| 15400_masked / recovering_position_hold | 30.105 / 0.071 | 29.932 / 0.070 | 29.932 / 0.070 | 90.989 / 0.321 | 90.989 / 0.321 |
| 15400_masked / recovering_position_hold_alpha05 | 32.757 / 0.187 | 32.419 / 0.240 | 32.628 / 0.186 | 77.378 / 0.392 | 77.992 / 0.229 |
| 15400_masked / recovering_position_hold_alpha0 | 27.210 / 0.190 | 33.582 / 0.379 | 26.912 / 0.201 | 54.795 / 0.479 | 47.832 / 0.201 |
| 8800_masked / position_hold | 112.143 / 0.231 | 112.143 / 0.231 | 112.143 / 0.231 | 112.143 / 0.231 | 112.143 / 0.231 |
| 8800_masked / recovering_position_hold | 35.551 / 0.135 | 34.977 / 0.133 | 34.977 / 0.133 | 64.315 / 0.359 | 64.315 / 0.359 |
| 8800_masked / recovering_position_hold_alpha05 | 23.960 / 0.056 | 24.920 / 0.195 | 23.718 / 0.056 | 44.459 / 0.351 | 44.547 / 0.149 |
| 8800_masked / recovering_position_hold_alpha0 | 38.669 / 0.294 | 38.690 / 0.413 | 37.157 / 0.286 | 106.373 / 0.681 | 105.898 / 0.286 |

每组3条合计90物理秒。下表模型时间与discard/bridge/pause均按实际日志计，不把暂停减少误差当能力；逐case消费fraction、未执行后缀及原始/有效目标保留在完整审计JSON。advance标志可包括完成桥的端点，故不必小于post-step bridging计数。

| checkpoint / 方案 | 消费模型秒 / 90s | consumed / generated | discard模型秒 | bridge / advance / pause端点 |
|---|---:|---:|---:|---:|
| 15400_masked / position_hold | 90.000 | 1.000 | 0.000 | 0 / 0 / 0 |
| 15400_masked / recovering_position_hold | 40.730 | 0.727 | 13.487 | 60 / 52 / 480 |
| 15400_masked / recovering_position_hold_alpha05 | 47.275 | 0.763 | 12.396 | 60 / 43 / 284 |
| 15400_masked / recovering_position_hold_alpha0 | 47.082 | 0.759 | 11.024 | 138 / 59 / 218 |
| 8800_masked / position_hold | 90.000 | 1.000 | 0.000 | 0 / 0 / 0 |
| 8800_masked / recovering_position_hold | 38.687 | 0.691 | 13.447 | 69 / 66 / 374 |
| 8800_masked / recovering_position_hold_alpha05 | 58.333 | 0.858 | 6.465 | 49 / 41 / 244 |
| 8800_masked / recovering_position_hold_alpha0 | 32.129 | 0.574 | 20.089 | 104 / 94 / 259 |

## 核验与范围

- 004旧v2无gov、新v3无gov、新v3加recovering各3/3 oracle稳定，TCP≤10mm/SO3≤.05rad；旧v2物理序列逐值复现。reference按当前物理step索引成功真值，仅是特权执行资格，9条从24条WAM分母中排除。独立机械阻塞的observed_blockage/recovery_from_blockage门禁仍未通过，不提升为广义恢复资格。
- 新auditor显式fork旧checker，复算响应逆、controller goal、三值电机积分/零保持、qpos→开度/跨步连续性，以及独立GripperReplay关闭/释放锁存。RecoveryReplay独立复算桥gate、时钟和速度界；各query核checkpoint/seed/mask/域/反归一化、raw/effective128和physics前后SHA。
- controller goal旋转重建仍为预先声明的1e-6 vendor数值容差；其余状态/误差1e-10、dt1e-9保持，未因结果放宽task阈值。查询前后full-state哈希由冻结logger记录，未保存每次full-state数组以重新独立哈希；源码/包版本也不等于完整OS/共享库二进制闭包。
- 物理仿真CPU/OSMesa，模型登记仅物理GPU5/7；外部Cosmos与R1/R2只读，无真机。3旧开发初态、1 sampling seed与当前Panda范围不能证明训练未见或新初态泛化，不能唯一判定权重损坏或把余失败全部归于WAM/仅S2A。
- 首query回归与开度诊断回答局部接口行为；原生稳定任务谓词回答执行结果。contact/目标高度仅作观测指标，未作为adapter/governor输入；contact0也不等于任务必失败。

## 图与证据

每个case图分别记录三种governor alpha与无gov对照的实际路径、开度/motor、成功谓词和消费时钟，不将后续不同观测的轨迹画成同一参考回放。

![8800_masked_task0_init0](../outputs/wam_stability_20261021/position_report_final_20261021/8800_masked_task0_init0.png)
![8800_masked_task1_init0](../outputs/wam_stability_20261021/position_report_final_20261021/8800_masked_task1_init0.png)
![8800_masked_task2_init0](../outputs/wam_stability_20261021/position_report_final_20261021/8800_masked_task2_init0.png)
![15400_masked_task0_init0](../outputs/wam_stability_20261021/position_report_final_20261021/15400_masked_task0_init0.png)
![15400_masked_task1_init0](../outputs/wam_stability_20261021/position_report_final_20261021/15400_masked_task1_init0.png)
![15400_masked_task2_init0](../outputs/wam_stability_20261021/position_report_final_20261021/15400_masked_task2_init0.png)

- 完整33审计：`outputs/wam_stability_20261021/position_full_complete_20261021.json`，SHA `fe714dad35dad3d6d3ab933be00e166c94c0c54cff6611730c439e782759299e`。
- 24对局部开度诊断：`outputs/wam_stability_20261021/position_gripper_diagnostic_complete_20261021.json`，SHA `913accfb454d8280521a249b4acf32b870cb15e8506c9682374875e743d6f623`。
- 图、七页PDF与输出/输入指纹：`outputs/wam_stability_20261021/position_report_final_20261021/report_manifest.json`。

稳定成功只按完整冻结矩阵报告；若有成功，仍限本开发协议，后续须固定候选另做新初态验收。若仍失败，保留已识别的映射改善与参考/消费/任务指标，继续分开验证模型参考和执行范围。
