# 首次预警起点的原约束细化重建

2026-11-10。按[登记计划](../../../plans/v0.0.1/tracking/joint_control/early_refiner_reconstruction_20261110.md)，从两条v4开发轨迹原初态分别重放43/52条已保存动作，核对所有前缀状态与已保存控制缓存。在首次0.10 rad预警端点，仅使用当前实测动力学、已生成参考及上一周期已完成的动作、计划和预测，重建两个反馈残差，再恢复preview与约束细化器历史；计算完成后才读取原当前步骤作为比较。

| 案例 | 重算preview首动作最大差 | 重算约束动作最大差 | 重算完整约束计划最大差 | 细化route | 优化响应数 |
| --- | ---: | ---: | ---: | --- | ---: |
| task0 s43，端点43 | `3.94e-16` | `2.81e-13` | `7.28e-13` | `interior_witness` | 69 |
| task1 s43，端点52 | `1.05e-15` | `3.13e-14` | `3.08e-13` | `interior_witness` | 77 |

两个案例的mass/J/q/qvel/site与原记录逐值相同，twist最大差 `5.55e-17`；从上一周期预测和当前实测端点重算的两类反馈创新最大差 `2.82e-18`。完整preview计划、细化计划、反馈及动作均在预登记 `atol=1e-8, rtol=0` 内重现，route一致。模型内所选计划的最小预测关节余量均约0.05 rad，首次目标的原模型质量指标均达标；原`interior_witness`分支没有将首点精度作为硬约束，这不能替代实际轨迹质量或安全评价。

重建期间无后续动作派发，仿真state、ctrl、传感器和controller旋转goal前后逐值不变。当前运行时的 `controller.initial_joint` 已记录，历史记录未单独保存该值，不能宣称历史运行时及全部隐藏状态相同；本结果只证明这两个起点的当前决策数值重现。两次CPU求解加复核分别约1.61/1.66秒，不能视为20 Hz实时资格。

## 费用和证据

两个worker及owner会话81808均实际wait/reap、exit0，全部源码及输入/资产指纹前后一致。实际新增原命令重放95周期/2375个2 ms积分步、6模型和6个OSMesa context；新派发动作0、WAM/GPU/fresh 0。两次约束细化合计146个优化响应，31项32并发CPU专项与零物理步输入dry通过。这里没有新的闭环控制、任务或最终验收结果。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `early_refiner_reconstruction_20261110/result.json` | `650c09893dcd43c0148cd5cdc178ca1c5f67b390be6802b05f645d3f50e82fc0` |
| `early_refiner_reconstruction_dry_20261110/result.json` | `cb642070275bbe10bf6429ca2d23627958fd8ca9630f8973927bed3bcbc16d52` |
| `qa/early_refiner_reconstruction_20261110/result.json` | `14e20ad7010a9314a769f16dc1f43936da1d41b90acbdd6d5c9472337249b1a1` |

下一步应从这两个可因果重建起点实际执行细化器，每周期基于分支自身测量更新预测和历史，在同一已生成chunk下核对内环安全和跟踪，并独立重放动作。不得沿用旧分支未来动作或后续世界模型输出；原18正确参考、12 WAM开发和24留出验收门槛保持。
