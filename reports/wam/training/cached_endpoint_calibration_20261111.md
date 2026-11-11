# 已保存参考的累积状态与开度校准

2026-11-11。按[预登记](../../../plans/v0.0.1/wam/training/cached_endpoint_calibration_20261111.md)，
在30条训练demo的150锚点、300个合格D0末节点代理上，固定拟合一个位置/旋转
累计误差ridge和一个开度绝对误差线性规划；再对独立六条、已被先前研究查看的
validation demo，使用其24锚点、42/43两个seed的48条**真实完整raw生成响应**
做离线参考后处理。训练不读取validation未来标签，validation与训练demo不重叠。

## 离线状态指标

| 前缀 | 位置 RMSE raw → 校正 (m) | SO(3) RMSE raw → 校正 (rad) | opening MAE raw → 校正 | 联合分数 raw → 校正 |
| --- | --- | --- | --- | --- |
| H8 | 0.0229615 → 0.0232253 | 0.0985775 → 0.0818531 | 0.0631513 → 0.0609621 | 10.09477 → 8.95270 |
| H16 | 0.0375091 → 0.0382713 | 0.1798328 → 0.1463998 | 0.0467947 → 0.0456445 | 27.58131 → 23.77147 |
| H40 | 0.0673457 → 0.0621933 | 0.4578547 → 0.3623676 | 0.0978977 → 0.0962034 | 133.08034 → 94.96882 |

原054联合门在这48条保存响应的离线校正上全部通过：短期联合分数分别改善
11.31%和13.81%，九项组件保护、6/6 episode保护和两个seed保护均通过，短期
联合分数优于hold/历史匀速。组件保护允许少量退化，H8/H16位置实际上分别变差
0.264/0.762 mm；task0 demo16的H8联合分数也上升4.32%。seed42的H8联合分数
上升约0.80%，仍在原5%保护内。H8/H16姿态误差仍高于hold和历史匀速，不能把
联合分数胜出写成每个分量均优于基线。

该候选是生成参考后处理，模型训练输入是末节点理想FP64 C0代理，不是完整
scheduler输出；验证输入则是已保存的完整raw响应。它没有重跑WAM，未验证闭环
图像随控制变化后的新响应。离线误差不是机器人跟踪误差，也不证明接触、抓取、
关节安全或任务成功；未授予物理开发或最终验收资格。六条validation此前已被
分析，结果是有限的开发证据，不是全新盲测或预训练未见证明。

## 数值核验与费用

独立程序不导入生产fit/apply/metric：以增广最小二乘重建位姿参数，最大绝对差
`1.28e-15`；开度LP原始—对偶间隙`6.11e-16`，最大对偶约束违反`1.27e-14`。
独立积分并重算48条raw和校正参考的端点误差及原054分组/门槛，最大端点差
`7.78e-16`、最大聚合差`8.53e-14`；hold/匀速原基线沿用原056独立审核结果。
训练/验证输入与源码的消费前后SHA保持一致。

固定一次CPU拟合、一次离线验证和一次独立审计，三个子进程均实际wait/reap、
exit0、未发信号；无自动重试。34项32进程CPU专项最终通过、0跳过，全仓Ruff与
格式检查通过；首轮专项中的1项测试列表类型错误已修复，失败记录保留。源码提交
`da5c7da2`。没有新WAM调用、物理推进、GPU或fresh样本。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/cached_endpoint_calibration_20261111/candidate.json` | `729187cabc5d67146f914a749b2ad29bf1526d2307c92081efa65e10fd63238e` |
| `outputs/cached_endpoint_calibration_20261111/validation.json` | `243329418179168ded43ba08894f36e454941819585956b010654feff070def3` |
| `outputs/cached_endpoint_calibration_20261111/independent.json` | `8b412a16651f115305a28817f06a623445247f0a61b84959ac18c63688ac9b97` |
| `outputs/cached_endpoint_calibration_supervisor_20261111/fit_exit.json` | `0a6523a5e11c2c640b2c3a8463ad60b1e141a7ba7b051ede1a3adc71d55d1c85` |
| `outputs/cached_endpoint_calibration_supervisor_20261111/validate_exit.json` | `69fb489d81d1311648c056d7d7aac8c2335b3dd78a94037cbb67f31d42ac8bc6` |
| `outputs/cached_endpoint_calibration_supervisor_20261111/independent_exit.json` | `bc1c9d11b4925ac66858692755aebd1b09d100e5b663b2a2933d044a66b9898e` |
| `outputs/qa/cached_endpoint_calibration_audit_fix_20261111/result.json` | `8b0cd91ea576d92196a29a165bb0561cf2202c1b55a7bf9c8391395f13bbb540` |

下一步需评估固定参考下的实际机器人跟踪、关节约束与真实夹爪动作，并单独
验证正确参考正控；缓存参考物理重放只能作为有限反事实，不能代替新的WAM闭环。
最终18条正确参考、12条WAM开发、24条最终留出与≥20/24、各任务≥4/8门槛不变。
