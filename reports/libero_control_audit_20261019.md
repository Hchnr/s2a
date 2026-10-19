# LIBERO 控制合同与夹爪诊断 — 2026-10-19

完整性检查通过与任务效果分别报告。所有结果为独立MuJoCo仿真，没有真机执行。

## 已见5条回归

| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |
| --- | ---: | ---: | ---: |
| native_zero | 4/5 | 4.566 | 0.006278 |
| native_zero_refreshed | 5/5 | 0.468 | 0.002518 |
| native_open_unrefreshed | 4/5 | 4.580 | 0.006293 |
| native_open | 5/5 | 0.414 | 0.002513 |
| native_measured | 5/5 | 0.429 | 0.002513 |
| affine_position | 2/5 | 27.231 | 0.030992 |
| affine_native | 5/5 | 30.689 | 0.032901 |
| affine_intent | 4/5 | 29.543 | 0.031948 |
| response_position | 1/5 | 4.311 | 0.007791 |
| response_native | 5/5 | 4.857 | 0.008012 |
| response_intent | 4/5 | 4.779 | 0.008077 |
| affine_latch | 5/5 | 30.672 | 0.032895 |
| response_latch | 5/5 | 4.820 | 0.007725 |

## 额外10条episode

| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |
| --- | ---: | ---: | ---: |
| native_measured | 10/10 | 0.462 | 0.004473 |
| affine_position | 2/10 | 24.651 | 0.036690 |
| affine_intent | 10/10 | 28.197 | 0.033904 |
| response_position | 0/10 | 3.934 | 0.008782 |
| response_intent | 9/10 | 4.266 | 0.007429 |
| affine_latch | 9/10 | 28.060 | 0.036882 |
| response_latch | 9/10 | 4.212 | 0.007278 |

## 输入延迟与噪声压力测试

| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |
| --- | ---: | ---: | ---: |
| affine_position_delay50ms | 1/10 | 26.645 | 0.044271 |
| affine_position_delay100ms | 1/10 | 30.565 | 0.055004 |
| affine_position_noisy | 2/10 | 24.694 | 0.037048 |
| affine_position_delay50ms_noisy | 1/10 | 26.629 | 0.044278 |
| affine_intent_delay50ms | 10/10 | 30.805 | 0.041663 |
| affine_intent_delay100ms | 9/10 | 34.296 | 0.051471 |
| affine_intent_noisy | 10/10 | 28.225 | 0.034007 |
| affine_intent_delay50ms_noisy | 10/10 | 30.902 | 0.041425 |
| response_latch_delay50ms | 3/10 | 46.810 | 0.052785 |
| response_latch_delay100ms | 0/10 | 141.355 | 0.227819 |
| response_latch_noisy | 8/10 | 4.278 | 0.007769 |
| response_latch_delay50ms_noisy | 3/10 | 42.997 | 0.045566 |

## 完整轨迹与终态保持

| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |
| --- | ---: | ---: | ---: |
| native_measured | 10/10 | 5.045 | 0.016961 |
| affine_position | 2/10 | 23.622 | 0.036957 |
| affine_intent | 10/10 | 23.594 | 0.036863 |
| response_position | 0/10 | 3.934 | 0.008782 |
| response_intent | 9/10 | 3.944 | 0.008782 |
| affine_latch | 9/10 | 23.591 | 0.036895 |
| response_latch | 9/10 | 4.054 | 0.008838 |

| 条件 | 最终成功 | 末1秒成功比例均值 |
| --- | ---: | ---: |
| native_measured | 10/10 | 1.000 |
| affine_position | 2/10 | 0.200 |
| affine_intent | 10/10 | 1.000 |
| response_position | 0/10 | 0.000 |
| response_intent | 9/10 | 0.900 |
| affine_latch | 6/10 | 0.540 |
| response_latch | 3/10 | 0.300 |

## 发布R2状态参考

| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |
| --- | ---: | ---: | ---: |
| native_measured | 14/14 | 11.073 | 0.034402 |
| affine_position | 3/14 | 24.919 | 0.032738 |
| affine_intent | 12/14 | 28.410 | 0.035044 |
| response_position | 1/14 | 4.270 | 0.008741 |
| response_intent | 14/14 | 4.844 | 0.008385 |
| affine_latch | 13/14 | 28.625 | 0.035697 |
| response_latch | 12/14 | 4.719 | 0.008240 |

## 发布R2参考终态保持

| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |
| --- | ---: | ---: | ---: |
| affine_intent | 12/14 | 23.841 | 0.034466 |
| response_intent | 14/14 | 4.257 | 0.008726 |

| 条件 | 最终成功 | 末1秒成功比例均值 |
| --- | ---: | ---: |
| affine_intent | 12/14 | 0.857 |
| response_intent | 14/14 | 1.000 |

## 8800固定模型参考前向回放

| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |
| --- | ---: | ---: | ---: |
| affine_position | 0/5 | 36.208 | 1.899606 |
| affine_intent | 0/5 | 63.101 | 1.914419 |
| response_position | 0/5 | 255.483 | 1.883577 |
| response_intent | 0/5 | 216.068 | 1.841047 |

## 8800实时图像闭环

| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |
| --- | ---: | ---: | ---: |
| baseline | 0/5 | 37.093 | 1.899606 |
| compensated | 0/5 | 42.116 | 1.914515 |

HDF5 oracle使用obs[t]作为actions[t]执行后的目标，保持原world frame；native正控与native-gripper诊断使用专家命令，其余候选不使用专家命令生成或发送控制动作。
发布参考从当前R2 state0开始接入，初态仍为固定HDF5场景，不声称恢复了R1隐藏控制状态。两种来源观测不能互换。
保存模型目标的回放使用已知外生参考和当前实际反馈；它包含原分支未来图像条件下生成的目标，不能冒充当前分支的实时策略。

RMSE为各episode完整执行区间的均值，提前成功会改变长度；JSON另列相同参考、相同步数前缀的arm对照。
oracle误差点为OSC source pose：grip site位置、right-hand body方向，均在world frame；实时模型表沿用canonical TCP位置误差，两者不混合汇总。

五个已见episode属于诊断回归；额外十条在数值评测前冻结，属于同任务adapter episode对照，不能声称世界模型未见任务泛化。

模型两分支在首chunk后按各自真实图像重新预测，后续参考允许分叉；仅首chunk可视为相同模型目标。

原生初始化缓存、夹爪控制语义与模型输出质量的证据分别见实施记录和结论。完整trace与指纹位于outputs/。

[对照图](../outputs/libero_audit_20261019/figures/oracle_control_comparison.png)
[夹爪与物体轨迹](../outputs/libero_audit_20261019/figures/gripper_contact_diagnostic.png)
