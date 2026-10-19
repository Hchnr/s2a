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

## 8800实时图像闭环

| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |
| --- | ---: | ---: | ---: |

oracle使用obs[t]作为actions[t]执行后的目标，保持原world frame；native正控与native-gripper诊断使用专家命令，其余候选不读取专家命令。

RMSE为各episode完整执行区间的均值，提前成功会改变长度；JSON另列相同参考、相同步数前缀的arm对照。

五个已见episode属于诊断回归；额外十条在数值评测前冻结，属于同任务adapter episode对照，不能声称世界模型未见任务泛化。

模型两分支在首chunk后按各自真实图像重新预测，后续参考允许分叉；仅首chunk可视为相同模型目标。

原生初始化缓存、夹爪控制语义与模型输出质量的证据分别见实施记录和结论。完整trace与指纹位于outputs/。

**实施中：实时模型矩阵尚未完成，不计为全部通过。**
