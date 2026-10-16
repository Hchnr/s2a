# LIBERO S2A 独立仿真对照

完成状态：27/37；进行中，未执行项不计为通过。

Checkpoint：`iter_000008800`，同一warm-start run；历史5600权重已清理，旧0/5不能直接作为新权重的baseline。

任务：put the bowl on the plate；5个冻结初始状态，seed42+trial，20Hz，chunk16/execute8，35采样步，guidance1.5、shift5、无warmup/EMA，模型闭环最多1000步。

| 输入 / 证据 | 转换 | 完成 | 任务成功 | 平均位置RMSE (m) | 平均旋转RMSE (rad) | 平均夹爪opening RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| probe | baseline | 1 | 不适用 | 0.001845 | 0.008919 | 0.015370 |
| probe | compensated | 1 | 不适用 | 0.001576 | 0.006084 | 0.015216 |
| native | baseline | 5 | 4/5 | 0.012000 | 0.011078 | 0.073517 |
| oracle | baseline | 5 | 1/5 | 0.026210 | 0.031406 | 0.077312 |
| oracle | compensated | 5 | 1/5 | 0.004447 | 0.007593 | 0.062653 |
| prediction_replay | baseline | 5 | 0/5 | 0.039077 | 1.902458 | 0.122909 |
| prediction_replay | compensated | 5 | 0/5 | 0.126066 | 1.807854 | 0.113670 |

baseline沿用TCP局部delta积分、TCP→source转换、OSC affine inverse及原夹爪position目标。新增补偿使用固定内环PD响应的二阶解析近似、当前实测速度、0.5逆权重及最终每轴5/s命令变化率限制；旋转误差超过0.25rad时关闭旋转逆近似。原OSC增益、物理模型和夹爪控制路径不变。

probe和oracle输入来自独立运动请求或专家状态。prediction_replay使用旧5600预测且不重新查询图像。policy才是8800权重的实时图像闭环；分支执行不同后，后续预测允许不同。所有结果均为仿真，没有真机执行。

表中是各episode完整执行期的RMSE均值；成功提前停止可改变样本长度。JSON另存两候选相同步数前缀指标，oracle/固定预测逐点核对相同目标。policy误差衡量各自模型目标的跟踪，不能把下降单独解释为模型预测更正确。

5个历史episode属于训练可见数据，一个任务的回归试验不代表泛化。解析内环模型未包含接触、力矩限幅和完整非线性；状态跟踪改善与任务成功分别判断。原生expert正控如果未成功也如实计入，不排除样本。

两候选的夹爪均沿用历史评测的aperture→原position actuator转换，本轮补偿仅作用于机械臂；native正控执行原始gripper action，不覆盖为aperture目标。补偿读取当前仿真control site的线速度/角速度，反馈没有额外估计噪声。历史baseline最初8步的发送命令/位置已与原记录逐项比对；长接触轨迹不要求浮点位级复现。

policy采用与历史相同的同步仿真：模型推理期间物理时钟暂停，未模拟推理延迟，不代表实时部署性能。初始chunk的专家标签仅由独立离线诊断脚本读取，未进入正在运行的策略或用于重新调参。

完整请求、物理前后状态、目标、原始/发送命令、控制器参数、视频与源码/数据指纹见 `outputs/libero_simulation/`。
