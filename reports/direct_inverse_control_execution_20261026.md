# 213 直接逆控制接入和首轮执行

固定210唯一拟合，211平移运行层接入新执行器`src/s2a/direct_inverse_control_simulation.py`。原18profiles/4907周期、seed42、初态、参考时钟、实际控制器和083/159旋转、079夹爪保持。直接逆采用完整desired SITE姿态输入，平移限制为实际上一条命令±.25/20Hz及实际接口范围，不再混合旧前向逆、支持权重或5mm预测限速。

命令形成后，实际action与仿射goal_delta一致写入日志；173 SOURCE残差commit使用最终执行命令；150 SITE创新仅在env.step完成后更新。前向物理nominal只用于创新和明确标注的物理基线诊断，不将直接逆描述为前向状态预测。新独立checker按独立输入、解析逆、标量特征与系数乘积及裁剪等式审查，原KKT审查不适用于本候选。

`outputs/direct_inverse_control_binding_20261026/independent_verified.json`已通过：在205首条164个记录状态上重新生成本候选命令与完成后的历史，独立回放核对全部164行；命令、physical_delta、残差预测、限幅下界、previous command、两处创新及物理nominal共8类损坏均拒绝。这仍是记录状态上的命令/历史核验，physics_steps=0，不能当反事实控制证据。

该proof绑定初步配置`configs/direct_inverse_control_20261026.json`；其保留旧objective/residual文字，不作正式物理执行配置。正式配置另存`configs/direct_inverse_control_registered_20261026.json`，以明确的直接逆目标和150物理创新语义取代旧文字，并绑定该proof；模型参数、profiles和执行策略相同，未改写已绑定初步配置或proof。

新执行namespace为`outputs/direct_inverse_control_20261026`。准入20个检查全部通过：117项CPU测试、18实际CLI dry-run、独立checker导入；新入口/执行器/checker Ruff检查及格式检查通过。sources清单绑定所有当前src/scripts、所用测试、配置、计划和资格证据。CPU和OSMesa均显式空CUDA/LD，无新模型调用，无新拟合，无fresh任务。

已启动并实查PID/start_ticks：supervisor 276276/588661065、owner 276403/588661663、首条采集worker 276452/588662081身份一致存活。首条正在物理执行，此处不预填性能结果。每条采集及独立checker均由owner实际wait/reap，整批36stage、owner收尾、来源复核和原18/18性能资格仍需全部完成。后续汇总见同namespace的case独立结果、execution_result及closure；未产生的文件不作为通过证据。

WAMdev及最终24新初态继续等待原控制资格。直接逆动作误差下降与入口测试通过尚不能证明总目标完成。

首条已完成独立机械/标签/图像检查，采集与checker均正常wait0/reap，但性能失败：任务从未成功，位置RMSE .27038684m、SO3 RMSE .05180290rad。case01随后也完成独立检查，任务未稳定，位置RMSE .23034896m、SO3 RMSE .02583731rad。不能采用本候选进入WAMdev/fresh；原整批继续执行，不改冻结参数、不覆盖失败。

214对case00全部164行独立重算解析基线、标量特征和残差，证据`outputs/direct_inverse_departure_20261026/verified.json`。step48位置误差首次超过5mm，随后恢复（step60仅.00037996m），step67首次超过1cm；step70残差首次某轴超过1，同时出现已执行命令与饱和解析基线该轴反向；step71/74误差依次超过5/10cm。step80解析基线[-1,-1,-1]而最终命令[1,1,1]，残差[6.5551,5.6037,9.1039]，其中desired motion线性部分[5.6483,5.1268,8.4093]，Fourier部分仅[.0163,.0191,.0043]。此时不是独立重算与实现不一致。

这些是已执行过程中的数值事实，并不唯一证明最初接触失败的物理原因。但它揭示当前“先裁剪解析基线，再加无界线性残差”的结构允许目标偏差增大后，残差抵消甚至反转解析纠偏。后续应先检查对desired position的净增益与饱和顺序，再预登记修正结构；不能以扩大样本或动作RMSE再降低作为闭环修复证据。
