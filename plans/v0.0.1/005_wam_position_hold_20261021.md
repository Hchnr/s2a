# 连续开度映射下的WAM任务对照 — 2026-10-21（Asia/Shanghai）

状态：条件预算冻结，须004全部资格通过及独立源/命令审计通过后才启动模型。本计划不将实现完成、oracle成功或暂时task成功当稳定成功。

003的24条WAM完整独立核验：raw alpha1共12条均未成功，alpha.5有8800 bowl两个非连续成功端点，alpha0有15400 bowl连续11.4秒成功，最终terminal/stable均0/24。原intent_hold把高开度小下降直接解释成close-to-limit，会在reference=.922时令actual=.148，因此还不能把全部早闭合/后续观测偏离唯一归于WAM。004采用独立模块修正这一映射，不按物体或task控制。

## 固定范围和对照

沿用旧8800与新23源15400、guidance1.0，各旧task0/1/2 init0、sampling42，全部三任务都保留。权重与mask服务合同不变；同WAM首个query的输入图、完整H40×128输出须与各自003对应case精确相同。后续闭环观测会分叉，不称全程相同输入单因素实验。

每checkpoint四方式各3场景，总24条/14400物理步、最多360个H40模型请求：

| 方式 | 夹爪 | 参考治理 | 模型旋转 |
| --- | --- | --- | --- |
| position_hold | 004固定新映射 | 无 | alpha1 |
| recovering_position_hold | 同上 | 恢复版共同参考时钟 | alpha1 |
| recovering_position_hold_alpha05 | 同上 | 同上 | alpha.5 |
| recovering_position_hold_alpha0 | 同上 | 同上 | alpha0，每chunk保持当前观测anchor朝向 |

与003同H40/执行40、20Hz、最长80物理步重新观测、600实际步、末40连续原生成功；35采样步、shift5、response逆inverse_weight=.5/rate5不变。无governor在40端点耗尽时重新预测；有governor在参考完成或80步时重新预测。没有因成功/失败事后缩短或延长预算。

alpha.5/0改变生成参考，不是纯S2A补偿；alpha0不验收完整WAM朝向。raw/effective/admitted/nominal参考、模型时间消费、bridge/pause/未执行后缀和实际电机目标分开记录。contact/对象/成功只作执行后指标；WAM分支配置不提供oracle_reference_root，控制不读取专家未来或task标签。

## 启动与完成门禁

004的旧v2/no-gov、新v3/no-gov、新v3/recovering共9条/5400步必须完整通过：各3/3稳定、TCP≤10mm/SO3≤.05rad、旧v2三条物理序列精确复现、独立motor/命令/时间/来源核验。配置/源码/真实依赖SHA在004开始时冻结，WAM复用同一position执行函数；资格失败保留并停止本计划启动，不改阈值重跑。

只在物理GPU5/port9345与7/port9347各启动自有单卡模型，确认原003服务complete/release且空闲；CPU编排/OSMesa CUDA与LD_LIBRARY_PATH为空。root启动记录绑定完整资格artifact SHA与全部输入SHA、权重完整指纹、配置SHA、PID/startticks。已有8400在GPU4的不同阶段对照保持原字节跑完，GPU6无必要不占用。

24条完整后独立审计并报告全矩阵，不按成功病例挑task/seed拼接。若新映射已有稳定开发能力，再固定候选及两seed的新初态验收；新初态来源排除尚有未解，不能提前宣称独立。当前计划只追加这24条开发，不自动启动新初态验收、训练或其他预算。
