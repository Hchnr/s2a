# 单块轨迹下正关节buffer的物理对照

沿用原总目标与原18/WAM/fresh完整门槛。原单块40步未抓取且朝向误差.1545rad，局部模型误差与真实相近；q7持续接近.05rad缓冲。固定实测状态的.01rad候选降低模型目标，但晚段仍不达标，不把它当作已验证修复。

独立版本将主要interior规划buffer设.01rad，真实joint range、动作幅值/变化率、2ms内层与终端非外向约束、目标/误差门槛保持。原adaptive/soft数值后备明确保留，不能把后备当成同一硬保证。当前配置只用于该CPU原生探针，不进入已资格生产策略。

两组各176周期（相同原136前缀+40续跑），总352。基线精确重放原176动作，候选在step136沿原已生成块的40参考持续闭环跟踪，不读之后生成块；所有独立夹爪/控制历史从前缀实际执行重建，初态seed42，WAM/fresh0、无自动重试。对比既有.05候选时保持完整共同前缀与已生成目标；分别记录实际关节margin、超界端点、TCP误差、真实grasp/task，禁止把局部接触替代600步稳定任务。

入口 `scripts/execute_committed_chunk_buffer01.py`；实际执行前源/预算登记位于 `outputs/committed_chunk_buffer01_execution_20261027/registration.json` 与 `outputs/committed_generated_chunk_buffer01_20261027/registration.json`。独立回放入口 `scripts/execute_committed_chunk_buffer01_check.py` 只在生产完整结束后运行，追加176物理周期，核对所有177状态/实际任务及抓取谓词/controller goal/TCP误差/关节margin。该回放不计作新的任务试验。
