# 086 固定native前缀的成对arm命令响应采集

085尚在完整收尾：加性残差预测在失败轨迹依旧改善，但控制/任务退化，说明预测误差不能替代反解增益与耦合的资格。下一步不挑新的补偿系数，先采集实际命令通道的一周期成对响应。必须085完整关闭后才执行新物理。

固定既有fit轨迹task0/demo8、task1/demo1、task2/demo2（沿用078的三条来源，不使用6条内部holdout或最终新初态）。每条取第一次native close命令前的实测端点和close后第一次native open命令前的端点，各两个anchor；周期分别33/78、76/157、42/85。每anchor重新从原初态回放完整未改native前缀，再执行1周期：原native命令baseline，以及仅修改translation命令x/y/z分别±0.1的6个分支。实际归一化命令裁剪在原[-1,1]内，记录裁剪后的真实差值；旋转和夹爪保持同周期native命令不变。42条总3339个物理周期，控制20Hz，每周期25个子步；所有前缀成本都计入。

三fit轨迹的实际完整state、motor cache、controller goal及图像/标签沿用独立核验协议。每anchor先完成baseline并对其前缀和一步端点与原native轨迹逐位/原声明容差核验，再允许该anchor六个探针。每个探针都独立回放，不通过不完整恢复的hidden controller状态分叉；原native reset motor cache保持0，不能由实测q重置。原失败、执行/审查退出凭据保留，不自动重试。

记录当前实测source pose/twist、当前关节状态、上一实际命令和当前实际command，下一实际pose/twist、controller goal/motor cache；对象状态只作离线指标。独立机械核验实际发送差值、各通道范围、固定旋转/夹爪、完整前缀与原native、时间/状态标签及实际wait/reap。不能把native专家命令写成已由state反解得到。

baseline和±分支共同估计每归一化translation输入的下一位移、下一线速度及SO3旋转差异，分别报告正/负单侧与双侧差分，保留接触/非线性差异，不能仅报告一个平均增益。SO3差异使用world坐标Log(R_branch R_baseline^T)，不对Euler角相减；物理delta比例从实际固定OSC接口获得。与名义PD响应和零交叉耦合假设比较，报告证据范围；局部差分不是全局动力学模型，也不证明闭环稳定。

本阶段不fit、不构建新控制候选、不跑WAM/fresh；采集和审查完成后才另登记模型/控制方案及划分。最终总目标、原18/18正确参考门槛、WAM开发门及24条留出新初态门槛不变。当前仅完成计划与显式配置，采集、独立审查和物理尚未实现/执行。

实现补充：paired_arm_response.py单周期/单translation通道命令及裁剪已实现，8项约束/源不变检查通过；collect_paired_arm_response.py保留078原生前缀协议，增加实测site twist及anchor/下一端点图像，首baseline实际3.11 CLI dry通过。XMLSiteKinematics独立从compiled XML和time/qpos/qvel重建site pose/twist，3项几何/索引检查及085三case的9个真实端点采样核验通过（outputs/xml_site_kinematics_actual_20261025/result.json）。verify_paired_arm_response.py已接入完整前缀、实际差值、motor、XML运动学、标签及图像审查，Ruff/导入通过。持久化owner/全42条准入/完整物理及新数据审查仍未完成，不能据9端点采样授予新物理资格。

持久化执行补充：run_paired_arm_response.py实现42条采集/独立核验顺序及owner实际wait监督；启动前独立核对3来源属于旧24fit、native首close/open周期、每组baseline/±3轴分支、全部预算3339步及085完整关闭。collect/verify固定归一化接口±1并仅改目标轴。close_paired_arm_response.py要求84个stage及owner实际wait/reap、6个baseline、42条XML速度/图像/标签/来源核验，仅授予data资格。代码/导入与Ruff检查通过，尚无086物理性能或新模型资格；全量准入通过后才启动。
