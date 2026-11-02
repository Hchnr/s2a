# dual recovery：严格 fresh 决策失败的独立上下文复算

本计划在新的真实复算之前登记，起点代码为`f19fe7c`。目录整理与分轮CPU验收
已提交；总控制目标、原18/12/24验收和GPU禁令保持。上一目标轮属于有验证的
实际进展。本轮不重跑producer，不增加或延长原搜索预算。

## 已有事实与问题

唯一producer `outputs/tracking_v5_state_vector_dual_recovery_20261029/verified.json`
SHA256 `436fbe20f20cf6678aa329c05e928845b563f76cb0a976c11395e16481aba551`，其
registration SHA为`039029faf7b34d34ba3c7f679da0a5a638f6af6c32d1a3738f279917d752e797`。
原checker verified SHA为
`6036f0cd4a711ef44b556b489cf946252d2d30e8dafb8cb232cfef0c4d4fcb85`，仍是
complete=False/passed=False；虽然122响应、36600数值内步/38065查询和全KKT账本
已经完成，fresh_ratio误差`2.137171550842254e-10 > 1e-10`使最终严格门失败。
原owner complete=True/exit1不等于checker complete=True。

现有失败JSON没有首个失败stage/response或逐候选fresh指标。只读保存数据诊断
`outputs/fresh_ratio_contract_review_20261029/report.md`未能证明工具矩阵舍入差
足以解释该错误，不能据此修改输入或放宽容差。

## 实现与准入

1. 保留原八个native函数完整AST：query、reconstruct、compare_response、
   reconstruct_trajectory、compare_rollout、command_groups、classify、check_audit。
   数学、native工具函数、MuJoCo/NumPy/Numba版本、2ms时域、TCP和输入保持。
2. 仅适配实际初始化的历史资产证明及AST helper读取：旧声明/SHA用catalog和
   Git字节核验，当前源码另核批准清单；helper完整模块仅允许已登记namespace
   迁移。禁止旧根访问、平铺wrapper、全局hook和执行历史源码。
3. 原fresh_actual_reduction/fresh_ratio比较仍在原位置以原`1e-10`进行，保持
   严格接纳优先级与半径门。在比较前保留stage/anchor/response/alpha、mode、
   两侧merit、分子/分母/ratio及差、reserve最小行、各门输入和比较进度。
   诊断只观察既有fresh指标；不得用saved ratio替代fresh或吞掉异常。
4. 保存全部122份fresh classification指标（4369条安全slack、12×2 pose分数及
   原flags），便于不再调用native模型的后续定位。含末次initial repeat，不去重。
5. 新入口与新config绑定本计划、原producer/registration、初次失败owner及v2
   失败owner/checker；旧False、原数据和原SHA前后保持。先完成有意义的CPU
   拒绝边界测试、原相关模块回归、真实新入口dry，验证后提交实现再启动。

## 一次隔离复算

新checker独占`outputs/tracking_v5_state_vector_dual_recovery_fresh_diagnostic_20261029/`，
新owner独占`outputs/state_vector_dual_recovery_fresh_diagnostic_owner_20261029/`；
不得覆盖任何已有路径或自动重试。owner显式空CUDA/LD、数值库单线程、seed42，
记录实际命令、代码/数据指纹、真实耗时、exit及wait/reap，1800s上限。

仅复算原122条完整响应，预期36600数值内步/38065查询、原121唯一命令；新增
QP/NNLS/物理步/WAM调用/fresh case/实际参考消费均0。不运行仿真积分或模型服务，
不打开最终留出。仍按原solve构造工具输入，本轮不并行引入SE(3)规范化候选。

任何加载、数值或决策门失败都保留False与已发生计数、上下文；成功也只说明这次
固定保存路线通过独立数值核验，不授予实际控制、完整H12或任务稳定资格。后续
动作根据首个可归因差异决定，不把容差内聚合误差当成完整轨迹成功。
