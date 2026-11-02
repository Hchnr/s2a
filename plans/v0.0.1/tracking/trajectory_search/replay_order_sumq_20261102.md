# 同预测器sum-q对照（2026-11-02，Asia/Shanghai）

采用独立验证的新replay-order响应、native_solve TCP评分、新初始化/72探针，
仅建立原sum-q目标基线；尚不实现cap-distance，不同时改变两个目标作归因。
初态368、参考、控制接口、时间、安全caps、坐标系、在线因果输入保持。
旧运行、配置、失败记录和预算不改。GPU禁用，CPU数值库单线程，CUDA/LD空。

前置结果 outputs/replay_order_stencil_20261102/result.json SHA256
80e1fd703fe67a8d44ad90d5b603e86e1af7d9fba50084dc6009ea71157792c5；
新stencil SHA256 0d447082a50a2a8425c438a0a7cfe8a4a9742cb53bba445e015e49db6f6ecc3c。
必须先核对passed、全部响应和矩阵exact、来源不变和全部32进程回收；新模型
从保存的新base/probes重建并与新矩阵核对。禁止读取旧矩阵充当新初始化。

算法调用未改的state_vector_dual_recovery.search，使用相同QP原单位证书、
原NNLS子集/迭代预算、restore/pose状态机、反馈及拒绝/缩域处理。原完整配置
逐字段保持，pose目标为0.5*sum(q)。本轮最多60阶段/60QP/6000QP迭代、122生产
响应（含原始重复），最多一次NNLS调用/每阶段，迭代上限仍3倍原full-active数。
无补跑/自动重试，无额外FD探针。另独立复算全部实际响应（最多122），重建
完整KKT/反馈/严格fresh决策账本；原numeric1e-10、torque1e-8、布尔/类别门保留。

新预算最多244响应/73200内步/76130查询/2model。独立checker即使生产无witness
也应审核全部已完成响应。0物理、WAM、新场景/最终留出、GPU；不追加旧预算。
先写registration和代码/输入SHA，独占outputs/replay_order_sumq_20261102，运行
失败保留False及实际成本。任一审核失败都不能记为完整通过或控制资格。

比较初始/最终anchor的总q、前9/后3分块、最大q、caps数量、prefix、安全reserve、
QP证书、成本和完整witness。无论sum-q是否改善都不代表闭环任务成功。
通过后才允许另行登记cap-distance同预测器对照；真实物理pilot仍须完整执行
轨迹、独立安全审核及命令回放准入。原18/12/24及最终20/24、各任务4/8不变。
