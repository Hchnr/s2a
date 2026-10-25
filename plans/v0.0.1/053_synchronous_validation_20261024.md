# 053：原六个validation demo的同步物理参考准备

目标沿用036。052已完成一次新CPU fit与独立全量QR，但没有新策略采样或任务结果。旧STATE的initial_sync_certified=False和逐端点物理时间戳缺失必须通过新证据解决，不回写旧证据。

固定同一六个validation demo：task0 demo16/32、task1 demo4/27、task2 demo14/24，禁止替换或按结果筛选。预检source_identity_preflight.json已证明选定HDF/初态身份和训练/fresh隔离；完整源SHA、状态0、XML、全部native动作还须绑定到执行preparation。不得使用这六条训练新head，也不读取fresh性能。

仅CPU/OSMesa、CUDA_VISIBLE_DEVICES和LD_LIBRARY_PATH显式空，seed42。每case严格恢复对应XML与states[0]，记录controller/gripper cache初始化和20Hz真实物理端点；沿用034固定控制器、工具/TCP、完整N条native前缀及最后controller SITE目标锁存、末native夹爪命令80步。N固定95/92/158/177/106/104，共732native+480hold=1212物理步、6次环境初始化，不早停、不重试。native正控失败完整保留，不换demo。

每条四个完整原生H40窗口anchor为j*(N-40)//3，j=0..3；24个当前端点各渲染现有合同的两相机和当前canvas，绑定同一sim state/time、实测TCP/opening、实际控制器状态、模型XML和真实前缀命令。保存全部后续40个真实端点与0.05s间隔；future RGB为0，不用历史R2未来图像或合成静止padding。末40hold端点原生predicate用于正控评分。

实现与验证后才能执行：先准备所有六个输入、runtime/code快照、独立来源审查和实际CLI dry；几何、时间、控制状态与来源约束必须可复核。新namespace outputs/synchronous_validation_20261024，不改034/051/052冻结来源或旧False结果。产物必须区分metadata/current-only模型输入与future-label archive，模型入口只能读当前PNG/实测状态及已生成参考。同步物理数据独立审查后，另行登记3 zero回归与完整96配对新采样或明确等价预算；不把metadata通过当成采样许可。

新参考与旧validation来源不同，不声称逐值复现旧STATE指标；保留raw与新头在同一24当前端点、seed42/43的完整对照。新state门槛、当前状态/图像causal合同及之后开发统一候选均在模型执行前冻结。最终原24条≥20/24且每任务≥4/8不变，fresh保持关闭。
