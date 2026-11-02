# 守护QP后的原完整WAM开发矩阵

总目标与原最终留出门槛保持。局部闭合提前、单块连续执行和.01正关节buffer均未产生抓取，不把这些实验作为新生产策略。本版本只把已通过完整原18控制资格的守护QP入口接入原tracking事务，保持.05关节buffer与原夹爪逻辑。

完整12开发案例：三任务×seed42/43×raw/terminal058，每条600步、每8步以当前观测生成40参考。固定一次模型初始化、900请求、31500去噪/头调用、7200物理周期，fresh0，无自动重试。terminal至少5/6且各任务至少1/2稳定，原最后40步成功条件保持；未通过则不授权最终留出。

新包为tracking_preview_wam_*_v4，全部新根 `outputs/tracking_preview_wam_development_v4_20261027`，不修改或覆盖v3失败记录。实际模型仅允许UUID GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f；端口31375/31376，CPU CUDA/LD显式空。模型启动前需要真实12输入预检、组件QA、源码/数据冻结、独立metadata、14实际CLI dry及独立执行登记全部完成。

独立验收器使用正确31375端点。实际执行前argv/配置SHA、prepared源码指纹和运行manifest共同绑定；若生产未输出单独execution_snapshot，不捏造该产物，明确记录实际具备的来源证据。该版本保留全900请求、全原生控制日志、独立数值与task指标审查，不用仅metadata或局部诊断替代全量。

入口准备 `scripts/prepare_tracking_preview_wam_development_v4.py`，独立metadata `scripts/verify_tracking_preview_wam_metadata_revision4.py`。生产前所有来源通过新 `dependency_safe_wam_admission.py` 绑定完整18/36项修复资格和实际恢复事务。旧18资格不可替代新修复入口资格。组件QA和CPU输入预检完成时仍没有模型/物理调用；执行授权与是否已实际执行分开记录。
