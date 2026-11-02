# 8400 guidance3 state 诊断 — 2026-10-21

**结论：固定的 guidance3 候选未通过预注册的 state 门槛，不进入额外物理评测。** H8/H16位置只改善2.338%/0.137%，两个时间段的SO(3)与夹爪开度误差均回退；H40三项也均回退。这一轮没有提供用更强文本引导替换guidance1的证据。结论仅覆盖本次固定值3.0，不外推到其他guidance值，也不把state误差直接等同于任务成功率。

[012计划](../plans/v0.0.1/012_guidance_state_20261021.md)与[配置](../configs/wam_stability_screen_8400_g30_20261021.json)在`17844f2`、任何新请求之前冻结。保持旧22-source的8400非EMA checkpoint、十个有效槽mask、35采样步、shift5、H40/20Hz、task-only条件及原24个diagnostic phase样本、6条旧episode、22个不同anchor、seed42/43。仅推理guidance改为3.0；资源位置改为物理GPU7/9577，输出写入独立目录。新增48个模型请求、0物理步；旧guidance1的48个缓存请求没有重跑。未打开fresh初态。

| 预测前缀 | 指标 | 缓存guidance1 | 新guidance3 | 固定门槛中的表现 |
| --- | --- | ---: | ---: | --- |
| H8 | 位置RMSE / mm | 24.383877 | 23.813740 | 改善2.338%，不足10% |
| H8 | SO(3) RMSE / rad | 0.088644 | 0.184220 | 回退 |
| H8 | opening MAE | 0.058538 | 0.072470 | 回退 |
| H16 | 位置RMSE / mm | 44.622633 | 44.561478 | 改善0.137%，不足10% |
| H16 | SO(3) RMSE / rad | 0.177223 | 0.313873 | 回退 |
| H16 | opening MAE | 0.086038 | 0.093883 | 回退 |
| H40 | 位置RMSE / mm | 85.521032 | 91.675223 | 报告项；回退 |
| H40 | SO(3) RMSE / rad | 0.479894 | 0.558012 | 报告项；回退 |
| H40 | opening MAE | 0.114660 | 0.119576 | 报告项；回退 |

H8/H16/H40分别汇总384/768/1778个有效端点，未把不足H40的未来填充参与评分。保持当前状态与历史外推基线、逐任务/phase及逐episode结果均保存在完整审计文件。重复phase与两个采样seed不作为独立episode；这些样本用于开发，不能声称训练未见或独立最终验收。

门槛要求H8/H16的六项全部不回退，各项SI数值比较只容`1e-10`绝对误差，并至少一个前缀的位置RMSE满足`candidate <= 0.9 * cached_guidance1`。实际有四项回退，且两个位置改善均不足10%，因此两部分条件都失败。门槛实现用直接比例比较，避免`1-candidate/baseline`在精确10%边界的减法舍入；精确10%通过、略低于10%拒绝、单项回退拒绝及数值容差边界已进行CPU验证。满足门槛本身也不会自动授权物理评测。

[新审计入口](../scripts/audit_wam_guidance_state.py)复用冻结的`verify_cell`逐请求核对观测/条件/源代码、checkpoint、mask/反归一化、完整40×128响应、有效未来和状态误差；再用不导入生产几何/指标函数的`audit_metrics`独立进行Gram–Schmidt解码、局部delta积分及SO(3)误差重算。新48和旧缓存48均复核，缓存聚合逐值精确等于原summary。新数据独立重算最大端点差为位置`2.22e-16`m、旋转`6.66e-16`rad、开度0。358个输入文件SHA在审计前后核验；原163个完整权重字节指纹通过path/size/mtime复核，本次没有重新读取32GB权重计算SHA。准备时30个唯一PNG/NPZ逐SHA核验，旧g1产物与summary未修改。

模型服务PID173138/startticks167654394显式`CUDA_VISIBLE_DEVICES=7`、空LD；CPU coordinator173067/startticks167654358及state worker174201/startticks167658396均为空CUDA/LD。三个已登记进程均已退出，模型返回-15、worker返回0，未记录执行错误。审计没有启动任何模型或仿真。

本入口没有记录实际initial-noise witness。因此seed相同是固定请求条件，不能声称已证明逐值相同初始noise的单变量实验；011的noise-paired采样研究独立保留。本轮不新增控制补偿、训练、参考校正或任务规划，也不改变010的固定物理预算。

可复核产物：

- [完整48审计](../outputs/wam_stability_20261021/guidance3_audit/complete48_verified.json)：`5967e9a75918505eeefc95468b034f8ec5ffad72c18b185c245949721745662d`，`complete=true`、`frozen_admission.passed=false`。
- [CPU准备与预算/代码review](../outputs/wam_stability_20261021/guidance3_preflight/prepared_cpu_review.json)：`2fab8ce9df690bbb6c95c0e1725caf4d3c59d5005523e5558ee9b7115f44fb5f`。
- [运行时进程身份witness](../outputs/wam_stability_20261021/guidance3_preflight/owned_process_identity_snapshot.json)：`522cdf05dff2d0ec484116d046ee03c414691b6ff6e98299e285176b35b62e96`。
- [缓存g1桥接独立验证](../outputs/wam_stability_20261021/guidance3_preflight/cached_g1_bridge_verification.json)：`bf8b4590a4b8752eec06e7fe8a483df91c79ca1f0bed7b7dbe82886c8a70c9fc`。

后续保持guidance1的010实验完整运行与独立验收；013如按自身注册开展，只检验其独立backend假设，不将012失败转换为追加guidance3物理预算。
