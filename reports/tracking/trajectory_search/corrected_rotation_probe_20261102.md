# 因果安全模型修正后的新旋转方向

2026-11-02。预登记实现提交373abe1；全程CPU，GPU禁用。

## 结果

QP及完整独立审核通过，但新候选仍不安全，不下发、不授予控制资格。
旧方向安全样本的一次因果秩一修正不足以保证新优化方向安全。完整24caps仍失败，
总目标与原18/12/24及稳定20/24、每task4/8验收保持，最终留出未打开。

| 指标 | anchor / repeat | 新候选 |
| --- | ---: | ---: |
| 原joint audit | True | False |
| 最小raw safety slack | +9.9806666065e-6 | -5.5023169425e-6 |
| 最紧行 | 4346 | 4358 |
| 通过caps | 12/24 | 13/24 |
| 完整合格prefix | 2 | 2 |
| 原固定cap loss | 2.0100673083 | 2.0486008319 |
| 第12点旋转误差rad | .1311260695 | .1281591879 |
| 完整witness | False | False |

修正模型对row4358预测+4.9953333035e-6，实际-5.5023169425e-6；
全安全行最大预测误差1.4200329081e-5。新方向实际step inf约.005，不能将
旧方向约1e-4样本上的拟合精度直接推广到它。当前数据仍无法区分新方向的
一阶失配与有限步曲率；需要新方向的固定小尺度完整响应才能判别。

## 实现与实际费用

入口 `scripts/tracking/trajectory_search/probe_corrected_rotation.py` 使用原stage58/
anchor85模型与局部性第5号安全响应，仅更新Js，Je/Jq及原全部硬约束不变。
更新矩阵与独立审核实现逐值一致。目标仍为原最后旋转cap-distance方向，
原.01m/.05rad caps及安全余量、QP证书、独立容差均保持。

QP前保存矩阵、输入及预算登记；实际1QP/3迭代/1NNLS，原certificate、objective
scaling和dual recovery审核通过。alpha1唯一候选通过原box/rate/global/trust及
非重复预检，没有裁剪、补选、自动重试或新增搜索阶段。

候选固定后共享runtime内部0QP/NNLS；整体必须加外部qp_result记录，合计仍为
1QP/1NNLS，不能将runtime的局部计数说成整体0QP。
生产3响应/900内步/937查询；独立3响应/900内步/937查询，合计6/1800/1874、2model。
全部完整响应逐值一致，anchor/repeat exact，源码/来源前后不变。0物理/GPU/
WAM/fresh/最终留出/实际参考消费；不是独立物理仿真或任务结果。

工具session28119退出0，timeout owner已wait/reap，child PID179251确认不存在，
600秒TERM/10秒KILL上限未触发。未复用或覆盖旧失败产物。

## 验证与产物

统一CPU runner jobs32，46项实际通过、0skip、0collection错误，worker回收，
源码前后不变。包含新安全拟合的因果性、非安全/过大/零步/非有限/不完整输入
拒绝，以及原旋转、局部性、cap objective与audit、search driver专项。
Ruff、format和Git空白检查通过；未运行全仓回归。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 完整结果 | corrected_rotation_probe_20261102/result.json | 177b188ec11af38829466466db9ef768263c6d40ef7c32a0ecc7f9f9bc2351f6 |
| QP与独立证书 | corrected_rotation_probe_20261102/qp_result.json | 607462e7469bf5d6f306d0aa44aed16540e4916a52ad6a7eb5dd30c31999519a |
| 保存响应分析 | corrected_rotation_probe_20261102/saved_analysis.json | 98fa5adf4f2dce45e6998225612ac7c852b341c71a2d2881f3bc1dd8372dd3e5 |
| 工具终态观察 | corrected_rotation_probe_owner_20261102/observation.json | 10a8f76f9ef04ed19a819168b031952c5e323def494f55bf1768253a8fe6b5a0 |
| 46项专项 | qa/corrected_rotation_probe_20261102/result.json | 702e2bd12449d8ea8f3ef6266e1bf7f99f4003f59b92c4b569384cbcb5eddd76 |

下一步先独立登记新方向的固定缩步/小尺度探针，判别修正后新方向是否仍有一阶
失配，或主要受大步误差影响；不能反复优化后按安全结果补选方向，也不能将
旧方向误差下降当作新方向安全证据。未证明不可达，不放宽原目标与安全门。
