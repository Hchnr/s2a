# 023：WAM 生成参考的 44 参数局部校正

**结论：存在可校准的误差结构，但本方案未通过事前登记门槛，不推广到物理评测。** H8/H16 的共同状态分数分别改善 11.07%/14.36%，且都优于保持、历史匀速、无 WAM 和打乱参考对照；同时 H8 开度、H16/H40 位置退化，只有 4/6 个 episode 的两个短窗口同时不退化，要求为至少 5/6。不能只选朝向或共同分数的改善作为通过依据。

本项只读旧 `8400_masked`、guidance 1.0 缓存，采样 seed 42/43。六条已被查看的开发 episode，48 条请求去重为 44 query/22 anchor/4 alias。沿用 008/018 的整 episode 六折、固定跨 episode donor 和共同有效前缀；H8/H16/H40 共 352/704/1398 个端点。每折只用其余五条完整 episode 拟合 paired/noWAM/shuffle，共 18 组；第 19 组用全部六条和同一固定 1398 端点拟合，仅保存参数，不参与留出评分。0 新模型请求、0 物理步、0 fresh 初态、0 参数搜索。登记见 [023 计划](../plans/v0.0.1/023_generated_reference_calibration_20261021.md)。

校正输入只有当前 measured anchor、已生成的 world 参考和固定端点时间。以 `u=R0.T(p-p0)`、`v=Log(R0.T R)` 构造 7 维输入，拟合 7×6 的零残差先验 ridge；开度另拟合 2 参数，共 44 参数。输出为 `p0+R0(u+δu)`、`R0 Exp(v+δv)`，按端点时间限制平移/旋转/开度校正，拒绝近 π 分支。没有任务标签、物体、阶段、接触、成功判据或未来实际 state 进入部署校正。它改变 WAM 参考，属于生成参考干预；不是机器人响应反解，也不能把训练示范先验归为原 checkpoint 的任务能力。实现依据为 `src/s2a/generated_reference_calibration.py:35`、`:92`、`:148`，划分/输入依据为 `scripts/diagnose_generated_reference_calibration.py:85`、`:320`。

| 预测前缀 | 位置 RMSE，raw→paired（mm） | SO(3) RMSE，raw→paired（rad） | 开度 MAE，raw→paired |
| --- | ---: | ---: | ---: |
| H8 / 0.4s | 25.191→25.084 | 0.090896→0.072943 | **0.062439→0.062843** |
| H16 / 0.8s | **45.922→46.690** | 0.182583→0.134739 | 0.092479→0.091176 |
| H40 / 2.0s | **81.742→85.253** | 0.470659→0.321284 | 0.110965→0.110066 |

共同分数是同一组端点上 `mean((位置/10mm)^2+(SO3/.05rad)^2+(开度/.1)^2)`，不是将三个聚合指标简单相加。所有方法在相同评分边界 clip 生成开度；拟合特征/残差保持原始未 clip 开度。评分直接合并全部留出 query 的共同有效端点，按端点等权；fit 权重另按 episode、query、有效端点逐级等权并总和为 1。

| 前缀 | raw | paired | hold | 历史匀速 | noWAM | shuffle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H8 | 11.1183 | 9.8873 | 18.4176 | 11.5457 | 18.3710 | 53.3339 |
| H16 | 38.7501 | 33.1841 | 64.4586 | 65.0651 | 61.7360 | 104.6779 |
| H40 | 161.1196 | 119.1561 | 203.8209 | 344.5584 | 191.9950 | 269.3424 |

登记判定是联合条件，没有结果后容差。九项组件不退化中三项失败；两个短窗口的 ≥10% 改善与优于全部四对照均通过；整 episode 条件只过 4/6，episode 850、898 失败。因此 `eligible_for_further_validation=False`。paired 的朝向虽比 raw 改善，H8/H16 仍劣于 hold 和历史匀速；H40 的 0.321284rad 也略劣于 hold 的 0.318497rad，不能据此宣称朝向已可靠。分组件与共同分数应一起 review。判定依据为 `scripts/diagnose_generated_reference_calibration.py:261` 和独立 `scripts/audit_generated_reference_calibration.py:284`。

独立验证没有调用生产 feature/fit/apply/metric/eligibility，只复用原 008 独立读取器的缓存积分与固定 shuffle，使用 SciPy 重建 body 几何、增广加权最小二乘、逐端点 cap 和输出，重算 19 组参数、44 个留出 query 的七种方法、任务/episode/阶段统计及全部门槛。206 个输入 SHA 前后相同；参数重算最大差 5.83e-15，全部评分/分组最大差 2.84e-13，核验阈值 2e-12 只用于两个数值实现一致性，推广比较容差仍为 0。增广 LS 与独立输出代码见 `scripts/audit_generated_reference_calibration.py:169`、`:202`。15 项生产及独立 CPU 测试、Ruff/format 均通过。

| 证据 | SHA256 |
| --- | --- |
| [唯一实际 19fit 启动收据](../outputs/wam_stability_20261021/generated_reference_calibration_implementation/actual_fit_launch_691e3c0.json) | `120a96749ff77f1bb3f79cc0d7bd4f4fae555224bb9528f85d137e853d56fea7` |
| [生产 diagnostic](../outputs/wam_stability_20261021/generated_reference_calibration/diagnostic.json) | `ce2a73959c4cfab8edb40ba1f97d753aca119d50286ed41f0589fd63a60529b9` |
| [完整独立重建](../outputs/wam_stability_20261021/generated_reference_calibration_review/complete19_independent_verified.json) | `224d60eef89a4b19361de2bad0d50d8dac7a6e1dfd8e3c0d53e2bebc174662ba` |

本结果支持“生成参考包含一些可利用信息和可校准残差”的有限开发证据；它没有达到多组件、跨 episode 稳定改善，更没有新增闭环任务成功证据。保持当前失败，不依据本结果改尺度、ridge、cap、特征、开度开关或按任务重拟合，也不将第 19 组训练候选部署到物理实验。后续方案须另立协议，独立于本项的失败结论。
