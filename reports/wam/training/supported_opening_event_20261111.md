# 已生成闭合事件支持的训练折外筛查

2026-11-11。按[预登记](../../../plans/v0.0.1/wam/training/supported_opening_event_20261111.md)，
复用30条训练demo、300个D0请求的五折概率，仅在同一已生成H40参考的后16个端点
出现低开度目标时允许提前闭合；位置、姿态和时间不变。固定原八阈值与原事件门，
**0.7和0.8通过训练筛查，按预登记的最大新增命中规则选择0.7**。这是查看前项
训练结果后提出的探索，不能视为原方案通过，也不是独立验证或任务成功。

| 阈值 | H8 新增真实闭合 / 误闭合 | H16 新增真实闭合 / 误闭合 | H40 新增真实闭合 / 误闭合 | 筛查 |
| ---: | ---: | ---: | ---: | --- |
| 0.5 | 22 / 30 | 57 / 48 | 106 / 95 | 误闭合门未过 |
| 0.6 | 20 / 23 | 47 / 32 | 69 / 58 | H8误闭合门未过 |
| **0.7** | **11 / 11** | **25 / 15** | **41 / 25** | **通过，按原规则选中** |
| 0.8 | 11 / 0 | 18 / 0 | 28 / 3 | 通过，但新增命中总数较少 |
| 0.9、0.95、0.975、0.99 | 0 / 0 | 0 / 0 | 2 / 1、0 / 0、0 / 0、0 / 0 | H8命中与跨任务收益不足 |

0.7的H8 opening MAE为0.060967→0.061439，虽满足既定容差，实际略变差；
H16为0.071317→0.071171，H40为0.114344→0.114274。H8的11个新增真实命中
均在task0，11个新增误闭合中task0为3、task1为8；task1在H16新增10个真实命中，
task2在三个前缀均无新增命中。0.8误闭合更少，但不能在查看结果后改用它替代
预登记的0.7。端点与同demo的多锚点/seed相关，不能当作独立任务成功次数。

对12000个训练端点，3409个有后续生成事件支持；独立逐点实现的mask和全部阈值
统计与生产结果一致，最大统计差0。5个fold的240/60请求身份、逐请求概率及
原始fold组合独立重建，概率最大差0；没有重新拟合fold。支持只来自当前已有的
生成参考，不读取未来实测状态。无闭合预测的完整chunk不能被此规则补出事件，
因此task1 seed43已知的75个无闭合窗口仍不能由本规则解决。

零模型准入与筛查各执行一次，两个子进程均实际wait/reap、exit0、未发信号，
无自动重试。32进程CPU专项22项通过、0跳过，全仓Ruff和格式检查通过；源码
提交`83c532d6`。本阶段未拟合最终模型、未读取validation未来标签，也没有新
WAM、物理、GPU或fresh。下一阶段才按登记拟合一次并对48条既有validation raw
响应做离线评分，不改变阈值。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/supported_opening_event_20261111/result.json` | `34d3c63489776d4ef509d1749216a21050ea3443bc56db92bec39d85243c0034` |
| `outputs/supported_opening_event_dry_20261111/result.json` | `0ca37f904b1ee15af0fbaa90cf3bce007ce08188fa00f117a89314220fffa3f6` |
| `outputs/supported_opening_event_supervisor_20261111/dry-run_workers.json` | `6e54c2e2f7fe9fcc3e01efe7dbb3eb30cae0912b2de88bda7b162b7fbfc5eb88` |
| `outputs/supported_opening_event_supervisor_20261111/analyze_workers.json` | `509805868bf85f64e87d82273ca37d46db53978da55f963a0301205286bf635c` |
| `outputs/qa/supported_opening_event_study_20261111/result.json` | `2890a0a39240c334793f3fc1a6b7c552864a01647c193939344f3bb5694937e1` |

离线开度门不证明抓取、位置/姿态跟踪或控制安全；统一候选仍须正确参考、实际
物理、完整WAM开发和最终留出，18/12/24及稳定成功门槛均不变。

## 一次最终拟合与保存响应的离线评分

按固定阈值0.7，在全部30条训练demo上拟合一次；独立计算的最大梯度
`8.37e-8`、有限差分误差`1.57e-11`。48条此前已查看的validation raw响应
评分**未通过**原事件门：H8/H16开度MAE分别变差8.36%/6.76%，超过原5%保护；
仅task1有H8/H16新增真实闭合，未满足至少两个任务受益的条件。

| 前缀 | opening MAE raw → 候选 | 新增真实闭合 / 误闭合 | 修改开度端点 |
| --- | --- | --- | ---: |
| H8 | 0.063151 → 0.068429 | 8 / 1 | 9 |
| H16 | 0.046795 → 0.049957 | 10 / 1 | 11 |
| H40 | 0.097898 → 0.099779 | 21 / 1 | 22 |

H8的8个新增真实闭合均来自task1的同一validation demo、seed42；task0有1个
新增误闭合，task2无修改。低于0.35只表示二值事件，对连续opening误差仍可能
有害；这些相关端点不是8个独立任务。候选最大opening修改0.300，位置/姿态
参考未修改。这是对已保存响应的离线评分，不是实际闭环、抓取或新WAM结果，
也未执行原054状态门或授予物理开发资格。

拟合和评分子进程均由监督器实际wait/reap、exit0、未发信号；评分内部独立重算
特征的最大差`4.45e-16`，支持mask与分组事件指标独立重算差0。监督器随后错误
读取`validate.json`而非实际的`validation.json`，自身退出1；失败未被改写或重跑。
将另行独立核查已保存结果、输入和子进程收据，不改变候选或阈值。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/supported_opening_event_final_20261111/candidate.json` | `0f629cfb1a7a4aaaf9c611bc5b172257fd5cdeeb6039745ee04a33c238a49ed2` |
| `outputs/supported_opening_event_final_20261111/fit.json` | `21b333902b121d6c92ce3fe25e09afa850e9680654f89aab4c48446f28329d2e` |
| `outputs/supported_opening_event_final_20261111/validation.json` | `da7a495e3a385ac6e692efc16a559fda71afa2b687f4014d40675abeb9bdb18d` |
| `outputs/supported_opening_event_final_supervisor_20261111/fit_workers.json` | `551171ff77993a60d5dbcc03773ff6743683cf25d77d467a587d860c26a428aa` |
| `outputs/supported_opening_event_final_supervisor_20261111/validate_workers.json` | `b9d287d72f98067ebc34f1b126443d59676126b062de90b2ec163f9f502f45b0` |
| `outputs/qa/supported_opening_event_validation_20261111/result.json` | `d5f09335454135a3d325f056fc1b6a55b6390c3c0736a92118d8499b3e895a71` |

本阶段32进程CPU专项24项通过、0跳过，全仓Ruff和格式检查通过；没有新WAM、
物理、GPU或fresh。

## 独立复核

按[独立核查登记](../../../plans/v0.0.1/wam/training/supported_opening_validation_audit_20261111.md)，
新进程重新读取同一批48条raw响应及六条既有validation demo，独立积分局部SE(3)
参考、计算因果特征及概率、逐点生成支持mask和候选开度，再重算整体、任务、demo
与seed分组。raw开度、标签、支持mask及候选开度逐值一致，概率最大差
`6.67e-16`，分组指标最大差`1.39e-17`；仍确认H8/H16 MAE与跨任务收益三项门
失败。核对的68个输入文件前后SHA一致，没有新拟合或新样本。

独立核查确认原fit及validation子进程均实际wait/reap、exit0，日志指向上述精确
结果SHA。原监督器随后文件名错误退出1的观察保留在本会话；当时未单独保存其
owner退出收据，因此不能把子进程收据当作原owner正常退出的证据。新核查的dry和
评分进程均实际wait/reap、exit0、未发信号；32进程CPU专项19项通过、0跳过，
全仓Ruff/格式检查通过，核查源码提交`4df68c1b`。仍无新WAM、物理、GPU或fresh。

| 仓库内独立核查证据 | SHA256 |
| --- | --- |
| `outputs/supported_opening_validation_audit_20261111/result.json` | `13ba923a02173a34f325f3a0e0cc8fa5985950c06c013cbb17d40cf1f625c78c` |
| `outputs/supported_opening_validation_audit_dry_20261111/result.json` | `5e559f4f1de0812c1c9dc7845ae9498e838caf5390d771b14a9f5305e3ec3713` |
| `outputs/supported_opening_validation_audit_supervisor_20261111/dry-run_workers.json` | `9ccd9dcf38876983d97cd6031cc1ef7dd9af219ee0356934cbf52743dc5e0194` |
| `outputs/supported_opening_validation_audit_supervisor_20261111/audit_workers.json` | `f4f40afe8872df66cef79e073f2e9fb40be707d824f85ddfaf1398ae9b515030` |
| `outputs/qa/supported_opening_validation_audit_20261111/result.json` | `40ff00a766c6caac5a594f413904dff9e10ab6c44eb6b42abb7078f0ae9245f9` |
