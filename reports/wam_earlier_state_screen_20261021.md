# 旧训练阶段 guidance1.0 开发 state 对照（2026-10-21）

追加的8400/8600共96次预测已完成，两组各48/48，服务正常释放；与本轮已有8800/15400 guidance1.0预测构成192次配对 state 比较。**这次没有新增物理任务评测，不能据此宣布某个 checkpoint 稳定完成任务。**

8400在 pooled H8/H16 的位置、SO(3)、opening误差均低于另外三组；H40由8800位置误差最低，8400旋转和opening误差最低。8400因此提供了不同训练阶段/采样条件的有价值开发候选，但不能用 state 排名代替闭环任务验证。

## 固定协议与范围

- 8400/8600/8800均属于原22-source run；15400属于从8800继续训练的另一个23-source run。15400并非原run的后续迭代，本报告不把最大iteration称为“最佳”。所有服务加载显式非EMA权重，并启用十有效槽mask。
- 保留同一份历史开发集合：3任务、6episode、24phase样本、22个不同 `(episode, anchor_frame)`；采样seed42/43，各checkpoint48请求。阶段为机器人状态proxy，7/24使用fallback，不能当作已验证抓取事件；预训练实际消费仍未知。
- H40、task-only、20Hz、35去噪步、guidance1.0、shift5固定。5/24样本真实未来不足40步；H40统计每组1778个有效endpoint，其中38/48请求具有完整40步。H8/H16分别384/768个endpoint，无padding计分。
- 8400/8600复用旧完整byte SHA清单，每组163个文件的路径、size、mtime一致；未重新读取全部权重字节计算新SHA。原样保留arrays/image绝对路径和SHA，样本逐字段与首轮guidance1.0筛查相同。
- 准备前检查显式8400/8600路径下3759/3533份request记录，未发现guidance1.0请求，因而未发现可复用的完全相同协议预测。追加预算只有96请求；没有重新查询8800/15400。
- 预算引入commit为`5b41f06`，启动快照HEAD为`fd9d333`；前者已验证是后者祖先。原launch的`budget_git_commit`字段记录启动HEAD，保留其原始字节。

## 配对 state 指标

误差从当前实测pose积分原始预测delta，比较真实未来state，完全绕过S2A和控制器。单位为位置mm、SO(3) rad、opening归一化绝对误差。数值按有效endpoint汇总，并非对每请求RMSE再平均。

| checkpoint / guidance | 前缀步数 | 位置RMSE mm | SO(3) RMSE rad | opening MAE |
| --- | ---: | ---: | ---: | ---: |
| 8400 / 1.0 | 8 | 24.384 | 0.088644 | 0.058538 |
| 8600 / 1.0 | 8 | 27.996 | 0.113729 | 0.067291 |
| 8800 / 1.0 | 8 | 29.297 | 0.105865 | 0.073929 |
| 15400 / 1.0 | 8 | 25.902 | 0.096772 | 0.067205 |
| 8400 / 1.0 | 16 | 44.623 | 0.177223 | 0.086038 |
| 8600 / 1.0 | 16 | 50.113 | 0.235302 | 0.102860 |
| 8800 / 1.0 | 16 | 47.812 | 0.217048 | 0.116804 |
| 15400 / 1.0 | 16 | 46.237 | 0.198837 | 0.097410 |
| 8400 / 1.0 | 40 | 85.521 | 0.479894 | 0.114660 |
| 8600 / 1.0 | 40 | 103.871 | 0.546207 | 0.118672 |
| 8800 / 1.0 | 40 | 82.414 | 0.542640 | 0.146364 |
| 15400 / 1.0 | 40 | 89.913 | 0.507355 | 0.116766 |
| 保持当前state | 40 | 128.832 | 0.309419 | 0.255167 |
| 当前历史匀速外推 | 40 | 179.117 | 0.327273 | 0.255167 |

四组H40位置和opening均优于保持当前state，朝向均差于保持当前state。结果支持模型在这些开发观测上产生了有用的部分状态变化，同时仍有明显旋转质量问题；示范state并非任务唯一可行轨迹，误差本身不能证明任务可成功或不可能成功。 pooled排名也不等同于各任务分量排名，完整按任务/阶段统计保留于JSON。

## 独立核验与运行结束

只读汇总核对配置、样本、服务权重身份、有效mask、完整48项清单、请求参数、响应SHA和反归一化，再从原始响应重算全部state误差。只有四组48项全部完整后才输出比较；部分结果只报告进度。

第二套审计独立核对642个输入文件SHA，直接用Gram-Schmidt、逐delta位姿积分和SciPy SO(3)角度计算，未调用S2A几何或指标函数。与已保存逐endpoint误差的最大差为位置`2.50e-16 m`、SO(3)`8.89e-16 rad`、opening`0`。审计不产生新模型请求或物理步。

两组state worker退出码均0；模型由所属编排正常SIGTERM结束，退出码均-15；均无error.json，记录的模型/worker进程已退出。结束时物理GPU4/6均为0MiB。没有启动GPU0–3任务。

入口：`scripts/prepare_wam_earlier_screen.py:15`、`scripts/summarize_wam_earlier_screen.py:31`、`scripts/audit_wam_earlier_screen.py:34`。新增三个脚本的ruff check/format通过。

| 证据文件（均相对本仓库） | SHA256 |
| --- | --- |
| `outputs/wam_stability_20261021/earlier_screening/preflight.json` | `969f215dd52b292a2550d0e9c613e2c2a34edb06ac2e3317866317c7a5968b69` |
| `outputs/wam_stability_20261021/earlier_screening/summary_final.json` | `578184f63d0e54a2cac8f87b01775e9edb2544bbd739c0f77347bf65b954ef85` |
| `outputs/wam_stability_20261021/earlier_screening/independent_final_audit.json` | `dfbd07b7376053a56fbfc79b8eecc5b1feb52e6ca1953bbe880c9601d1da24e8` |

复核命令使用新的输出文件名，已有快照拒绝覆盖：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/summarize_wam_earlier_screen.py --output outputs/wam_stability_20261021/earlier_screening/summary_review.json
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python scripts/audit_wam_earlier_screen.py --summary outputs/wam_stability_20261021/earlier_screening/summary_review.json --output outputs/wam_stability_20261021/earlier_screening/independent_review.json
```
