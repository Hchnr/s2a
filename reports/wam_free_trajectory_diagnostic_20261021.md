# 029：自由生成与 teacher 对照的完整结果

**完整采集和独立数学审计通过；现有027校正头没有在自由轨迹上带来点态收益，并表现出明显的噪声seed敏感性。稳定任务目标尚未达到。** 本项没有训练新模型或运行任务仿真，不能把诊断完成当作任务成功，也不能由teacher/free差异单独认定失败根因。

## 执行与证据范围

固定8400非EMA、normalization31/projection0/active10、guidance1、35步UniPC/shift5、H40/20Hz，使用原六个开发validation EP、每EP四anchor、seed42/43。先完整48条raw生成（1680 denoise），全部40×128 raw及normalized与028缓存精确复现，完整manifest落盘后才读取未来真实数据；随后同一模型完成144个匹配实际sigma/网络时间的teacher单次denoise，总1824。保存144个自由轨迹点和144个teacher点，原参数冻结、每步单次head/velocity/convert、完整初噪声/条件/mask/时间/来源均已独立核验。

仅物理GPU6运行模型。模型PID1019888/start171208087及CPU审计PID1046490/start171373807均实际wait0/reap、无信号退出；GPU6于07:13:52 UTC、模型wait后0.248秒记录0MiB/apps空。外层工具session49172实际终态0；root另存真实工具观测，不冒称手动Popen.wait。111份执行源码及快照不变，163份原权重stat不变。新fit、参数重建、训练EP读取、任务物理、fresh和真机均0。

本项teacher是“真实future加噪、推理时间配对诊断”，不是复算未改的原训练loss。teacher监督为epsilon−x0；自由轨迹监督另定义为向所选示范的点态pull `(sample_FP32−x0)/actual_sigma`。两者不是同一个目标，不能将两组MSE相除当作相同loss的性能退化倍数。匹配了实际sigma/网络时间/当前条件，不据相同seed字面值假定两个初始化路径的epsilon逐值相同。

所有统计先EP内均值、再六EP等权；144点和5760端点高度相关，不是144个独立trial。这里的clean/velocity误差都是normalized Action128空间，位置槽并非米，rotation6d并非SO(3)弧度；物理state误差及实际适配轨迹仍以[028报告](wam_scaled_readout_state_20261021.md)为准。

## 关键结果：teacher收益不能跨seed直接推广

固定027系数仅在保存的raw hidden上离线应用，没有重新拟合或执行新的适配轨迹。

| 匹配时间teacher，active10 velocity MSE | raw | 固定027 | 相对raw变化 |
| --- | ---: | ---: | ---: |
| 全体 | 0.005877863553 | 0.004587146932 | −21.96% |
| seed42 | 0.008100962536 | 0.003640422666 | −55.06% |
| seed43 | 0.003654764570 | 0.005533871197 | **+51.42%** |

027训练teacher只用了noise_seed42。因此本轮直接说明已有校正头对第二个噪声seed不稳健，支持检查噪声覆盖及可能过拟合；尚未独立证明具体机制。新teacher同时使用不同于027的实际sigma和第二个seed，不能把原027的57.45%与本轮21.96%当作完全相同协议的前后退化。

| actual sigma / step | teacher raw → 027 velocity MSE | free raw → 027 actual-clean MSE | free变化 |
| --- | --- | --- | ---: |
| 0.8077949285507202 / 19 | .001238251534 → .001279303813 | .027814060664 → .028114490994 | +1.08% |
| 0.5081728100776672 / 29 | .002627702441 → .002228765340 | .027806715532 → .028196799410 | +1.40% |
| 0.2323688119649887 / 33 | .013767636685 → .010253371642 | .028396320603 → .029050735276 | +2.30% |

三个实际时间点的free active10 clean MSE均变差。pooled为.028005698933→.028454008560（+1.60%）；位置槽.082214891056→.083741379661，rotation6d槽.000284595699→.000293304971，opening槽.031704741969→.031556116794。单个opening小幅改善不抵消其它分量。seed42/43的free clean总体分别变差2.58%/0.69%，不按seed选择系数。

## 输入差异与更新方向

当前condition video latent的配对差MSE为**0**。整个packed video/action latent的配对差MSE分别.707986079686/.055744970123；原027中心尺度下hidden平均行范数teacher/free为1.014521442/1.070376247，配对scaled hidden差MSE为.001742374053。范数没有数量级膨胀不代表方向相同；这些输入共同变化，不能把差异唯一归于未来video、hidden或某个接口错误。

free实际clean误差变化满足保存的逐点恒等式 `MSE_after−MSE_before=mean(2e·delta)+mean(delta²)`。pooled方向项+.000055241673、更新平方项+.000393067954，合计+.000448309627。seed42两项均为正；seed43方向项−.000189041940有利，但更新平方项+.000387909551更大，最终仍退步。这支持新训练应直接监督实际生成上下文中的clean残差，而不支持按已查看validation调缩放挽救028。

packed BF16与原FP32 sample的q MSE为1.208978921e−6，q/sigma MSE为6.349044287e−6；实际CUDA转换与理想FP64公式的差MSE约1.24e−6。它们比本项clean误差小多个数量级，但不能将不同误差的平方直接当作可相加的因果预算，也不能称原epsilon−x0标签有错。实际CPU/CUDA标量语义已按[事前数值合同](../plans/v0.0.1/029_scalar_promotion_addendum_20261021.md)分别独立核验。

## 验证、产物与后续决策

冻结实现整仓649 passed / 1外部integration deselected，Ruff353文件check/format通过；包含真实脚本入口正控及旧顺序负控。最初110源准备及其647测试原证据保留；[运行前隔离源码修订](../plans/v0.0.1/029_source_revision_addendum_20261021.md)在任何scheduler/GPU/WAM执行前登记，新111源prepared独立通过后才执行。20场景/40实际原scheduler转换各自对CPU/CUDA预定gold逐位通过，跨backend四个场景乘积不同如实保存。

展示入口[plot_wam_free_trajectory.py](../scripts/plot_wam_free_trajectory.py)只消费prepared/public/audit/owner四份最终JSON，不读EP/PNG/NPZ或重新评分。合成fixture验证了拒绝未审计数据、输出拒覆盖和五页绘图。正式5PNG/5页PDF已导出；初版长ylabel裁切的呈现问题保留，独立v2修图不修改实验结果或111源。

| 完整证据 | SHA256 |
| --- | --- |
| source_v2/preparation/prepared.json | `3960608c3c41dc45c999658ef517ce6b498c7a5260a1aa0b20006132ae3f254c` |
| collection/worker_result.json | `22e150c4759962618db4717aa5baa8e6494a7932b9534784124237b5bc76b7b7` |
| collection/result.json | `6ac7bcf810d88bc327316204fdb3179b87417e4e9b01b60427d0b7f2ad61511f` |
| review/collection_verified.json | `ae5887f543d139af268dca92503eadfe27bff9ff2da463251af3c2d96d05d979` |
| orchestration/collection/result.json | `1eff4c031a54a53a1a4a188a6ff31bc9586b02c5e693002d5aff9914dabedbf8` |
| final_json_figures_v2/manifest.json | `0bcc22130b0d8fc909704ac788b9978ce33fcd3e32f9eeb71b264ac1aac9a11a` |
| root工具终态观察 | `47425e1b42619c9d58b57d92fe891696b53f3786c5a37f97f7c030e2dca2e3f7` |

前六项位于`outputs/wam_stability_20261021/free_trajectory_diagnostic_source_v2/`，最后一项为`outputs/wam_stability_20261021/free_trajectory_runtime_observer/collection_owner_tool_exit.json`。额外最终只读peer也已通过：`review/formal_final_peer.json` SHA `a9003f1d0f993c83433469b25192bc4e52605ae28f23b553e4a4aacfce1410c9`，2426项实际消费SHA前后同、完整48/144/144协议及144点实际CUDA转换gold差0；math.fsum独立重聚合7800个分组标量，最大差8.881784197001252e−16。没有重新打开原EP/PNG或重新评分标签、拟合参数。展示另有实际五图审阅和v2执行收据，首版保留。

保留028 gate=False；029没有自动fit/物理授权。下一项在运行前另登记同步native训练数据，明确排除原fresh12初态及原六validation整demo，覆盖更多训练EP。随后候选使用当前自由生成上下文、显式`C0=sample−sigma*base`、直接clean残差监督和固定两轮latent轨迹聚合，训练noise按anchor派生且排除评测42/43。仅最终统一候选做完整state诊断及事前固定的有限开发任务评测，不以反复teacher改善代替任务实测。

最终目标仍是冻结新系统在原预留12初态×两seed达到≥20/24、每task≥4/8、末40端点原生连续成功；fresh性能未打开。任何适配模型的潜在成功都不追认为原raw8400 checkpoint正确，S2A仍只执行参考，不利用task标签重规划。
