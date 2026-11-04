# 完整刷新模型后的联合目标新方向

2026-11-02。预登记1698267，实现f4ff7f1。新方向完整独立审核通过；出现安全、
下降且不立即触发恢复的步长，但尚无完整跟踪witness或物理资格。

## 全部固定尺度结果

使用已完整重建的anchor85 Js/Jq/Je，原完整cap-distance目标、float64算术及全部
硬约束语义保持。新模型的线性系数变化不是安全门或原.01m/.05rad caps放宽。

| alpha | joint/buffer safe | min raw slack | cap loss | 实际下降 | GN预测下降 | 比值 | reserve | 下一原模式 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| anchor | True | 9.98066661e-6 | 2.0100673083 | 0 | 0 | — | .997067 | pose |
| 1 | True | 9.05452106e-6 | 2.0106732070 | -.0006058987 | .0018071173 | -.335285 | .904452 | pose |
| .5 | True | 8.55603381e-6 | 2.0097517207 | .0003155876 | .0009772690 | .322928 | .854603 | restore |
| .25 | True | 8.60828766e-6 | 2.0096196160 | .0004476923 | .0005070621 | .882914 | .859829 | restore |
| .125 | True | 9.29584511e-6 | 2.0098115464 | .0002557619 | .0002581380 | .990795 | .928585 | pose |
| .0625 | True | 9.64343947e-6 | 2.0099375324 | .0001297758 | .0001302207 | .996584 | .963344 | pose |
| repeat | True | 9.98066661e-6 | 2.0100673083 | 0 | 0 | — | .997067 | pose |

全部原pose ceiling通过，全部caps12/24、prefix2、witness False。alpha1虽安全且
reserve>.9，但loss变差，不能接纳。按原接纳顺序会先接纳alpha .5，然后restore；
本轮固定探针未实际运行搜索接纳或更新，这些模式/接纳结论是明确的后验分类。

alpha .125显示新模型在局部可靠尺度可产生联合下降，并保持原reserve进入条件。
其最大安全预测误差1.33694802e-7；alpha .0625为6.50891813e-8，而alpha1为
1.39705535e-4，有限大步仍不可靠。不能用某个较好尺度替代全尺度记录或追认
原算法已经选择该尺度，也不能把一次下降当作完整控制成功。

## 执行与实际计费

QP前独立构建B/g/A/b逐值比较，登记完整矩阵与输入。实际1QP/26迭代/1NNLS，
原1e-6证书、objective scaling和dual recovery审核通过。固定五尺度全部先冻结、
通过原command预检，不裁剪、补选、重排或重试。0新模型拟合，不覆盖旧模型。

共享fixed_cap_probe_runtime处理QP及五尺度，既有response_probe_runtime处理完整
响应和独立复算，旧已关闭入口未改。生产7响应/2100内步/2185查询，独立7/
2100/2185，合计14/4200/4370、2model。runtime内部0QP不代替整体1QP/1NNLS。
原1e-10/torque1e-8布尔审核保持，完整响应、历史anchor/repeat exact，源码/输入
前后SHA不变。0GPU/物理/WAM/fresh/最终留出/实际参考消费，所有候选未下发。

工具session36575退出0，实际wait/reap完成，PID199859确认不存在；600秒TERM/
10秒KILL上限未触发，无自动重启。CPU空CUDA/LD、数值库线程1。
43项jobs32专项全部通过，0skip/collection错误，worker回收、来源前后不变。
Ruff/format及Git空白通过，未运行全仓回归。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 完整结果 | refreshed_cap_distance_probe_20261102/result.json | 42b322675f102b032a31b3286e4392b38c1c3f45587b516c00e3e3dfab34f8e8 |
| QP与证书 | refreshed_cap_distance_probe_20261102/qp_result.json | 9149e83f97d73f31fa3b71a7da52ee10e7cfb6e9a8639c7c8bca11dacff9ff49 |
| 保存分析/原规则分类 | refreshed_cap_distance_probe_20261102/saved_analysis.json | 41d57c866e77833d1675774f818d7fdd867836b4a298bae8bd7b3972a0d128ec |
| 工具终态 | refreshed_cap_distance_probe_owner_20261102/observation.json | 4423e19ff235366bca42397e4c9927652e29f5163ce344f5c0c967b8a84dd7f4 |
| 43项QA | qa/refreshed_cap_distance_probe_20261102/result.json | ca041b4fa371114cd4119a4622e08a0c143cf1e9798caca5325986d14c14e360 |

下一步设计并显式登记保护原reserve进入条件的pose接纳机制，作为新变体比较，
不能偷偷改旧接纳规则或放宽任何安全/跟踪门。有限迭代必须继承原全局命令边界、
pose ceiling与完整目标，记录预算、拒绝反馈、恢复次数及全部新响应独立审核，
不能把一次合适缩步当成整个搜索或任务成功。原18/12/24及稳定验收仍未达成。
