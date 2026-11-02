# 修正安全模型下的完整cap-distance方向

2026-11-02，预登记实现9c1264d，CPU only。完整联合目标恢复后出现安全下降步，
但仍无完整跟踪witness，且会重新进入原reserve恢复模式，尚未解决可靠跟踪。

## 全部固定尺度

anchor85、原Jq/Je、已登记修正Js保持；硬约束A/b与上一轮旋转QP逐值相同，
仅B/g恢复全部24组位置/旋转cap-distance GN。独立构建与生产逐值一致。
原reference、控制接口、rationalized float64 loss、.01m/.05rad及安全门保持。

| alpha | min raw slack | joint/buffer safe | cap loss | 实际下降 | GN预测下降 | 比值 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| anchor | +9.98066661e-6 | True | 2.0100673083 | 0 | 0 | — |
| 1 | -8.94080491e-5 | False | 2.0078687905 | .0021985178 | .0007490084 | 2.93524 |
| .5 | -1.41848985e-5 | False | 2.0089540778 | .0011132304 | .0003856641 | 2.88653 |
| .25 | +9.25633040e-7 | True | 2.0095042429 | .0005630654 | .0001956220 | 2.87833 |
| .125 | +5.81759269e-6 | True | 2.0097840553 | .0002832530 | .0000985085 | 2.87542 |
| .0625 | +7.90081285e-6 | True | 2.0099253721 | .0001419362 | .0000494286 | 2.87154 |
| repeat | +9.98066661e-6 | True | 2.0100673083 | 0 | 0 | — |

全部caps12/24、prefix2、witness False。全部原pose ceiling均通过；数值安全且
loss下降不等于完整合格。第一原规则可接纳尺度为.25（仅后验分类，本轮并未执行
搜索接纳/更新），但reserve由.99707降至.09156；.125/.0625分别.58076/.78908，
均低于原.9模式切换阈值，下轮仍会restore。不放宽reserve或重新解释为成功。

安全模型最大误差随alpha1/.5/.25/.125/.0625为1.5569e-4/7.5996e-5/
3.8041e-5/1.9032e-5/9.5183e-6，近似随步长线性缩小；GN下降比值在小步仍约2.87。
这些结果支持当前完整方向的安全及误差模型还需校准，不能把单旋转方向上较好
拟合推广到完整目标，也不能仅靠缩步宣称模型已准确。这里只测了固定五尺度，
未证明真正导数极限、完整Jacobian或不可达性。

## 实际执行与计费

1QP/25迭代/1NNLS，原1e-6 KKT、objective scaling与dual recovery独立审核通过。
五尺度先冻结且全部通过原command门，不裁剪、补选或重排。复用旧已审核安全
模型，0新拟合；QP前另登记完整矩阵、输入和预算，不追加原60stage搜索。

生产7响应/2100内步/2185查询，独立7/2100/2185，共14/4200/4370、2model。
完整独立响应逐值一致，anchor/repeat exact，原numeric1e-10及布尔审核保持，
来源/源码前后不变。runtime内部0QP不能代替整体计数，整体1QP/1NNLS另有记录。
0物理/GPU/WAM/fresh/最终留出/实际参考消费；所有候选未下发。
session14807退出0并实际wait/reap，PID186564确认不存在，600秒TERM/10秒KILL
上限未触发，无重试。保存分析与原规则reserve分类不新增查询或优化。

47项32并发专项全部通过，无skip/collection错误，worker已回收；覆盖固定五尺度
准入、完整cap objective/audit、search driver与已有安全更新/局部性约束。
Ruff/format和Git空白检查通过，未运行全仓回归。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 完整结果 | corrected_cap_distance_probe_20261102/result.json | 72d7dc35130261aff75f89b1b9cbd23b5576a03409133c12bf2a11022cbf7b5c |
| QP证书 | corrected_cap_distance_probe_20261102/qp_result.json | 2d7a51f5f3fb272b0a237daec30875aa20311b8f8a41b37132b69cfde73f5691 |
| 保存分析 | corrected_cap_distance_probe_20261102/saved_analysis.json | ed9c2dcf0c038901f2bd0fb4f82e2e440d0a60753cd33bc0c0f5fa7f570420b4 |
| 工具终态 | corrected_cap_distance_probe_owner_20261102/observation.json | 4129d2de2d47733f4463a125b5455dbc90977ce6b2c53e37440b957e07d03776 |
| 47项QA | qa/corrected_cap_distance_probe_20261102/result.json | 20b2393f80371b5be668c6c40eb0ace714befd0f775cbeee2adc175c9b867ed1 |

下一步优先核查当前anchor完整局部安全/有符号误差模型重建的可执行采样域与数值
条件，不继续靠单方向秩一修正反复重新优化。任何新模型探针须显式登记样本、
预算、因果性和原command准入，不暗中改变原FD来源或使用资格。完整模型改进后
再验证联合目标及恢复行为。总目标和原18/12/24、稳定20/24且每task4/8保持未达成。
