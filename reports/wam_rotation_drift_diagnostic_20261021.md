# 018：世界坐标旋转漂移校正的开发诊断

固定三参数角漂移校正在整episode留出的开发交叉诊断中降低了WAM旋转误差，但**未通过事前推广门槛，不投入物理或fresh验收，不据结果重新调参**。H8/H16仍未优于保持、历史外推及独立拟合的noWAM/shuffle对照。位置和opening保持输入值不变，原checkpoint及017正在运行的raw控制协议没有改变。

018在任何新拟合前以`447865f`登记；代码为`c037e77`（算法与几何测试）、`4fe738b`（诊断入口与门槛测试）。先通过独立来源/分组审查，再执行一次注册的19次拟合：六折各paired/noWAM/shuffle三次，共18次；第19次仅保存全六episode训练候选参数，不计入留出成绩或推广判定。

仅使用8400/guidance1.0的既有48次state请求，按episode/frame/seed去重后为44个query、22个anchor、六个已查看开发episode。沿用008相同跨episode shuffle donor和共同前缀，H8/H16/H40分别计352/704/1398个有效端点；不补齐未知未来，不将重复seed或重叠窗口视为独立场景。这里的raw指标对应这些共同端点，不能混用原48次完整1778端点的H40数值。

校正为`R_out(tau)=Exp(-tau*b) @ R_pred(tau)`，world左乘，`tau=(endpoint_index+1)/20`秒，每query重置；位置为同一点的world位置。以fit-only标签的`Log(R_pred R_gt.T)`求固定weighted ridge闭式解，lambda=0.1秒²，再径向投影到`||b||<=0.25 rad/s`，最长2秒，因此校正角上界0.5rad。权重逐层episode/query/有效endpoint等权。部署形式只接收已生成参考及其时钟，不接收未来真实state、episode、task/object/contact标签。

下表各格为 **位置RMSE mm / SO(3) RMSE rad / opening MAE**。Identity逐值复现raw；corrector本身不裁剪opening，计分按008在所有方法的view/metric边界统一clip，之后保持该输入不变。

| 方法 | H8 | H16 | H40 |
| --- | --- | --- | --- |
| 原始WAM | 25.191 / 0.090896 / 0.062439 | 45.922 / 0.182583 / 0.092479 | 81.742 / 0.470659 / 0.110965 |
| Identity | 25.191 / 0.090896 / 0.062439 | 45.922 / 0.182583 / 0.092479 | 81.742 / 0.470659 / 0.110965 |
| Paired漂移校正 | 25.191 / 0.081348 / 0.062439 | 45.922 / 0.143087 / 0.092479 | 81.742 / 0.320673 / 0.110965 |
| 保持当前state | 36.678 / 0.054252 / 0.083365 | 67.826 / 0.118082 / 0.176362 | 120.009 / 0.318497 / 0.256812 |
| 历史匀速外推 | 26.418 / 0.044138 / 0.083365 | 68.798 / 0.110185 / 0.176362 | 168.252 / 0.324992 / 0.256812 |
| 独立拟合noWAM | 36.678 / 0.057800 / 0.083365 | 67.826 / 0.121646 / 0.176362 | 120.009 / 0.317563 / 0.256812 |
| 独立拟合shuffle | 53.599 / 0.070740 / 0.451764 | 97.571 / 0.124652 / 0.480361 | 166.102 / 0.354171 / 0.481093 |

Paired旋转误差相对raw在H8/H16/H40分别降低10.50%/21.63%/31.87%，三个窗口均不退化；H8、H16各六折均不退化，超过原定5/6要求。但两个短窗口都没有严格优于最佳对照，故`eligible_for_further_validation=false`。H40校正后也未优于hold/noWAM；这些事实全部保留，不据pooled改善选择其他lambda、normcap、坐标或时间函数。

| 留出episode | H8 raw → corrected (rad) | H16 raw → corrected (rad) | H40 raw → corrected (rad) |
| --- | ---: | ---: | ---: |
| 850 | 0.065317 → 0.063077 | 0.129919 → 0.113751 | 0.275117 → 0.153188 |
| 893 | 0.127075 → 0.103256 | 0.256111 → 0.188154 | 0.680987 → 0.472441 |
| 898 | 0.061410 → 0.049435 | 0.107244 → 0.077015 | 0.281324 → 0.152250 |
| 1098 | 0.078634 → 0.074123 | 0.141479 → 0.104709 | 0.338144 → 0.156940 |
| 1151 | 0.112717 → 0.106504 | 0.238501 → 0.206753 | 0.587679 → 0.470544 |
| 1172 | 0.066158 → 0.063873 | 0.139904 → 0.096649 | 0.391236 → 0.216036 |

原产物保留全部19组参数、fit/test/donor与权重、每endpoint时钟和训练残差、44个留出query的七方法逐endpoint误差及各episode全部位置/旋转/opening指标。全六episode候选沿用44条OOF donor的固定1398前缀，候选参数未部署，也没有当作留出模型另算成绩。

独立审计器`c0a3879`没有导入018生产fit/apply/pool/eligibility函数；它从既有缓存直接重建数据，使用三列augmented least-squares与零目标ridge行、L2投影及独立SO(3)计分复算相同19组模型和全部44×7方法。203个输入SHA重核；参数最大差约`1.11e-15 rad/s`，留出case全部字段最大差约`1.22e-15`，独立门槛判定同为False。生产13项、独立审计器5项针对性CPU测试及Ruff通过。

源码依据：[世界左残差与pi分支](../src/s2a/rotation_drift.py#L27)、[生成参考的逐点校正](../src/s2a/rotation_drift.py#L75)、[fit-only闭式解](../src/s2a/rotation_drift.py#L97)、[整episode分组](../scripts/diagnose_rotation_drift.py#L73)、[冻结推广门槛](../scripts/diagnose_rotation_drift.py#L207)。

| 证据（相对`outputs/wam_stability_20261021/`） | SHA256 |
| --- | --- |
| `rotation_drift_preparation_4fe738b/preparation.json`（0fit） | `3b4aa86539605098bc9795f9ffee1d6d3df226f7b4fbf01dfee913c7a1d9c733` |
| `rotation_drift_review/preparation_independent_verification.json` | `b5498a85089af505831b274d9ab14acecee45fa70560ace5de25181f5fecb9e7` |
| `rotation_drift_implementation/actual_fit_launch_4fe738b.json` | `10db594339c21b800cd5406afc1f2f869a4af3133d54efa34a9a5eca72eeab38` |
| `rotation_drift_diagnostic/diagnostic.json` | `04ea5945fce629bb6bae99777fb2e048cf7b73369f9ee87f31f02c4ac8a6a2a7` |
| `rotation_drift_review/independent19_verified.json` | `93bb9c98798934a8807b244e77ae47345e229582c11e63d8ac67a945fb2e244e` |

结论限于已查看开发数据中的这一固定假设：方向性偏差确实能够解释部分旋转误差，但当前校正未建立超过对照的短期预测能力。它不是机器人响应逆，也不能认证原checkpoint、独占归因权重或S2A，或从单条示范state误差推出任务成败。新增模型请求、物理步和fresh性能读取均为0；原始参考、旧报告、017运行与验收门槛保持不变。
