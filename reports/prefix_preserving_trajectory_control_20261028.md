# 保留达标前缀的轨迹修复结果（2026-10-28，CPU）

总目标仍为固定机器人/控制接口下的实际轨迹可靠跟踪及留出新初态任务稳定成功。GPU继续禁用。本轮按[执行前登记](../plans/v0.0.1/prefix_preserving_trajectory_control_20261028.md:7)保持actual368、已经生成的chunk46前12点、原50ms目标时间和全部原精度/关节/动作条件。

唯一搜索已正常结束：4000个完整H12响应、1200000数值内步、54次迭代，耗时647.030379s。连续合格预测前缀从6点提高到9点，但cursor9仍超原位置及旋转门槛，完整witness为0。这里只改善了局部模型中的早段跟踪，尚未扩大实际推进范围；新增物理、世界模型调用、fresh案例均为0，实际仍止于cursor8、task/grasp均0。全部4000响应的独立动力学复算正在进行，不能提前称为完整数值资格。

## 搜索结果与范围

[新solver](../src/s2a/prefix_preserving_trajectory_plan.py:10)保留已达到的连续前缀精度caps，全部72个命令变量仍可调整。每阶段仅对第一个失败点引入误差辅助量；完整H12动作、内步关节、制动及终端条件始终约束，最终必须全部12点合格。与原全窗口最坏误差优化的区别在于修复顺序，未调整参考时间、跳点、放宽门槛或增加预算。初值仍是原H12 warm，没有采用后来真实动作/状态。

| 阶段 | 连续前缀 | 响应计数边界 | 本阶段迭代 | 退出原因 |
| --- | --- | --- | --- | --- |
| 修复cursor6 | 6→7 | 1→220 | 3 | callback发现前缀增长 |
| 修复cursor7 | 7→8 | 220→366 | 2 | callback发现前缀增长 |
| 修复cursor8 | 8→9 | 366→658 | 4 | callback发现前缀增长 |
| 修复cursor9 | 9→9 | 658→4000 | 45 | 共享响应预算耗尽 |

4000条中1797条动作/关节合格，1014条同时保留最初6点，610条达到连续9点；没有任何候选达到连续10点。selected为从0计数的trace3268；closest_safety_attempt为trace1742，只有4点连续达标，不能替代selected。上述计数及选择已用保存数据独立做算术审阅，但尚不代替完整动力学复算。

| 原cursor | selected位置误差（m） | selected旋转误差（rad） | 原0.01m/0.05rad条件 |
| --- | --- | --- | --- |
| 0 | 0.00222815261 | 0.0180321467 | 达标 |
| 1 | 0.00782816079 | 0.0358002257 | 达标 |
| 2 | 0.00920004318 | 0.0451665180 | 达标 |
| 3 | 0.00746149021 | 0.0449147683 | 达标 |
| 4 | 0.00757769678 | 0.0499614682 | 达标 |
| 5 | 0.00993002412 | 0.0429844442 | 达标 |
| 6 | 0.00803147961 | 0.0343734860 | 达标 |
| 7 | 0.00881910111 | 0.0156864321 | 达标 |
| 8 | 0.00998838133 | 0.0230532490 | 达标 |
| 9 | 0.01151628224 | 0.0575804846 | 两项超限 |
| 10 | 0.01577047881 | 0.0905268929 | 两项超限 |
| 11 | 0.02004972971 | 0.1279457877 | 两项超限 |

selected九组约束最小slack依次为：box下3.58117924e-5、上0.224538724；rate下3.93531000e-6、上1.30956882e-4；内步关节下0.283237612、上0.00488230644；制动下0.262010376、上6.42068366e-6；terminal为0.00554230457，即最终q7速度为-0.00554230457rad/s。保存q/v/命令重新计算与归档slack最大差4.44e-16。制动最小项在cursor6的第五关节；这不构成其为唯一瓶颈的证明。

selected有69个内步出现力矩裁剪，共83个关节内步（第六关节29、第七关节54）；不应称为无饱和。

前缀保持确实避免了原H12 selected只保留2点的退化，但尾段最大旋转误差0.127945788rad高于原全窗口selected的0.112776418rad。不能只报告前缀增长或较小的位置最大误差。有限4000次局部搜索未找到解，也不能据此宣称原目标在物理上不可达。

selected之后731个响应没有改善合规九点前缀的最佳值，但不能写成优化已收敛。最后callback为trace3981，cursor9误差仍改善到约11.4712mm/0.0573554rad，同时违反cursor4/5的前缀门槛及nonlinear safety（最小slack=-9.31257141e-6rad）。即使忽略安全和前缀，全部4000条中cursor9最好的最大归一化误差仍为1.144489；主要缺口不能解释为严格边界判定。selected的cursor9位置误差中z分量为0.01039523081m，cursor8/9的z命令已接近-1，且cursor9第六关节有14/25内步力矩裁剪；这些是联合动态约束的线索，不是单一根因证明。

下一项[只读相位诊断](../plans/v0.0.1/saved_trajectory_phase_diagnostic_20261028.md)先区分保存预测的路径滞后与空间偏离。它不生成新响应或动作，不会把离线最佳相位作为原50ms跟踪通过证据。已有16周期真实保持未推进，而早期governor也曾因大量暂停/桥接未取得任务成功；任何新增调速机制必须解释这些历史限制，并保留原时序评价和全部原节点。

## 执行与独立复核

执行代码提交`263ce1d`，solver提交`58440fa`；正式Python3.11.15/NumPy1.26.4/SciPy1.15.3环境33项CPU组件及账本测试通过，证据[持久QA](../outputs/prefix_native_qa_20261028_vm4v8_yg/result.json)，SHA `738878bd5b9ed52a524ccafe19cf6b4e3bd8278412bac258a1b14e2e089a37f2`。组件测试使用合成响应，不是本case的控制资格。

| producer证据 | SHA256 |
| --- | --- |
| [verified](../outputs/tracking_v5_prefix_preserving_trajectory_control_20261028/verified.json) | `89da5e5287ef592edefa6eaea90b801c603b6abfaaf527a56e7dcb8f4258bebb` |
| [registration](../outputs/tracking_v5_prefix_preserving_trajectory_control_20261028/registration.json) | `334e795379074f15d4d03a98220f655e56bdaf5b4f213291012b50958cd468ed` |
| [owner result](../outputs/tracking_v5_prefix_preserving_trajectory_control_execution_20261028/result.json) | `15572bc960c3190e7686308852d50da70613a25ccbb1b4786b408d5e23a8e371` |
| [owner registration](../outputs/tracking_v5_prefix_preserving_trajectory_control_execution_20261028/registration.json) | `0875b735c2ba12e12cce6028764489af597adddf53584ea08a6609edd0a574d6` |

producer child PID3117933/start_ticks613192333/parent3117803；root会话90660已取得owner实际exit0。child actual_wait0/reaped0、无信号、无原始错误，身份一致。1766项owner来源、上轮v2的1820引用及本轮登记引用经重新核对SHA一致，去重1842路径。唯一历史迁移仍是原先审计过的两份测试文件9db6efa→2f09739；没有新增例外。

[独立checker](../scripts/check_tracking_v5_prefix_preserving_trajectory_control.py:324)已通过原owner以`--stage check`启动，root会话70434，child PID3131773/start_ticks613267008/parent3131636，CUDA/LD均为空。登记范围为全部4000响应/1200000内步/1248001次forward查询，并核对三份完整候选、每点误差、全部约束、阶段及预算。输出使用独立`outputs/tracking_v5_prefix_preserving_trajectory_control_check_20261028`，不覆盖producer；实际退出和完整结果尚未取得。

后续须先关闭完整独立复算。没有完整witness，本轮不派发action；下一机制需依据首失败点、约束和参考时序的证据单独登记，不能以重复搜索或放宽原指标代替推进。原18条正确参考、完整12条WAM开发、最终24条新初态及全部任务门槛保持，总目标尚未完成。
