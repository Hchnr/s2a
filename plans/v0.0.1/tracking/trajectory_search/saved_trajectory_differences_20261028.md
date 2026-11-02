# 保存差分组与后续候选的局部线性诊断（执行前登记，CPU）

总目标仍为固定机器人和控制接口下，依据当前实测状态、因果历史与世界模型已生成参考，产生可靠实际跟踪并在留出新初态稳定完成任务。原18条正确参考、12条WAM开发、最终24条新初态及任务门槛不变。GPU继续禁用。

H12 prefix完整独立复算已关闭；H13 early/late各4000响应的producer已正常结束，完整witness均0，完整独立checker仍运行。三个selected均只覆盖原cursor0..8；原cursor9同时超过10mm/.05rad。只读结构检查还发现三个selected本身都是差分探针，末callback没有完整差分组。不能用构建线性模型的探针验证该模型，更不能把预算中断称为收敛。

本诊断只检验保存的局部线性预测如何解释后续新候选的pose偏差和前缀损失。它不调用优化器或动力学、不生成新action、不查询MuJoCo、不推进仿真、不调用世界模型、不读取新的实际反馈。输入是已保存的以下三份producer结果及其注册/来源、既有checker的身份和终态记录：

| case | outputs下的结果 | 固定SHA256 | 原cursor9的内部index |
| --- | --- | --- | --- |
| prefix | tracking_v5_prefix_preserving_trajectory_control_20261028/verified.json | `89da5e5287ef592edefa6eaea90b801c603b6abfaaf527a56e7dcb8f4258bebb` | 9 |
| early | tracking_v5_paired_reference_phase_early_20261028/verified.json | `9f751f169790027391cfc9317e71a4687245b9b0688e01d8c0c83967b84f5101` | 10 |
| late | tracking_v5_paired_reference_phase_late_20261028/verified.json | `8c817b99c99405ce2b5c29e8e0221279b461190b4a929a742ca9500c01d8be1e` | 10 |

核对同actual368、已完成367的previous command/latch、同chunk46来源、共同机器人/工具/增益/关节界/力矩界和因果error profile。H13原节点映射必须把原cursor9放在index10；两组与H12在时长、变量数和hard前缀上有区别，不据描述性统计宣称同合同优劣。H13只读算术可在独立checker运行时执行，但结果必须分别保存checker运行身份快照或已关闭的完整资格，不能用producer的passed代替独立动力学证据。

## 固定识别与配对规则

1. 用trace的float64 commands精确bytes建立唯一索引，不用allclose。逐个base按原solver复现每轴FD：若`x[i]+1e-4<=upper[i]`，用`+1e-4`，否则用`-1e-4`；upper含首命令rate与box交集。
2. 完整差分组要求全部72或78个探针均已保存、在base首次出现后按轴顺序连续出现、对应唯一base。所有探针须位于同一phase的`[evaluations_before,evaluations_after)`；base可来自此前缓存，单独标记。缺轴、缓存探针、乱序、非唯一、跨phase或预算截断单列，不补算缺失点。
3. 每完整组只配全部探针之后**紧邻的新行**。它须同phase，且不是任何更早base的合法FD探针，包括不完整差分组中的探针。若不满足，排除该组的配对，不能跳过它寻找更有利点。保存完整识别和排除计数。
4. selected和最后callback仅按精确bytes关联，未匹配就保留未匹配。不得仅因selected身份就把它当作有完整差分组的base；所有点统一按上述规则识别，不按selected身份筛选。也不把callback或其邻近行称作已接受、可行或最终优化步。

trace只保存唯一命令的首次插入顺序，未保存cache-only函数/梯度调用。即使上述配对完全精确，也不能恢复实际SLSQP Jacobian调用、线搜索、接受过程、完整epigraph目标、乘子或KKT状态。输出名称使用“完整差分组与后续新评估点”，不声称恢复优化器执行历史。

## 固定计算与汇总

从保存TCP误差按原solver重算两项平方归一化值`q=[sum(e_xyz²)/.01²,sum(e_rot²)/.05²]`，严格重算原prefix和feasible标签，不以诊断容差放宽caps。每轴用**名义signed step**计算`J[:,i]=(q_probe-q_base)/delta_i`，对保存后续候选`y`计算`q_pred=q_base+J@(y-x)`。不将负的线性预测截成零。

保存base/probes/trial索引、phase、全轨迹q_base/q_pred/q_actual，target两项及max评分、此前hard前缀最大超限、预测残差，以及步长L∞、RMS、L∞/1e-4。下降比仅在预测target max下降大于1e-12时定义，否则为null；它不是优化器objective/merit ratio。

box和相邻rate从commands精确计算，原box范围±1、rate最大.25，单独保留原1e-12容差与raw slack。动态安全只允许对保存的标量minimum做差分外推，命名为minimum safety scalar prediction，并报告actual minimum、1e-8 buffer和原joint_feasible。最小约束可能切换，即使每个原约束都是线性的，标量minimum也可非线性；不能据其残差指认完整safety Jacobian有错或得到第五关节制动导数。

主分析只取当前phase目标为原cursor9（H12 index9、H13 index10）、base满足原joint_feasible、原hard前缀、box/rate和minimum≥1e-8的合格配对。其余所有配对也保存并单列。按事先固定的L∞区间`<=1e-4`、`(1e-4,1e-3]`、`(1e-3,.01]`、`(.01,.1]`、`>.1`报告数量、pose预测误差及“预测前缀通过而实际失败”数量；不选择有利base或末段单一pair，不将不同case的配对作为独立重复实验作显著性推断。

若大步频繁破坏在线性近似中仍通过的hard前缀，而小步残差较小，只能支持下一次检验trust region/回缩机制，不能证明该修改会找到witness，也不能据本数据事后选择一个必胜半径。若rate或buffer频繁失败，可另检验可行性接受规则；SQP中间候选允许不可行，当前最终选择已经过滤安全资格，强制每步可行也可能阻碍恢复。若缺乏可行base或小步配对，明确样本限制，不强行得出机制结论。

## 实现、运行与来源

新增`src/s2a/saved_trajectory_differences.py`、`scripts/diagnose_tracking_v5_saved_differences.py`和有意义的合成测试。旧producer、checker、solver、测试和已执行计划全部保持原字节。合成检查覆盖精确bytes、FD方向、缓存/缺轴/phase边界、排除拟合探针、callback未匹配、仿射误差和minimum切换反例及原caps/rate边界。

先在正式`outputs/libero_env311/bin/python`执行合成测试并保存QA的实际子进程wait、返回码、命令和源码/计划前后SHA，再提交实现和计划。正式只读诊断显式空CUDA/LD、数值库单线程，固定seed42但无随机抽样。新结果目录`outputs/tracking_v5_saved_differences_20261028`拒绝覆盖，无自动重试；保存输入/源码/计划SHA、运行环境、全部配对、统计和checker前后身份快照。根工具会话实际exit单独记录在报告中，不伪造额外owner。

诊断passed仅表示解析、来源和算术合同完成；control/retiming/task资格始终false，new forward responses/query/physics/world-model/fresh均0。H13完整checker关闭后另记证据，不修改本执行前合同。当前没有完整witness，不进入任何新物理pilot。
