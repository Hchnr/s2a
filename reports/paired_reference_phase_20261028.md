# 提前与临近参考相位分配（2026-10-28，CPU）

总目标仍是固定机器人/控制接口下的实际轨迹可靠跟踪和留出新初态任务稳定成功。GPU继续禁用。本对照按[执行前登记](../plans/v0.0.1/paired_reference_phase_20261028.md)在同actual368、已生成chunk46和650ms总时长下，仅比较额外50ms放在5→6或8→9段；保留全部12个原节点、原空间caps、安全约束和未消费后缀。原时序cursor0..12误差与新13相位误差分别记录，不将调速后的预测合格当成原时序控制通过。

## 实现及执行准入

- `1034daf`：参考合同、几何中点、两种时序及原节点评分、严格warm准入和30项合成测试。
- `9406fbd`：独立case入口、唯一响应缓存、两组同源owner及8项计账测试。
- `d3d70df`：完整独立H13重构、严格分类、原时序/新相位/原节点三套误差及10项合成测试。
- `68c6515`：最终执行合同；原prefix全量复核已在`6093134`关闭。

最终正式Python3.11.15/NumPy1.26.4/SciPy1.15.3环境的三个新测试文件及原prefix solver/ledger回归共81项通过（pytest 3.43s、实际exit0）。[持久QA](../outputs/paired_phase_native_qa_20261028_6l6nkn_q/result.json) SHA `e47c7c908b70812a1aa8ab7778a4f31d06224288eb4b0cd587b811a11e63e61a`，包含命令、日志SHA、运行环境和最终源码/计划/测试前后指纹，来源未变；owner对该记录的准入实际通过。新7个Python文件Ruff/format通过，三个CLI help通过。该范围不是全仓库检查，也不是实际控制性能证据。

执行前另行只读核对1864个前置来源，以及实际chunk46两组21字段：目标差异0、warm/used_start字节一致、两组命令准入通过、后缀12..39保留；该准备过程未调用MuJoCo或新增真实case响应。独立核心query/recurrence与已关闭prefix checker数学AST一致。两组动态warm随后均在正式唯一预检中通过；最小nonlinear safety slack分别为0.00735900207和0.04034866571，均未触发准入拒绝。

## 当前状态与结果解释

两个producer均已正常结束，各4000唯一完整H13响应、50共享迭代、1300000数值内步、1352001次query，均因共享响应预算耗尽而停止，完整witness为0。预检和solver初值恰好复用一次：每组4001次predict调用、1次cache hit；合计8000响应/100迭代/2600000内步/2704002次query。early搜索耗时733.140833s，late721.805304s；两组进程部分并行，不用壁钟差异比较方法效率。完整独立动力学复算现已正常关闭，两组严格分类、selected及阶段账本均确认一致；以下性能失败未因独立审核通过而改变。

| 指标 | early：5→6加50ms | late：8→9加50ms |
| --- | --- | --- |
| 动作/关节合格候选 | 1294/4000 | 678/4000 |
| selected / closest trace（从0计数） | 3440 / 2705 | 2396 / 2316 |
| 新13相位连续预测达标端点 | 6→10 | 6→10 |
| 原节点连续预测达标数 | 9/12 | 9/12 |
| 原时序连续预测达标端点 | 8/13 | 9/13 |
| 原cursor9位置/旋转误差 | 0.011150609m / 0.05575181rad | 0.011183500m / 0.05652764rad |
| 完整新路径witness | 0 | 0 |

表中误差和前缀均针对selected。10个新端点包含一个插入中点，不能写成原参考推进到cursor9：两组仍仅原cursor0..8达标，原cursor9在物理endpoint index10失败。原H12 prefix selected在同节点的误差为0.011516282m/0.05758048rad；多给50ms并重新优化仅略缩小该局部缺口，两组依旧同时超过原0.01m/0.05rad。early原时序前缀还从原H12的9降为8，不能省略这种退化或把新相位评分替代原时序。有限局部搜索未找到解，不构成物理不可行证明。

用两组各4000条保存TCP误差重算三套评分（每组12000个轨迹/合同组合），归一化平方误差差异0、严格pass/prefix标签无分歧；另从归档误差和安全标记重算prefix优先排序，selected索引一致。所有安全候选中达到新相位prefix10的只有early232、late23条，原节点prefix≥10均0；即使含不安全探针，两组保存原节点prefix的最大值也仍为9。该算术审阅没有重新运行动力学；全量独立动力学复算是下文另行闭合的证据。

selected末段的原节点误差如下，单元格为位置mm/旋转rad，原caps为10mm/0.05rad：

| 原cursor | 原H12 prefix | early | late |
| --- | --- | --- | --- |
| 9 | 11.516282 / 0.05758048 | 11.150609 / 0.05575181 | 11.183500 / 0.05652764 |
| 10 | 15.770479 / 0.09052689 | 15.270218 / 0.08903982 | 15.338634 / 0.08983412 |
| 11 | 20.049730 / 0.12794579 | 20.109695 / 0.12855902 | 20.231827 / 0.12992254 |

插入中点分别为early tick6的7.684899mm/0.04999065rad和late tick9的9.445279mm/0.04066552rad；它们达标没有解决后续原节点失败。原时序评价下，early在tick8已为10.533164mm；两组tick9为9.561963mm/0.06649711rad与9.368631mm/0.06757450rad，第13个物理端点对原cursor12则为18.601894mm/0.19229768rad和18.654394mm/0.19367854rad。H13比H12多50ms、6个变量和8.33%数值内步，且原timed多评价一个原节点，不能跨H直接按总损失宣称同合同胜负。

两组终止阶段均尝试修复新endpoint index10（原cursor9）：early从响应1107至4000、36迭代，late从1028至4000、37迭代，均未增加原节点连续合格数。不根据预算耗尽宣称优化收敛。

归档机制线索：selected最紧动态项仍为原cursor6的第五关节制动上界，early/late slack分别为1.64696e-6/8.94544e-6rad；原cursor9的Z命令分别接近-0.9999941/-0.9976502下界。该节点第六关节裁剪内步从原H12的14/25降至4/25和1/25，但位置/旋转仍双超限，不能归因于末点力矩裁剪一项。整个selected裁剪内步为early50/325、late36/325（原H12为69/300），不是无裁剪解。两组末callback均破坏已达标前缀；忽略安全和先前精度，保存候选在原cursor9的最小最大归一化平方误差仍为1.237896/1.222024，且这些候选原节点前缀只剩5/4。

后续优先审视已保存单轴差分组中的pose下降方向、前缀精度及box/rate冲突，再据证据选择步长接受或更早q5姿态/速度分配的干预。现有trace只保存TCP误差和全局safety minimum，不能据它声称已得到每个关节约束的独立差分梯度或KKT证明。完整独立资格现已关闭；不追加本次搜索预算、改变起点或继续试速度，完整约束的有限复算另行登记。

后续[保存差分诊断](saved_trajectory_differences_20261028.md)现已执行：H12/early/late分别54/50/50个完整差分组均与紧邻非拟合候选配对；原cursor9可行基点主分析仅10/5/1条，均无L∞≤.01样本。前两组8/10、4/5预测改善但实际target变差，支持另登记有限回缩检验，不能据此认定缩步必然找到解。该只读诊断没有新增响应或改变本次搜索；其执行时H13 checker仍运行的快照原样保留，当前完整复算已关闭。

## 正常退出及独立复算

两个producer owner的1876项来源SHA表完全相同，结束前来源检查通过。root工具会话99422和83708均实际取得exit0；下表child的actual wait/reap0、无信号、无原始错误，PID/start_ticks与launch一致。没有重试或覆盖。

| 组别 | owner工具会话 | child PID / start_ticks / parent | producer证据 |
| --- | --- | --- | --- |
| early | 99422 | 3225627 / 613724158 / 3225434 | [registration](../outputs/tracking_v5_paired_reference_phase_early_execution_20261028/registration.json)、[launch](../outputs/tracking_v5_paired_reference_phase_early_execution_20261028/capture/launch.json) |
| late | 83708 | 3226886 / 613730067 / 3226782 | [registration](../outputs/tracking_v5_paired_reference_phase_late_execution_20261028/registration.json)、[launch](../outputs/tracking_v5_paired_reference_phase_late_execution_20261028/capture/launch.json) |

| 归档 | SHA256 |
| --- | --- |
| [early verified](../outputs/tracking_v5_paired_reference_phase_early_20261028/verified.json) | `9f751f169790027391cfc9317e71a4687245b9b0688e01d8c0c83967b84f5101` |
| [early registration](../outputs/tracking_v5_paired_reference_phase_early_20261028/registration.json) | `d38a7164c73e3904549acd8dfb1d124468ea9ac99c8d1c87522c92cdd40c3b43` |
| [early owner result](../outputs/tracking_v5_paired_reference_phase_early_execution_20261028/result.json) | `d73b33655dde4d9bc3dea51c8acb574e6121de5a2e06f94c00a5bd6082b76050` |
| [late verified](../outputs/tracking_v5_paired_reference_phase_late_20261028/verified.json) | `8c817b99c99405ce2b5c29e8e0221279b461190b4a929a742ca9500c01d8be1e` |
| [late registration](../outputs/tracking_v5_paired_reference_phase_late_20261028/registration.json) | `ccf7cdf6ec3a295c7d0dac732379b63616a2ce5620f183d993b64ead93462149` |
| [late owner result](../outputs/tracking_v5_paired_reference_phase_late_execution_20261028/result.json) | `02cafc5370235432d9905dd06968cbb3fb95b1714d6fe073c6589fa5d3f35bfe` |

两个完整独立checker各唯一执行、正常关闭，时限各5400s、空CUDA/LD和数值库单线程。根工具会话81399与15578均实际取得exit0；early child3240665/start613804281/parent3240423，late child3241251/start613807509/parent3241135均有一致launch、actual wait0/reap0、无信号和无原始错误，进程已消失。没有重试或覆盖。

每组独立复算全部4000响应/1300000内步/1352001查询/52000预测周期，三套合同严格分类分歧均为空；安全轨迹1294/678、selected3440/2396、closest2705/2316与50次阶段迭代账本一致，完整witness均0。全trace TCP误差最大差分别3.21271e-15/3.94129e-15，安全minimum均2.66454e-15。四个完整候选名称warm/initial/selected/closest实际只有三个唯一命令，其中保存q/v/torque可逐项比较：early最大内步q/v/torque差4.99600e-16/7.16094e-15/1.26654e-12，late1.55431e-15/6.82787e-15/1.02318e-12。不能声称全部4000条有旧q/v数组可比较。

原rate违规探针分别217/1473条，未获得安全资格。原生JIT与纯Python AST goal的独立诊断最大差均8.94070e-8，分别涉及38554/37895个周期；这与producer对独立原生JIT的匹配误差分开保留。每组checker1883条引用及owner1882项来源，合并去重1890项由root重新hash全部匹配；另一只读审阅连同退出/结果凭据共核1904个唯一文件，绑定与来源均通过。

| 独立复算归档 | SHA256 |
| --- | --- |
| [early verified](../outputs/tracking_v5_paired_reference_phase_early_check_20261028/verified.json) | `4c0bf31490e701b2a804a26fd8280c0af463c80dc90bf577dcbc78403225a54e` |
| [early registration](../outputs/tracking_v5_paired_reference_phase_early_check_20261028/registration.json) | `ef610a617b98763923090d37a938820ca1b3d0cc36e058c8f81e7471c10de11b` |
| [early owner result](../outputs/tracking_v5_paired_reference_phase_early_check_execution_20261028/result.json) | `1dae567731040e13d825bc025d4c9fdceafefd04416ba8b7ad448c4b2bf357b5` |
| [late verified](../outputs/tracking_v5_paired_reference_phase_late_check_20261028/verified.json) | `91d27e2bd73bcc8e6d46cd5a4594f0658de53c2760671e80c6cf66dbc97e2e99` |
| [late registration](../outputs/tracking_v5_paired_reference_phase_late_check_20261028/registration.json) | `91131d64c32db9d50c3f42908f64fecc0abdb6de613b52d30bf44d3346ad7789` |
| [late owner result](../outputs/tracking_v5_paired_reference_phase_late_check_execution_20261028/result.json) | `393eb2ee31e4f2d385f8375d1b30c945e4b24e324fa74dc2bb2efa5975471613` |

登记的命令拒绝和动态warm拒绝分支本次都未触发。两组完整搜索仍无witness，故不进入13周期物理pilot。[固定方向回缩检验](../plans/v0.0.1/saved_direction_backtracking_20261028.md)已登记但尚未执行：全部16方向、六个固定比例、每侧96响应，保留完整约束和端点匹配，不追加本次搜索预算或悄悄增加起点。

当前新增物理、世界模型调用和fresh案例均为0，实际仍止于cursor8，task/grasp0。若找到完整模型witness，仍须另登记13周期原生物理及独立重放；本计划不自动派发动作。原18条正确参考、12条WAM开发、最终24条新初态和全部任务门槛保持，总目标未完成。
