# 完整H12的反复内点恢复与信赖域搜索

2026-10-29：138项专项CPU QA、四项实际入口准入、44响应生产及44响应完整独立复算
全部正常关闭。21个QP、15次接纳中，8次恢复安全余量、7次降低完整轨迹最大平方超限。
最终数值点的最坏超限下降2.3901%，但原达标前缀从9退到4、双caps合格点9→5，
完整witness仍0，selected为空。实际仍cursor8/task0/grasp0，总目标尚未达成。

实现提交为`faa48f9`。生产和独立复算合计88完整响应、26400数值内步、27458查询；
仅生产求解21个QP、累计756内部迭代。新增物理积分、world-model调用、留出初态
验收和参考消费均0。GPU继续禁用，原18正确参考/12WAM开发/24最终留出门槛保持。

## 固定输入与新增机制

[执行前计划](../plans/v0.0.1/interior_trajectory_search_20261029.md)及
[显式配置](../configs/interior_trajectory_search_20261029.json)固定seed42，以历史3557
作为初始点，复用3558..3629的72轴名义signed1e-4差分，不新增FD。输入封装为
`outputs/constraint_residual_repair_inputs_20261029/bundle.json`，SHA256
`eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023`。
初始安全Jacobian的144936个未来命令元素及pose的792个未来元素全部exact0。

每个响应都从actual368实测q/v、completed367命令与方向latch、固定finger及原历史
profile重新开始，使用当时已生成的12点参考。保留原box/rate.25、关节裕量.05rad、
braking.1s、terminal方向、位置.01m/旋转.05rad合同。模型随预测状态更新M/J，
finger和外部对象按原模型合同固定，接触等遗漏保持；这是局部数值预测与独立算术
复算，不能称为新的完整物理仿真或闭环控制证据。

新增机制是可反复进入的余量恢复、实际/预测改善比控制半径，以及接纳点间的因果
逐行割线更新。完整24项epigraph已在旧实验中尝试，本轮不把它或BFGS本身称为新方法。
恢复余量按rad等价比较：inner/braking权重1，terminal乘.1s；目标为原buffer之上
1e-5rad等价余量。该值是优化目标，不是已知模型误差界或物理安全保证。

余量比例r<.9进入恢复，目标r>=1；否则优化全部24项最大平方超限F。原安全buffer
1e-8始终硬约束，pose阶段还保留部分余量。数值搜索允许前缀暂时退化，但每项
平方误差上限固定为`max(initial_q_i,1)*1.01`，不逐步放宽，也不改变原执行caps。
接纳要求实际原安全、固定搜索上限与足够的同模式改善及改善比；仅完整原witness
才可被选中。初始和末次repeat收费且exact，数值anchor与执行解分开记录。

## 结果与停止原因

最终数值anchor为response32，仍通过原安全和本轮固定搜索上限，没有完整pose资格。

| 指标 | 初始3557（新response0） | 最终anchor32 |
| --- | ---: | ---: |
| 最大归一化平方误差 | 6.9087100408 | 6.7674856364 |
| 最大平方超限F | 5.9087100408 | 5.7674856364 |
| 归一化平方误差总和 | 29.6216064118 | 29.6531230004 |
| cursor9最大平方误差 | 1.3145505007 | 1.3266316535 |
| 最大位置误差/m | .01886907082 | .01896317700 |
| 最大旋转误差/rad | .13142212562 | .13007195736 |
| 原连续达标前缀 | 9 | 4 |
| 原双caps合格点数 | 9/12 | 5/12 |
| 原安全最小slack | 2.235571217e-8 | 9.186944087e-6 |
| rad等价目标余量比例r | .0001235571 | .9176944087 |

最坏超限虽下降，其余指标没有全面改善；不能用恢复次数或单一最大误差下降宣称
可靠跟踪。最终terminal为9.206433535e-5rad/s，安全余量只属于当前预测模型。

生产共44响应、43唯一命令，包含首初始响应、42次实际候选及末次初始repeat。
15次接纳均按预登记条件通过，27次实际准入失败：21次仅固定pose上限失败，5次
安全与上限同时失败，1次仅安全失败。末stage18/19的10条候选均通过安全buffer，
却全部越过cursor11平移的固定搜索上限；其F实际/预测改善比约1.066–1.129，
后期拒绝并非改善比不足。

18/21个QP合格，3个保留为不合格尝试。stage10/17达到100迭代，stage20为求解器
line-search状态8；这些点的KKT数值指标通过，原规则仍同时要求solver success，
所以没有事后放行。全部21个证书与状态机由独立审计重建，不把失败QP漏计。

最终半径减至7.8125e-5，低于固定最小.00015625，终止为`radius_below_minimum`。
剩余78响应和3个stage预算没有追加，也没有自动重启同一搜索。首响应与历史完整
11个数值字段exact，末初始repeat与本轮首响应exact。

## 独立复算与来源

独立检查从原保存数据重建每个QP/KKT、逐行因果割线、预测/实际改善比、接纳、
半径、预算与repeat，再用既有独立native路线完整复算44次。原九组audit、300内步、
12端点/goal及误差、strict分类和选点完全一致，分歧列表为空；没有重新求解QP。
跨路线最大差异：安全2.931872410e-15，归一化平方误差3.104183577e-13，
torque1.509903313e-12。390处JIT/Python辅助goal差异最大8.940696717e-8另记，
不混为跨路线复算误差。独立路线自身完整初始repeat亦exact。

QA为6个专项模块、138 passed in 17.89s，绑定此前119来源和本轮9来源，共128。
这不表示重跑全部历史测试。四项准入为Ruff、格式检查、两个真实输入dry，均无QP或
模型响应。生产session98466/子进程1678978与独立复算session30547/子进程1702060
均实际wait/reap0，无信号、自动重试。最终纯数据分析核验2556来源不变，无新响应。

| 证据 | SHA256 |
| --- | --- |
| `outputs/interior_trajectory_search_qa_20261029/result.json` | `5a2cfb25be4d896d391a9618b84540fb8eb03e759c9d373c40ced4f2c656dd92` |
| `outputs/interior_trajectory_search_admission_20261029/result.json` | `2b3e9935a600e852678c03908a5202e5b5f2cdac04118833eaa5b0fa640e5206` |
| `outputs/tracking_v5_interior_trajectory_search_20261029/verified.json` | `8e3a84db8e73ed655ceefe97cc52aa62eb5f0725220e9e4f51cc363458e16350` |
| `outputs/interior_trajectory_search_producer_20261029/result.json` | `c712e99a2f2d3b51b0ce553bc4df4715bb0f32322ed38d6b490c130711e7c611` |
| `outputs/tracking_v5_interior_trajectory_search_check_20261029/verified.json` | `93318aed383719c9f01406ac038bdbcdec0dc420d9385dfc8f786b631150a9e1` |
| `outputs/interior_trajectory_search_checker_20261029/result.json` | `7070844a22d04e1a2217168847faa41d870e54fb1c63a2bbd1d1f5dfa582b777` |
| `outputs/interior_trajectory_search_analysis_20261029/result.json` | `e80df4508072b2052f7864832e51bd2fd5c9f32e13495a4147e3be1571279a50` |
| `outputs/interior_trajectory_rejected_ray_diagnostic_20261029/result.json` | `84e33538ee8b1ae49371b9b3331127102299855512a8c9f5a42f2d04fcaf0049` |

## 保存拒绝方向的后验诊断

纯数据脚本`outputs/interior_trajectory_rejected_ray_diagnostic_20261029/analyze.py`
（SHA256 `f86e733289d75c02ce1e376c5b401a27f1b1ca47f2a50a8d90d13e002d17e905`）
从原72差分和15次接纳重建局部Jacobian，所有保存割线metadata逐值exact、未来列
仍为0。对stage19的cursor11平移行，以alpha为1、.5、.25的已存响应拟合过原点
线性预测残差：`beta=sum(alpha*residual)/sum(alpha²)=.00012338690081822837`。

另外两个未参与拟合的已存alpha .125/.0625，原残差为1.551134754e-5和
7.489327244e-6，修正后为8.798494122e-8和-2.223540572e-7；最大绝对值从
1.551134754e-5降至2.223540572e-7，两个点均改善。它提示已付费拒绝响应含有
可用于约束方向预测的信息，不能据此保证完整Jacobian更新或后续搜索必然成功。

stage和误差行是在看到失败后选择的，五个点都已被观察；这不是预登记的留出泛化、
误差界或新独立动力学证据。没有采用新Jacobian或发出新命令，所有原失败标签不变。
脚本actual exit0，15份引用结束重核不变，新增QP/模型/响应/查询全部0。

## 后续证据与限制

最后两阶段在同一anchor32使用相同未更新Jacobian，只缩半径；这些已收费却未接纳
的响应没有参与割线更新。cursor11平移的上限预测失真已成为当前主瓶颈，应先用
保存的分alpha响应检验可否改进该方向预测，再决定是否把拒绝响应纳入局部模型更新。
这与继续增加安全恢复次数不同，也不授权放宽固定搜索上限或原caps。

本轮没有证明完整目标不可达，也没有取得新实际控制、world-model或留出任务资格。
固定搜索上限并非带历史的filter，恢复仍可能撤销先前pose改善；未来必须同时考察
完整caps、达标点数、净误差和实际控制，而不以局部优化器内部指标替代总目标。
