# 保存方向的固定回缩检验（执行前登记，CPU，尚未执行）

总目标及原全部控制/任务验收保持，GPU继续禁用。[保存差分诊断](../../reports/saved_trajectory_differences_20261028.md)已确认原cursor9可行基点配对H12/early/late为10/5/1；这些方向的线性target预测全部仍超原caps，且主分析没有L∞≤.01的小步样本。不能据此认为缩步已能解决cursor9。本轮只检验同一保存方向上的较短步，是否比原完整步更能保住前缀、完整安全约束并改善target。

## 固定输入和硬前置

唯一选择来源为`outputs/tracking_v5_saved_differences_20261028/result.json`，SHA `f918aaf2709fd3bce09ef90d32e92a5f7c255bb753e502c9caea3fb6d954e5db`；读取其全部`main_subset=true`配对，保持原case及base顺序，不挑选成功方向、不加入其他基点。绑定该诊断的注册、源码/计划、QA和三份原producer结果。正式执行必须等待H12、H13 early和late的完整独立checker均正常actual wait/reap0、无信号和来源未变；严格比对其producer SHA与全部预算、分类，不能以文件存在或进程消失代替关闭。

每条轨迹从同actual368 q/v、固定手指状态、已完成actual367的previous command与旋转latch重置；仅递推当前候选预测状态，不读取实际369及之后反馈。保持世界坐标、TCP、50ms控制周期、25个2ms内步、逐内步更新M/J、原生goal语义、力矩裁剪和implicit damping Euler。H12使用其原12目标；H13使用各自冻结的13相位目标，同时单列原时序与原节点指标，不跨合同改写成功标准。保持原关节界、profile、制动时间.1s、terminal、增益、tool和夹爪，不引入新随机初态。

## 唯一采样网格与预算

每个方向固定`alpha=[0,.0625,.125,.25,.5,1]`。alpha0和1分别直接复制保存base和trial命令，以保留端点精确bytes；内部点只计算`base+alpha*(trial-base)`。不裁剪、投影、snap、改变rate、改变target、不自适应补点或提前终止成功方向。即使某点违规也保留并明确标记。每个alpha都从原actual368重新响应，不能沿alpha或方向累积状态。

| case | 方向数 | 完整响应数 | 新独特命令 | 数值内步 | forward queries（含该case一个current） |
| --- | --- | --- | --- | --- | --- |
| H12 prefix | 10 | 60 | 40 | 18000 | 18721 |
| H13 early | 5 | 30 | 20 | 9750 | 10141 |
| H13 late | 1 | 6 | 4 | 1950 | 2029 |
| 每侧合计 | 16 | 96 | 64 | 29700 | 30891 |

执行前精确bytes枚举已确认96条批内与跨case无重复；与三份旧trace的交集恰为32个端点，64个内部点均未保存。这里端点重算也计完整响应，不能把它们记成免费缓存。producer与独立checker各完整重构96条，合计192响应/59400内步/61782查询，新增独特命令仍64。没有优化迭代或新搜索预算，物理/世界模型/fresh均0。

最小alpha使15/16方向的L∞≤.01；early的base2621→trial2700仍为约.014797。保持固定网格，不依据结果再加更小alpha，也不把未覆盖小步当作已检验。

## 保存、判定及比较

所有96条保存完整commands、endpoint及inner q/v、FK、torque/clipping、goal递推和九组raw slacks。分别记录原动作/关节审计可行、原hard前缀严格通过、非线性minimum≥1e-8 buffer、target两项及max归一化平方误差、全部H caps、完整witness；原位置.01m、旋转.05rad不变，box/rate审计仍1e-12、动态组仍1e-10。短步target改善定义为`base_target_max - trial_target_max > 1e-12`，仅是诊断下降条件，不放宽原控制caps。

主结果为每个方向、每个alpha的完整表，以及各方向是否存在内部alpha同时保持原hard前缀、原动作/关节资格、1e-8 buffer并改善target。另列真正达到target及完整H witness的数量。每case和全表都报告失败，不将不同alpha或方向当作独立任务重复来推断成功率。没有符合全部条件的点就如实记录；不按最好alpha换掉整张表。

alpha0/1要对旧trace已有字段复核：commands精确bytes，TCP errors及safety minimum误差≤1e-10，严格prefix/joint/feasible标签必须一致；差异不能靠重新贴标签或放宽caps消除。旧trace未保存完整slack向量或每条q/v/torque，不能声称这些字段与旧存档相同。新完整向量由本轮producer与独立checker比较：q/v/pose/slack≤1e-10、torque≤1e-8，rtol0；严格资格标签独立一致。

若回缩恢复部分方向的可行下降，只支持另行设计和检验受限步长及真实响应验收机制，不证明其会找到cursor9完整解。若未恢复，不证明所有更小alpha、其他方向、全局优化或物理控制不可行。标量minimum变化不替代完整活动约束分析；完整新slacks仅描述当前局部模型，不是实际机器人安全证明。

## 实现与执行准入

新增独立的producer/checker/owner入口和必要合成测试，旧执行源码/测试/计划全部冻结。producer复用原冻结trajectory/dynamics/audit数学；checker不导入producer/scorer，使用已核验的独立native AST/JIT query与递推数学，在新文件中参数化H12/H13并证明其数学来源未改变。每侧预算与查询计数严格检查，独立状态重置、端点bytes和无自动加点用合成需求测试验证。

先完成原生Python3.11/NumPy1.26环境QA、Ruff/format、实际CLI dry/source admission，保存输入/源码/计划前后SHA并提交，再唯一启动。本批统一使用显式空CUDA/LD、数值库单线程和seed42，无随机采样。全新输出目录`outputs/tracking_v5_saved_direction_backtracking_20261028`与`outputs/tracking_v5_saved_direction_backtracking_check_20261028`拒绝覆盖；对应新owner保存launch、实际wait/reap与来源不变凭据，时限分别1200s和1800s，无自动重试。

本计划尚未产生新响应。即使发现完整局部模型witness，也不能自动进入物理pilot；原实际任务仍止于cursor8、task/grasp0。后续物理及独立重放需另登记，原18条正确参考、12条WAM开发、最终24条留出新初态门槛与总目标均不变。
