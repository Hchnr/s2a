# 60阶段与同122响应上限的CPU预算对照

2026-10-29：142项专项CPU QA、四项真实入口准入、90响应生产及90响应完整独立
复算均正常关闭。新旧前24完整stage/QP、34条完整搜索entry/request逐值exact，
末次初始repeat另比exact，预算对照资格通过。最终最坏平方超限F比24阶段再降
6.5090%，总误差却增加11.0702%、双caps点6→4，完整witness0、selected为空。
增加阶段有数值收益，没有取得完整轨迹、实际控制或稳定任务成功。

实现提交`628b2ba`。60次QP尝试中57合格，1362内部迭代；55接纳含28恢复、27pose，
拒绝反馈0。两路线合计180完整响应、54000数值内步、56162查询。新增物理积分、
world-model调用、fresh及参考消费均0，实际仍cursor8/task0/grasp0。GPU继续禁用，
原18正确参考/12WAM开发/24最终留出及完整总目标保持。

## 唯一变化与固定合同

[执行前计划](../plans/v0.0.1/stage_budget_search_20261029.md)与
[配置](../configs/stage_budget_search_20261029.json)只将max_stages的24改为60，
其余数值与文件内容逐字节保持。每QP最多100内部迭代，最坏QP预算2400→6000，
响应上限仍122且含首初始和末次初始repeat。这是增加QP预算的对照，不能称总
预算不变或60×5+2=302响应，也没有将原87响应余量接到旧运行。

新namespace仍从3557/原72条signed1e-4差分、空seen、初始半径.005开始，没有
从旧anchor33或优化器恢复。输入封装SHA256仍为
`eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023`。
同actual368实测q/v、completed367命令与rotation latch、固定finger和历史profile，
仅用当时已生成的原12参考，seed42/20Hz/2ms与原控制器保持。

搜索主体AST与前项完全相同；新validator只扩大阶段整数上限，独立薄wrapper
约束完整原配置、stage<=60和response<=122后复用冻结完整审计，不修改旧模块。
共同上限仍固定6.977797141162663，原余量恢复/pose模式、ratio、半径、有界因果
割线、原原因分类和最小alpha反馈、停止顺序与repeat资格保持。

原位置.01m、SO(3)旋转.05rad、arm command box[-1,1]、周期间rate<=.25、关节
margin.05rad、braking.1s、terminal方向、原audit和1e-8 buffer不变。buffer只施于
原4369维inner/braking/terminal safety_slack，box/rate保持原audit数值容差，允许
合法零slack。所有12点须满足原双caps才可授予数值
witness，不能把共同数值搜索上限当作执行caps。局部模型仍固定finger与对象，
接触等遗漏保留；独立算术复算不是新完整物理仿真或闭环任务证据。

## 真实共同前缀

新旧首24份完整stage（含QP/证书/因果更新/半径）及旧entries/requests0..33共
34条完整搜索响应exact，search AST亦exact。旧index34是强制末次初始repeat，
新此处继续stage24，不把它误作连续前缀。旧repeat34与新repeat89完整内容另比
exact；共35条匹配命令的完整响应均exact。阶段23边界仍anchor33/radius.00125。

新运行重新计算并支付这些共同响应，没有把旧响应缓存成免费调用。新增阶段
24..59共36个，其中34接纳；两条路径是实际同算法前缀的预算对照，不是后验
拼接局部反事实。metadata中的24/60当然不同，未据此要求整个answer相同。

## 全轨迹结果与实际成本

| 指标 | 24阶段anchor33 | 60阶段anchor88 |
| --- | ---: | ---: |
| 最大归一化平方误差 | 5.8612277489 | 5.5448089226 |
| 最大平方超限F | 4.8612277489 | 4.5448089226 |
| 归一化平方误差总和 | 33.8249384189 | 37.5694326944 |
| cursor9位置平方误差 | 2.4626932850 | 2.6719637726 |
| cursor9旋转平方误差 | .9693435820 | .9667788283 |
| 最大位置误差/m | .02399140931 | .02354741795 |
| 最大旋转误差/rad | .12104986317 | .11756541309 |
| 原连续达标前缀 | 4 | 4 |
| 原双caps合格点数 | 6/12 | 4/12 |
| 原安全最小slack | 9.997928465e-6 | 9.993804517e-6 |
| rad等价余量比例r | .9987928465 | .9983804517 |
| 完整witness/执行选点 | 0/无 | 0/无 |

本轮只有cursor0..3通过原双caps，不能用55次接纳称实际推进。cursor9旋转虽
达标，位置约16.35mm仍失败；最坏项下降伴随总误差和原达标节点退化。初始3557
原有前缀9、双caps点9/12，不能只选对本轮有利的24阶段比较。最终terminal为
9.990248273e-5rad/s，余量只属于当前预测模型，不是通用误差界或物理安全保证。

| 计费项 | 24阶段 | 60阶段 |
| --- | ---: | ---: |
| 每路线完整响应/唯一命令 | 35/34 | 90/89 |
| 每路线数值内步/查询 | 10500/10921 | 27000/28081 |
| QP尝试/内部迭代 | 24/666 | 60/1362 |
| 接纳恢复/pose | 11/10 | 28/27 |
| 剩余响应/阶段预算 | 87/0 | 32/0 |

本轮首初始+88实际候选+末初始repeat全部计费。55接纳以外33实际拒绝：26因
原安全失败，4实际改善量不足，3ratio不足；无共同搜索上限拒绝、无合格拒绝反馈。
57/60个QP合格，3个不合格仍为stage17/20/21的原互补证书超1e-6，solver均成功；
前24stage后没有再出现QP不合格，未重算或放宽任何证书。

增加的36阶段实际多用55响应、696QP内部迭代，F净下降.3164188263而loss增加
3.7444942754。最后10stage仍降F .0672759303，最后5stage仅降.0207002575，均
实际接纳；末半径.005，未因最小半径停止。最终命令相对初始Linf变化.062438328，
未碰±.1人工位移上限。终止为stage_budget_exhausted，32响应余量未追加。

这些结果不能证明不可达，也不能将仍有下降外推为继续增加预算就会通过全部caps。
阶段限制解释部分数值进度；它没有解释或解决总误差增长与完整资格缺口。下一步
应先用保存响应检验完整state向量预测及目标分配，不自动继续扩大同一maxF搜索。

## 独立复算与来源关闭

独立ledger复用冻结global审计，完整重建60个QP/KKT、模式、全部接纳/拒绝、因果
更新、半径、预算与repeat，无额外QP求解。native独立路线逐条重构全部90响应、
九组audit、300内步、12端点/goal与TCP误差，新鲜数值再次判定原安全、共同搜索
域、实际改善/ratio、反馈、原fullcaps和repeat；严格分歧为空。历史完整11字段
与新首响应exact，首末repeat exact，独立路线自身repeat亦exact。

跨路线最大torque差异1.683986284e-12，平方误差4.334310688e-13；829处
JIT/Python辅助goal差异最大8.940696717e-8另记，不混为跨路线复算误差。

QA为6模块、142 passed in28.47s，146旧来源加9新来源共155，不表示重跑所有历史
测试。Ruff/格式及两条真实输入dry通过，均0QP/响应/查询。QA session82677、
准入39754、生产10863/子进程2149546与独立64649/子进程2167826均实际wait/reap0，
无信号或自动重试。生产结束2641、checker结束2643来源不变。

纯保存数据分析session6243实际exit0，2658来源核验关闭；脚本
`outputs/analyze_stage_budget_search_20261029.py`的SHA256为
`6710b98f6f0690568a44df949bf60cc81d76748767d56d3bc710443c2a2f56b1`。
没有新QP/模型/响应/查询；旧大归档只做流式SHA，不解析旧3657响应。

| 证据 | SHA256 |
| --- | --- |
| `outputs/stage_budget_search_qa_20261029/result.json` | `d0fd3e29d96f5ac2981919824c4d7400d5ffc77039122f3568590245339533e2` |
| `outputs/stage_budget_search_admission_20261029/result.json` | `2b77be8e7535576c3132e4fbef4fcbf6c72d98cc41b1b3fff5203cb62663cb48` |
| `outputs/tracking_v5_stage_budget_search_20261029/verified.json` | `9d2667e102a93020ba2eaa819be252c1c00d7f3c031532822d4385b38783e25b` |
| `outputs/stage_budget_search_producer_20261029/result.json` | `9b40d01dd65402dd992038b9381522b30ec9c3178f548ddc747134498e503f03` |
| `outputs/tracking_v5_stage_budget_search_check_20261029/verified.json` | `c9b2ec12fefefae3ea1a680e4cdb70f1b07e45fc28b96559c415083371c63317` |
| `outputs/stage_budget_search_checker_20261029/result.json` | `dfca59bd4b7da65faaf8253c71eb4bbb3a7229835885747320dd03711bedae07` |
| `outputs/stage_budget_search_analysis_20261029/result.json` | `5e6ceea748c2d346d76d2265ddfe0c19f99ef65056be1d21e2abe16ab13a0767` |

## 实际前向验证仍关闭

本轮无数值witness，不派发新动作。未来若完整witness通过独立复算，须用新的隔离
物理producer/checker自然重放368周期恢复完整controller/contact/latch历史，再
执行固定12命令；仅set_state(actual368)不够。每路线最多380控制周期/9500积分，
所有前缀也计费。需从实际日志重建原九组安全，terminal仅在完整cursor11端点，
不能错误地在每个partial prefix套终端门；audit内部比较容差亦不能替代原4369维
safety_slack的1e-8 buffer，box/rate仍使用原audit容差。
原始依据见`src/s2a/native_runtime_replay.py:125`、
`src/s2a/joint_preview_constraint_audit.py:53`及
[已关闭运行时回归](native_runtime_replay_20261029.md)。即使该固定段物理通过，也
不等于多任务、在线重规划或最终fresh资格，原完整总目标保持active。

双语raw Markdown见[PR说明](pr_stage_budget_search_20261029.md)。
