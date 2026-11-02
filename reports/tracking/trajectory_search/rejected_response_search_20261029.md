# 安全拒绝响应的同anchor割线反馈

2026-10-29：124项专项CPU QA、四项真实入口准入、52响应生产与52响应完整独立
复算均正常关闭。最坏平方超限较前项下降约.1698%，但cursor9和总误差略差，原达标
前缀仍4、双caps合格仍5/12，完整witness0、selected为空。没有新的可执行轨迹，
实际仍cursor8/task0/grasp0，总目标未达成。

实现提交`82840dd`。本轮生产求解24个QP、770内部迭代，20次接纳（11恢复、9pose）
及3次拒绝响应反馈。生产与独立路线各52完整响应、15600数值内步、16225查询；
合计104响应、31200数值内步、32450查询。新增物理积分、world-model调用、fresh
验收与参考消费均0。GPU继续禁用，原18正确参考/12WAM开发/24最终留出门槛保持。

## 唯一机制变化与固定条件

[执行前计划](../plans/v0.0.1/rejected_response_search_20261029.md)规定：只有整stage
无接纳时，才从本stage已计费、原audit通过、全部slack>=1e-8、确实越出固定搜索
pose上限的实际拒绝响应中，按最小alpha选择反馈来源。使用该候选相对当前anchor
的命令和实际输出差，更新原有界因果逐行Js/Jq；anchor命令、实际截距及参考不动，
半径仍按失败规则减半，不增加callback、FD、QP或预算。

这不是在拒绝点重新设定anchor，也不把拒绝点选作执行解。已接纳stage只保留原接纳
更新；不合格QP、unsafe、ratio失败、无完整响应或过小步不能提供新反馈。

[新配置](../configs/rejected_response_search_20261029.json)与前项数值配置逐字节
一致：初始3557与原72条signed1e-4差分，seed42，同actual368实测q/v、completed367
命令与rotation latch、固定finger和历史profile，以及当时已生成的原12参考。
输入封装SHA256为`eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023`。
旧代码、旧结果和后验诊断均保持原SHA；诊断beta和旧拒绝响应未预加载到新模型。

原位置.01m、旋转.05rad、box/rate.25、关节margin.05rad、braking.1s、terminal方向
及buffer保持。固定搜索上限仍`max(initial_q_i,1)*1.01`，不逐步放宽；恢复、实际/
预测改善比、半径、完整witness、历史exact及repeat规则均保持。每个完整响应仍重置
同一初态，计算300数值内步和312查询；模型中的固定finger、对象与接触等限制保持。

## 与前项比较

共同stage0..14（忽略新增反馈字段）、完整entry/request0..27均与前项逐值exact；
首次变化是stage14末的拒绝反馈。以下比较最终数值anchor，而非执行选点。

| 指标 | 初始3557 | 前项anchor32 | 本轮anchor40 |
| --- | ---: | ---: | ---: |
| 最大归一化平方误差 | 6.9087100408 | 6.7674856364 | 6.7576952531 |
| 最大平方超限F | 5.9087100408 | 5.7674856364 | 5.7576952531 |
| 归一化平方误差总和 | 29.6216064118 | 29.6531230004 | 29.6553630360 |
| cursor9最大平方误差 | 1.3145505007 | 1.3266316535 | 1.3276763736 |
| 最大位置误差/m | .01886907082 | .01896317700 | .01896317524 |
| 最大旋转误差/rad | .13142212562 | .13007195736 | .12997783708 |
| 原连续达标前缀 | 9 | 4 | 4 |
| 原双caps合格点数 | 9/12 | 5/12 | 5/12 |
| 原安全最小slack | 2.235571217e-8 | 9.186944087e-6 | 9.223801942e-6 |
| rad等价目标余量比例r | .0001235571 | .9176944087 | .9213801942 |

本轮F比前项少.0097903832，总误差与cursor9误差略增。最终terminal为
9.371869428e-5rad/s。全部余量只属于当前局部预测模型，不构成物理安全或稳定跟踪
保证。不能用更多接纳或反馈次数替代完整caps、实际闭环和留出任务验收。

| 反馈stage | 来源response | alpha | 保持的anchor | 后续情况 |
| --- | ---: | ---: | ---: | --- |
| 14 | 27 | .0625 | 22 | 后续搜索继续接纳新点 |
| 22 | 45 | .0625 | 40 | 下一stage的cursor11平移上限违反消失，但cursor9旋转上限失败 |
| 23 | 50 | .0625 | 40 | 已达阶段上限，没有后续QP消费该更新 |

3次均独立重选为最小合格alpha，anchor及实际指标exact不动、半径减半。stage22/23
的10条实际候选都安全；后一stage只是切换了被违反的固定pose上限，完整解仍为0。
不能称3次反馈都带来推进，也没有证明得到完整Jacobian或通用误差界。

23/24个QP合格；stage10保留原100迭代solver失败，没有事后放行。终止原因为
`stage_budget_exhausted`，剩余70响应预算未追加。前项实际44响应/21stage，本轮
52响应/24stage：相同的是预算上限，实际成本不同。每路线51唯一命令，初始与末次
初始repeat都收费，repeat不算新推进。未自动重启任何已关闭搜索。

## 独立核验与执行收据

独立ledger从完整entries重新选择反馈来源、重建因果Js/Jq及下一stage的全部QP/KKT，
核对anchor、半径、所有实际接纳、预算和计费；没有重新求解QP。独立native路线
完整复算52响应，逐条核对300内步、全部九组audit、12端点/goal、TCP误差与strict
分类，再用新鲜数值重新判定接纳、反馈、完整witness和repeat。分类分歧为空；历史
完整11字段与本轮首响应exact，首末repeat exact，独立路线自身repeat亦exact。

跨路线最大torque差异1.509903313e-12，平方误差差异2.624567230e-13；464处
JIT/Python辅助goal差异最大8.940696717e-8另记，不混为跨路线复算误差。

QA为6个专项模块、124 passed in 37.54s，冻结137个来源；不表示重跑全部历史测试。
Ruff、格式检查及两个真实输入dry通过，dry均0QP/响应/查询。QA session58654、
准入24960、生产94552/子进程1896765、checker34727/子进程1909328均实际wait/reap0，
无信号或自动重试。生产结束2571来源不变，checker结束2573来源不变。

保存数据比较的v1脚本在末尾来源关闭失败（session7662实际exit1）：继承的相对路径
声明被直接传给要求绝对路径的检查器。纯数据定位确认是路径表示问题；原脚本与
失败收据保留。隔离v2仅规范仓库相对路径，保留110条原声明→绝对路径映射及原SHA，
不改数学、指标、成本或任何生产/checker源码。v2 session86556实际exit0，重核
2600来源全部不变，无新QP/模型/响应/查询。未重跑生产或独立复算。

v2脚本`outputs/analyze_rejected_response_search_v2_20261029.py`的SHA256为
`f3acc75ec8ba641915caa0ccebc8d18175f78e462f2c54ea16462fa730f0459a`；
v1脚本仍为`cd5864b6712ae688a7bd45722938b66f1b927496e97caf3ac76c547ac5ea5099`。
分析确认共同28条搜索响应及15个stage在前项字段中exact，首差异为response28/
stage15；含末次初始repeat的29条命令匹配响应，其完整内容均exact一致。

| 证据 | SHA256 |
| --- | --- |
| `outputs/rejected_response_search_qa_20261029/result.json` | `3ea79210540d70eff3d1f542676e4826ea69f280d2dd323470fbcaf6d458bb06` |
| `outputs/rejected_response_search_admission_20261029/result.json` | `d39511c44be980375e9bc0ca8d6dd03937452203abc374a63a7496c52a0232d6` |
| `outputs/tracking_v5_rejected_response_search_20261029/verified.json` | `915feac847ea22b4bc36c359cd6f596a3148787845a6532dc5ee9eab0cc62d9e` |
| `outputs/rejected_response_search_producer_20261029/result.json` | `43e9661dd0c57ca524a91e5530ac6894725dc26fd7842f4e94ead2cbb47627c6` |
| `outputs/tracking_v5_rejected_response_search_check_20261029/verified.json` | `cdf2c1e726fb2cd9f5f61860488980901b87437edfb4560e942c783cb2a51dfc` |
| `outputs/rejected_response_search_checker_20261029/result.json` | `6309a1558a2cc71dece89875c07e4383f01a4f677d1188f4d9e1577162fde296` |
| `outputs/rejected_response_search_analysis_failure_20261029/result.json` | `fc184b63b62e00980b93e8645072940b179e6aba11520fdc5cd5813f1b905f03` |
| `outputs/rejected_response_search_analysis_v2_20261029/result.json` | `e98ef0306c50fe4003ce9539b73c991f946a8639c30e9890f947c99811e9d5eb` |

## 后续判断

已收费的安全拒绝响应确实可改变后续局部模型与搜索，但没有扩大原caps合格范围。
固定搜索上限附近的误差会在不同项目之间转移，单方向预测改善不足以恢复完整轨迹。
后续应先结合保存的全轨迹约束残差分析该权衡，再独立登记有限新方案；不在本轮
剩余预算上继续搜索，也不放宽原执行caps或宣称目标不可达。固定搜索上限本身属于
本数值算法，不是已证明必要的物理可达边界。原总目标保持active，GPU禁用保持。

双语PR原文见[PR说明](pr_rejected_response_search_20261029.md)。
