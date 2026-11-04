# reserve接纳有限迭代的匹配对照

2026-11-02。预登记9cf0a1d、实现f4776d5。实际新响应对照已完成，完整独立审核
通过；reserve门单独启用没有改善本例完整跟踪，不能设为优胜默认方案。

## 同起点、同模型、同原域的实际结果

两组使用同一anchor85、新Js/Jq/Je、完整cap-distance目标、原60stage/122response
上限与原根global/pose ceiling。除显式reserve接纳外配置相同，独立模型/控制状态
和输出并行执行。初始loss2.0100673083、caps12/24、prefix2、reserve .99706666。

| 指标 | baseline | reserve |
| --- | ---: | ---: |
| 最终完整cap loss | 2.0076879552 | 2.0098115464 |
| 最终sum q | 30.0049728466 | 30.1054435621 |
| 最终caps | 12/24 | 12/24 |
| 最终prefix | 2 | 2 |
| 完整witness | 0 | 0 |
| 最终joint/buffer安全 | True | True |
| 最终min raw slack | 5.58285856e-6 | 9.29584511e-6 |
| 最终reserve | .557285856 | .928584511 |
| stage / 接纳 | 60 / 49 | 7 / 1 |
| pose / restore stage | 36 / 24 | 7 / 0 |
| 拒绝反馈更新 | 0 | 0 |
| 停止 | stage_budget_exhausted | radius_below_minimum |

reserve组保护了余量、没有restore，但只完成首步下降。随后24个实际响应因原
reserve进入条件被拒绝；原最小预测下降还阻止了8个更小候选的调用，均如实记录。
后续stage没有原pose ceiling失败资格的rejected-source，因此持续缩半radius而
没有拒绝反馈更新。没有减少目标/下降门或放宽reserve来制造继续推进。

baseline仍有24次restore，下降幅度也很有限，不能用它优于本变体宣称原算法已
解决问题。两组最末旋转误差仍约.1311rad，高于原.05rad；未达完整跟踪或任务门。
这次结果支持优先检查安全reserve拒绝数据未用于模型更新的缺口，但尚未证明
加入此反馈就能收敛；也未证明全局不可达。

## 全部实际计费及独立证据

| 费用 | baseline | reserve | 合计 |
| --- | ---: | ---: | ---: |
| 生产完整响应 | 91 | 28 | 119 |
| 独立完整复算 | 91 | 28 | 119 |
| 两侧总响应 | 182 | 56 | 238 |
| 数值内步 | 54600 | 16800 | 71400 |
| 动力学查询 | 56786 | 17474 | 74260 |
| model | 2 | 2 | 4 |
| QP / NNLS | 60 / 60 | 7 / 7 | 67 / 67 |
| QP迭代 | 884 | 168 | 1052 |

所有费用在登记上限内；reserve组未使用的阶段/响应不转给baseline或自动续跑。
旧148响应模型采集成本仍独立保留，不藏入本轮计费。本轮0新stencil拟合；实际
accepted causal secant更新由原ledger记录。0物理/GPU/WAM/fresh/最终留出/实际
参考消费，所有候选未下发，原安全失败响应不升级为控制资格。

先以独立GN/KKT/scaling/recovery/状态机审核生产答案，再逐响应完整native重建并
检查原1e-10/torque1e-8、布尔audit。决策相关metric及完整响应逐值一致，使用独立
metric替换producer记录后再次重建审核全部新决策与反馈/更新，未调用旧不支持新
拒绝原因的fresh checker冒充支持。完整历史anchor与repeat exact，来源/源码前后
不变。两次独立ledger均保留在各result中，完整native响应保存independent目录。

基线tool session45896与reserve95857均退出0并实际wait/reap；PID210041和210050
已确认不存在。各组1800秒TERM/10秒KILL上限、阶段alarm未触发，无重试。
CPU空CUDA/LD、数值库线程1。两组相互独立并行，不跨组共享模型更新。

最终40项jobs32专项全部通过，0skip/collection错误、worker回收、来源前后不变。
此前同40项初测通过后仅整理import并更新QA绑定，最终轮单列。Ruff/format与
Git空白检查通过，未运行全仓回归。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| baseline完整结果 | reserve_entry_search_20261102/baseline/result.json | 58b65f7f18ff59364a409a019d78b0411b7d8328e81c00c4d5a8d43c9c3d553b |
| reserve完整结果 | reserve_entry_search_20261102/reserve/result.json | 5777638f29cb252ae28137d37cb616d150d477cac8a2c7eb27c5acfd8b4ac1ef |
| 成对保存分析 | reserve_entry_search_20261102/paired_analysis.json | 3e286760371207f914ffa7709c0325122ecaacb644a8834a41dfd342a1508148 |
| owner终态观察 | reserve_entry_search_owner_20261102/observation.json | ca8c18e994a3b8df83a19c401723b42348f643cc168c3a2ba7119c5cfbc25eb1 |
| 最终40项QA | qa/reserve_entry_search_final_20261102/result.json | 26571d3711cb9087b5b65a2541313b9fb1cfbf1535094c20fbafba72d412c807 |

下一步先核查已计费、安全但reserve拒绝的响应对局部模型校正的作用，再显式设计
新的反馈资格与独立审核。旧反馈政策、接纳记录和本轮失败保持不变；任何新响应
预算另登记。原18/12/24及稳定20/24、每task4/8总验收仍未完成，最终留出关闭。
