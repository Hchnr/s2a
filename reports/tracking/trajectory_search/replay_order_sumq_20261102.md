# 同预测器sum-q对照结果（2026-11-02，Asia/Shanghai）

## 结论

使用新replay-order响应、native_solve TCP评分及新独立验证的初始化/72探针，
原sum-q搜索完成122响应。独立checker完整复算全部响应，完整KKT/反馈/严格
fresh决策账本通过，响应逐值一致，原1e-10比较未放宽。

完整H12 witness仍为0，数值prefix仍9、caps仍18/24，实际控制资格仍未获得。
总q下降完全来自已达标前九点；后三点和最大q反而上升。数值审核通过不等于
可靠跟踪。下一步是同一预测器、同一新矩阵和预算下的cap-distance对照。
原18/12/24、最终20/24且各任务4/8的验收保持，最终留出未开启。

## 执行与审核

- 算法调用未修改的state_vector_dual_recovery.search，完整原配置逐字段保持，
  原restore/pose、QP/NNLS证书与预算、拒绝/缩域及反馈状态机保留。
- 先写新registration、代码/输入SHA；独占新目录，无覆盖或自动重试。
- 生产122响应、121唯一命令、36600内步、38065查询；独立checker同样122响应、
  36600内步、38065查询。合计244响应/73200内步/76130查询/2model。
- 实际55阶段/55QP、484次QP迭代、55次NNLS调用，其中54QP通过原证书；单次
  证书失败沿原拒绝机制保留。因响应预算耗尽停止，最终anchor为response120，
  response121是最后的初始重复。实际费用未追加到旧实验。
- 所有响应均审核，完整KKT账本通过；fresh严格决策通过、无类别分歧，初始
  重复exact；9次拒绝反馈更新独立重建通过，没有新增响应调用。
- 来源前后核验通过，执行进程退出0并已回收。登记后耗时203.654秒，不含前置
  来源加载。0物理步、0WAM、0GPU、0新场景、0实际新增参考消费。

## 跟踪误差分布

以下都是局部模型完整响应，不是独立物理或真机证据。

| 指标 | 初始response0 | 最终anchor120 |
| --- | ---: | ---: |
| total q | 29.62160641175904 | 27.469554738654622 |
| state loss（0.5 sum q） | 14.81080320587952 | 13.734777369327311 |
| 前9点q | 10.819406810407006 | 8.512757678431187 |
| 后3点q | 18.802199601352033 | 18.956797060223437 |
| 最大q | 6.908710040755056 | 6.977780135730144 |
| 达标caps | 18/24 | 18/24 |
| 数值prefix | 9 | 9 |
| 最小安全slack | 2.2355713308163245e-8 | 8.717565562310625e-6 |
| reserve | 0.00012355713308163243 | 0.8707565562310624 |

总q减少2.152051673104417，前九点减少2.306649131975819，后三点增加
0.15459745887140386。最终reserve虽提高，仍低于原进入pose的0.9门。
不能仅以总误差下降选作合格执行轨迹。

保存结果上的cap-distance诊断值从2.23068628329355升至2.2753234046237307；
这只是计算已保存q的标量，没有执行cap-distance搜索、不能充当该方法的结果。
旧预测器的失败及其52QP等费用另存，本轮轨迹和55QP不能混写成旧记录复现。

## 证据与软件检查

| 内容 | 当前仓库相对路径 | SHA256 |
| --- | --- | --- |
| 总结果 | outputs/replay_order_sumq_20261102/result.json | 0bcc9962caabb8f74f92113a8994cfdb41e7e9a1f173781c4d20d603fbcb9ecc |
| 完整生产结果 | outputs/replay_order_sumq_20261102/producer_final.json | 61810393d711056f81503ff37e140dfb4488472c38a36ed9f5ee4529f22ef974 |
| 独立checker | outputs/replay_order_sumq_20261102/checker.json | b6fee0e456e1c59410cc92de69c845ea836fa17dfb7f06058fec08f587b29da0 |
| 保存结果分解 | outputs/replay_order_sumq_20261102/saved_analysis.json | 1fd58232a610b0fada934c9c1144f8156a436692bd3ce2f7b63f5bbfb082642d |
| 61项专项 | outputs/qa/replay_order_sumq_20261102/result.json | dfd8770b2dc5e65adac0c4aac8c7261d468b39993586ce32d265692347d9412d |

61项专项使用CPU runner、jobs32，全部通过，无skip/collection错误，worker均
实际回收、来源前后不变。覆盖新准入、stencil/比较器以及原dual-recovery算法
与审核；Ruff/format和Git空白检查通过，未运行全仓回归。实现提交7b2319b。

新配置和预登记分别为`replay_order_sumq_20261102.json`及同名计划；入口是
`scripts/tracking/trajectory_search/check_replay_order_sumq.py`。保存分析未执行
任何模型/查询/拟合/物理。新初始化来源见`replay_order_stencil_20261102.md`；
下一阶段机制与原验收边界仍按`cap_distance_gn_design_20261029.md`执行。
