# reserve进入条件接纳变体：实现与闭合回放

2026-11-02。设计2aea4da，保持全局目标和原18/12/24验收，GPU继续禁用。
本轮完成显式策略、继承域、独立审核及软件回放；未执行新的有限搜索或机器人响应。

## 生产与独立审核

`reserve_entry_acceptance.py` 提供ReserveEntryAcceptance：只对非完整witness的pose
实际响应增加reserve≥原reserve_entry_fraction门；restore与完整原buffered witness
保持原规则。objective_search_driver通过可选pose_acceptance显式启用，默认None。
新拒绝单列pose_reserve_below_entry，仍实际计费，按原alpha顺序继续；原安全、
pose ceiling、ratio/minimum下降、缩步、反馈资格、预算和repeat不变。

InheritedSearchDomain绑定原根commands和q，按原公式重建全局lower/upper和common
pose ceiling，从anchor85继续时不重新居中或扩大域。独立审核不调用生产域/策略，
而由调用方给定原root entry独立重建，逐值比较metadata、实际域与接纳理由。
默认审核拒绝未经显式开启的策略/域metadata，避免把新变体伪装为原默认。

## 默认完全一致回放

用已关闭sum-q的55个QP及122完整响应回调，分别重放原默认和当前显式objective
driver。所有B/g/A/b、参数及完整结果完全一致，完整答案SHA仍为
26c0fcd1286c08a503ca8899007416f5cefdab2e6fd0401a7da37402fa85a258。
这是软件回放，没有新增QP/NNLS/机器人响应或查询；旧来源及结果未覆盖。

## 新策略单stage回放

固定刷新模型的已审核QP/七个完整响应，由原全局根entry重建域，maxstage1、
maxresponse7仅作为这次软件回放的有限合同，不冒充已执行新模型实验。
QP输入逐值匹配已保存登记，不调用优化器；响应只允许匹配已保存command。

| 项目 | 原接纳策略 | reserve接纳变体 |
| --- | ---: | ---: |
| 接纳alpha | .5 | .125 |
| 接纳后reserve | .8546033809 | .9285845109 |
| 下一原模式 | restore | pose |
| 逻辑响应回调（含initial/repeat） | 4 | 6 |
| 逻辑QP回调 | 1 | 1 |

变体依次记录actual_reduction_insufficient、pose_reserve_below_entry两次、
accepted_reduction；原根lower/upper/ceiling与历史cap搜索逐值一致。全部独立
ledger/GN/KKT/scaling/recovery审核通过。分别篡改接纳理由、root metadata或
pose ceiling时独立审核均拒绝。未改变旧只限pose ceiling失败的反馈集合。

这里多出的回调是已保存数据读取，不是新响应计费；真实新实验必须付出这些
被拒绝候选的实际费用。一次软件选步正确不能证明连续迭代、避免停滞或完整跟踪。

## 验证与证据

29项统一CPU jobs32专项全部通过，0skip/collection错误，worker回收、来源前后
不变；覆盖策略/默认、restore、完整witness、原root域与越界拒绝，及原objective/
audit等。Ruff/format及Git空白通过；未运行全仓回归。全程空CUDA/LD、数值库
线程1。0新QP/NNLS/机器人响应/动力学查询/物理/GPU/WAM/fresh/最终留出。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 默认闭合等价 | reserve_entry_acceptance_implementation_20261102/sumq_equivalence.json | 771ce6f2f64dd7c9f6f9ae6907e037e646437dba3be4902dc625222378fe32b4 |
| 单stage策略回放 | reserve_entry_acceptance_implementation_20261102/cap_replay_result.json | 6e70d8816a3d7625abf0dddf6a2b02aa6d84653b7ded1137680c7b2539b06922 |
| 29项QA | qa/reserve_entry_acceptance_20261102/result.json | 94c4235ddbcd147dee19556fd9d0c97a9d477c191ff53c1ddc32a68d5d238c38 |

下一步单独登记新变体有限CPU迭代，绑定anchor85的新模型及原根域；所有新响应
重新完整独立审核。记录接受/拒绝、实际预算、恢复次数与停滞，不将避免restore
本身当作优化成功。完整24caps、物理及留出任务验收仍未完成。
