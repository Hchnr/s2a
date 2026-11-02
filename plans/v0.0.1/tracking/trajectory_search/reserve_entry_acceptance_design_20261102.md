# 保护原reserve进入条件的显式接纳变体

2026-11-02，接续刷新模型新方向的完整独立证据。总目标及原18/12/24和稳定验收
保持；GPU禁用。该变体不放宽原安全门，也不将局部下降替代完整witness。

## 变化范围

在objective_search_driver上增加可选、显式的pose接纳策略。默认None逐表达式保持
原行为及结果schema；新的ReserveEntryAcceptance仅在pose且不是完整原witness时，
要求实际完整响应的reserve≥原config.reserve_entry_fraction（.9）。
原command预检、QP/NNLS证书、实际joint/buffer/pose ceiling、最小下降及ratio均
继续适用。restore模式不加此门，完整原witness仍按原规则接纳。新拒绝原因单独
记pose_reserve_below_entry，不把它混作安全失败或低ratio。

所有实际响应均照常计费，继续按原alpha顺序试更小尺度。新拒绝不进入旧的仅限
pose ceiling失败的rejected-source更新集合；不混入新的反馈机制。若全部失败，
原缩半radius、停止门及repeat保持，允许如实报告停滞，不能为避免恢复而伪造进展。

## 不重新扩大全局域

从anchor85启动新有限搜索时，不能围绕新anchor重置±.1或重新计算更宽pose ceiling。
增加显式InheritedSearchDomain，绑定原根entry的commands与normalized squared errors，
由原公式计算全局lower/upper及common初始maximum ceiling；不允许随意传更松边界。
源entry及SHA由运行入口登记。当前anchor仍须满足原box/rate/global和buffer。
默认无domain时保持原从initial建立边界的行为。

## 独立审核与分步验证

生产driver记录额外策略和继承域metadata，仅显式变体有新字段。独立审核从调用方
提供的原根entry重建边界/ceiling，并独立按实际reserve重建拒绝顺序，不调用生产
策略或域构建函数。默认审核不得悄悄接受未启用审核的变体metadata。

先测试有下降但reserve不足时继续较小alpha、原默认不变、restore不加门、完整
witness仍保留原资格、真实root域继承而非重心平移、较松/错误继承metadata拒绝。
用保存55QP/122响应的sum-q闭合软件回放重新核对默认结果完全一致，不新执行QP。
再用保存的新方向及已审QP做单stage闭合软件回放，验证新变体按原顺序拒绝.5/.25
并接纳.125；明确这是旧响应重放，不充当新响应/任务证据。

最后另登记新有限CPU实验的seed、stage/response/QP预算、起点模型、原根域SHA及
所有新响应独立审核。原60stage/122response不能因探针/迁移自动追加；新变体独立
计费。模型/源码前后指纹、owner超时及实际wait/reap保持，最终留出关闭。
