# 更小算法进展门的有限对照设计

2026-11-02。固定未调用方向的新响应显示：stage11实际安全、reserve .901292，
真实cap loss下降3.097e-5，但预测/实际下降均小于旧1e-4。其他两个方向reserve
失败，说明仍必须完整响应审核，不能仅按正预测接纳。

下一对照只将minimum_predicted_reduction和minimum_actual_reduction由1e-4
改为预先固定1e-5，其余search config逐值保持。这个值低于已验证的3.097e-5，
是明确登记的算法停止/接纳策略，不是修改原跟踪/安全资格或追认旧搜索通过。
不反复按最终留出调参，不自动继续旧运行或使用旧未消耗预算。

使用同一anchor85、刷新Js/Jq/Je、原root global/pose ceiling、aligned reserve floor、
实际reserve接纳及仅Js反馈。仍60stage/122response，所有新响应/拒绝/initial/
repeat计费，独立native与ledger复算、原数值容差、owner回收、CPU/GPU约束保持。
新配置及policy显式标识，执行入口只允许两项准确变化，默认原配置不变。

主要比较完整caps/prefix/witness及状态误差，不把接纳数或微小loss下降当作成功。
原实际joint/buffer、.9 reserve、.01m/.05rad全24caps、原最终20/24且每task4/8
完全保持；因此降低算法进展门仍不能接纳那些真实reserve不足的方向。

先验证配置差异白名单及原默认闭合回放，再另写实际预算和指纹登记后执行。
若仍只是更多小幅下降而完整跟踪不改善，需转向完整瓶颈/控制资格审查，不继续
仅按标量loss挑选变体。最终留出保持关闭，GPU禁用。
