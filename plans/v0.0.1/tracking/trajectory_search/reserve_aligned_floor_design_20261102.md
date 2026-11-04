# 对齐pose优化下限与实际reserve接纳条件

2026-11-02。基于已关闭实际对照与后验门槛核查，不改旧默认或原失败记录。
目标仍是完整可靠跟踪和原稳定验收，GPU禁用。

新显式cap-distance约束策略：pose阶段使用
`new_floor = maximum(original_floor, buffer + reserve_entry_fraction*reserve_target/weights)`，
weights保持原terminal time scale，其他行权重为1；restore阶段完全保持原问题。
因此新floor只收紧，不放宽原安全门。当前pose anchor若reserve≥entry，零步仍
满足新reserve线性下限；不强求从.928直接恢复至1.0，也不重置搜索域。

不改变完整cap-distance B/g、投影算术、原输入/速率/global/trust、pose ceiling、
原实际joint/buffer、下降/ratio、完整witness、reserve实际接纳或拒绝反馈规则。
仅更新原safety行b=safety-new_floor，A和其他行不变；策略显式tag，默认不启用。
原terminal权重单位保持，不能把所有raw slack误用同一阈值。

独立审核从原公式分别重建old/new floor及完整矩阵，不调用生产floor helper。
必须拒绝篡改floor/b、未显式登记的策略、错误terminal换算，以及metadata冒充旧
默认；旧default闭合回放仍需保持完整答案哈希。新模型不因矩阵合格自动获资格。

先实现数学/审核并测原restore不变、pose只收紧、零步可行、terminal单位、其他
QP行/目标不变；再单独登记完整独立响应的新方向探针及有限迭代预算。即使模型
预测reserve≥.9，实际响应仍可能低于门，必须原样拒绝，不降低实际门兜底。
是否改善停滞、跟踪、恢复次数或费用由新实际结果决定；不靠训练/后验数据宣称。
