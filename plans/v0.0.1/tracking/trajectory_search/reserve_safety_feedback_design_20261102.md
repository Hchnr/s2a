# 下限对齐后的安全reserve拒绝反馈设计

2026-11-02。基于已关闭aligned实际搜索与新后验检查，不修改旧失败记录。
目标仍是原完整可靠跟踪与稳定验收；GPU禁用。

后验使用首个有reserve拒绝且完全未接纳的stage5，按最小alpha选择安全response13，
更新Js后明显改善该方向其他已见尺度及部分后续方向；Jq在最远后续方向反而变差。
因此下一变体只对reserve反馈更新Js，不同时改变Jq/Je，不把后验效果当作新执行。

## 显式选择及更新合同

仅在stage mode=pose、无任何接纳且原choose_rejected_source没有合资格来源时，
才尝试ReserveSafetyFeedback。保留原pose ceiling失败反馈的优先级和全3模型更新，
不把旧反馈偷偷改成新规则。默认None保持原行为与结果schema。

新的候选必须是已实际计费且未接纳、reason=pose_reserve_below_entry的响应，
joint/buffer安全、q≤原pose ceiling、非零step且squared norm≥原secant minimum，
实际reserve<原entry且模型预测reserve≥原entry。预测门严格比较，不加epsilon；
这排除旧半reserve结构已经预言不合格的响应，不以该反馈修饰旧内部不一致。
按(alpha,response_index)取最小，不按误差/残差或目标下降挑选来源。

只沿该已观察step对Js执行一次原bounded causal secant，原clip/minstep参数保持。
Jq/Je逐值不变，拒绝响应不成为anchor、不改变stage mode或reserve。stage仍按原
失败流程缩半radius。新record显式kind=reserve_safety_only，pose/error更新记None，
不能伪造已执行的零更新；单列策略tag与实际次数。未来列继续严格0。

原实际安全、完整witness、reserve接纳、最小下降、ratio、QP/NNLS证书、全局/root
域、pose ceiling、全部24caps、原stage/response预算均保持。只用已付费响应，
没有新方向补样、查询或隐含QP；任何新实验预算另登记。

## 独立审核与验证

独立审核自行重建来源资格/优先级和最小alpha，不调用生产选择函数；额外策略
metadata必须显式启用审核，默认审核拒绝未知策略。新source的Js独立重建，
Jq/Je不更新且record None；原pose ceiling反馈照原公式三模型重建。
cap scaling/recovery遍历和任何后续模型重建也必须识别安全only kind。

测试来源筛选、已接纳stage拒绝、原反馈优先、无新response/QP费用、严格预测门、
Js因果/clip、Jq/Je不变及源/record篡改拒绝。旧default完整闭合回放答案保持。
先用保存stage5资料核验选择response13及更新，再另登记有限实际比较；不能
把后验/回放当作新控制证据。大步残差和更远方向风险仍由完整新响应审核兜底。
