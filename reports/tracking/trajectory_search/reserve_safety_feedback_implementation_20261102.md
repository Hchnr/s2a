# 仅Js的安全reserve拒绝反馈：实现与闭合验证

2026-11-02，设计66cd45e，实现2883a39。完成可选策略、生产状态机、独立审核和
模型重建接线；尚未执行启用反馈的新连续搜索，不能宣称已经解决停滞。

## 显式新增行为

ReserveSafetyFeedback只在pose且stage无接纳时选择已计费reserve拒绝。原pose
ceiling反馈先选择，只有原规则无来源才尝试新规则，默认None保持原行为/schema。
候选须joint/buffer安全、满足原pose ceiling、非零且达到原min step，实际reserve
低于entry但模型预测≥entry（严格比较，无epsilon，NaN不合格）。固定取最小
alpha再response index，不按残差或目标下降挑选。

新来源仅沿实际step对Js做原bounded causal secant，Jq/Je不动、anchor不动，
stage仍按原失败流程缩半radius。record显式kind=reserve_safety_only，pose及
error_vector更新为None，不伪造零更新。原pose ceiling来源仍执行全部三模型更新。
新增reserve_feedback_policy单列，不冒充旧默认；响应/求解预算、实际门、完整
witness、目标、root域、下降/ratio与反馈优先级均保持。

独立审核不调用生产选择器，自行重建资格、原来源优先级和最小alpha，检查安全
only record及Js更新；默认审核拒绝新metadata。cap scaling/recovery遍历和后续
reconstruct_models同样识别kind，保证未来QP不偷偷更新Jq/Je。未来列仍严格0。

## 实际验证

最终52项统一CPU jobs32全部通过，0skip/collection错误、worker回收、来源不变。
覆盖来源各拒绝边界/NaN/严格预测门、原优先级、已接纳stage、restore、Jq/Je及
anchor不变、真实调用预算，以及旧完整三模型反馈审核。Ruff/format和Git空白
通过，未运行全仓回归。首次51项中1条测试在下一unqualified stage仍错误禁止
调用新选择器；将优先级测试限制到有旧来源的stage后最终轮通过，未改变生产
规则或数值容差，首次失败记录保留。

### 旧baseline完整软件回放

60个已保存QP、91响应的完整回放结果和独立ledger通过，答案SHA保持
e3f6690e754d8595720e024ef1c5709cb28d8dcfddecf75d9270c34b7c17728b。
新参数默认None没有改变旧结果；回调只是保存数据读取，不计新QP或响应执行。

### 新来源的六stage闭合回放

使用已关闭aligned actual数据的stage0–5和原root域，保留该前缀的全部保存QP、
响应与接纳，只在最后全拒绝stage启用新反馈。逻辑6QP/15响应回调，实际新增
QP/NNLS/机器人响应/查询均0。选中response13、alpha .25，与独立选择及后验设计
一致。Js改变，Jq/Je在反馈前后逐值相同，final anchor仍response10。

完整独立ledger/GN/KKT/scaling/recovery及来源/更新审核通过；篡改source、pose
更新、kind、safety theta或未显式启用新审核时全部拒绝。后续模型重建也确认仅Js
改变。该前缀恰在反馈后结束，没有验证新模型生成的下一方向，不能当成连续
收敛或新执行证据。合成测试另检查下一QP目标及非安全行保持、只有safety行变化。

## 计费与后续

本轮0新QP/NNLS/机器人响应/动力学查询/物理/GPU/WAM/fresh/最终留出，旧响应
回放及模型更新的复现单列。全程CPU空CUDA/LD、数值库线程1，未启采集owner。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 新反馈前缀回放 | reserve_safety_feedback_implementation_20261102/prefix_result.json | 5e7dd76bb56c1ae3de011b05b9981ec88de05f5467e30e03adb585d4b3b6c4a7 |
| 旧baseline回放 | reserve_safety_feedback_implementation_20261102/baseline_replay.json | 3f7f9e1eba2362a1c12a7e0265049089e31a666fc62f65ac71207fbc2c7ce9ab |
| 最终52项QA | qa/reserve_safety_feedback_final_20261102/result.json | 7878f188b88aaf68ee627be1718100ce7179ac5cf419b56291860344be435c7c |

下一步独立登记启用此反馈的新有限实验，保持原已对齐floor/实际接纳及匹配根域，
全部新响应重新native审核，记录是否减少停滞或改善完整24caps，不追加旧预算。
原可靠跟踪与18/12/24稳定验收仍未完成，最终留出保持关闭，GPU继续禁用。
