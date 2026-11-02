# 末端旋转与活动约束单方向诊断（2026-11-02，Asia/Shanghai）

## 依据与可证伪问题

cap-distance已关闭且无witness：结果SHA256
80f58cde84778d1e3b2f7ff3ca0b85c5414419da976a057710421d23f03012e9。
保存模型重建SHA256 f22f8e44e53d3dc7fd4f2f8f250feb6fafef02fe55fd4f9c10dc3f6c1245ef76。
stage58为最后合格pose，anchor85；其安全行A.T*mu贡献inf约5.097，command bound
约.0101，27/29已接受pose之后reserve低于原.9切换门，旋转误差仍约.131rad。
这不能证明不可达，也不足以把失败归因于global .1边界或Je失真。

问题：在该已收费anchor/原局部模型、同hard rows和同trust/global盒内，若仅给
第12点旋转组（0起组23）原cap-distance梯度，能否产生经完整响应验证的旋转改善？
这是后验诊断，不是新完整方法对照，不重置cap实验预算，不把候选派发到控制。

## 固定算术与约束

从新初始化/72探针和已关闭cap账本重建stage58之前raw Js/Jq/Je，核对原stage58
完整B及原证书。原初态368/367因果历史、原H12参考、replay-order响应和native_solve
TCP评分保持。原global_lower/upper、stage58 radius=.005、原command/rate、全部
关节/制动/terminal、全24组原pose ceiling及所有物理caps都不变。

只把pose QP的B/g替换为组23投影残差rR与JR的 B=I+JR.T@JR、g=JR.T@rR。
原damping1、完整约束、objective scaling、单次tight-subset NNLS与KKT1e-6保持。
alpha固定1，无缩线/重求解/调权。方向须先通过原命令box/rate/global/trust、
非零/唯一和正预测下降；不合格则跳过trial并如实记录，不能追认为改善。

## 唯一有限预算与审核

先登记代码/输入/目标及约束SHA。最多1QP、100迭代、1NNLS（原full-active数量
3倍迭代，无fallback）。生产[anchor, optional trial, anchor_repeat]最多3完整响应，
独立侧复算相同全部响应最多3；总最多6响应/1800内步/1874查询/2model。
无需新FD。所有成本另计，不追加cap或sum-q旧预算。0物理/GPU/WAM/fresh/最终留出。
CPU数值库线程1，CUDA/LD空，整轮600秒timeout、终止后10秒强制回收，无自动重试。

独立重建完整约束与旋转B/g、原单位证书/缩放/紧子集，并逐条native复算、原
numeric1e-10/torque1e-8/精确布尔/initial-repeat；逐字段记录原caps、安全和
原旋转cap loss的预测/实际下降。报告其余组变差，不用单组改善代替完整witness。
主进程由执行工具实际wait/reap登记；超时或任一合同失败保留False和已知费用。
若trial安全失败、其余caps失败、模型方向预测错误，都不能下发或宣称完整跟踪。
