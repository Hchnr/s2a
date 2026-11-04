# 固定旋转方向的局部性诊断（2026-11-02，Asia/Shanghai）

上一轮单方向结果SHA256：
7d5bd9502573cf122208a9c146c3baac8aaa62991f03688716fd58841238bdc8。
方向使末端旋转改善，但row4358线性预测raw slack约+4.995e-6，完整响应约-8.711e-5。
本轮区分已保存Js沿该方向的局部斜率失配与有限步曲率，不改安全门、不重新求解QP。

固定anchor85及已收费trial的实际command差d，仍从actual368自主预测完整H12。
注册顺序：[anchor, x+.5d, x+.25d, x+.125d, x+.0625d,
 x+(1e-4/||d||inf)d, x+(5e-5/||d||inf)d, anchor_repeat]。
原alpha1响应只作为明确标注的历史端点，不冒充本轮新执行。两种小扰动为预先
固定的单侧诊断尺度，不用测得结果选h或拟合新控制模型；不估计未采样方向的导数。

全部case事前核对原box/rate/global/trust和非零唯一（repeat除外），不裁剪、不
补选、不重排。保留相同原参考/控制接口/时间/安全与pose caps、native_solve TCP
和replay-order响应。role/alpha、实际command差及来源SHA执行前登记。

预算：生产单model8完整响应（含一次初态查询）2497query/2400内步；独立单model
同8响应/2497query/2400内步；合计16响应/4994query/4800内步/2model。
新增QP、NNLS、FD矩阵拟合、物理、GPU/WAM、fresh或最终留出均0。费用单列，
不追加旧探针/搜索预算。单模型复用以保持确定初态与低构造成本；专项QA jobs32。
先核对QA全通过/无skip及当前源码内容，先写registration，独占outputs。
整轮timeout600秒、超时10秒终止回收，无自动重试。CPU数值库线程1、CUDA/LD空。

完整initial/repeat与全部原native独立响应/审核、numeric1e-10、torque1e-8保持，
评分布尔精确。记录所有q/caps/安全及signed误差，不能只报道最小安全alpha。
逐row比较 s(x+alpha*d)-s(x) 与旧Js乘实际step，报告两小尺度的一侧斜率及其
相互差，与旧模型的偏差；不从两尺度宣称完整Jacobian或严格极限已验证。
若小尺度相近且偏离旧模型，支持该方向局部模型陈旧；若小尺度接近旧模型而
大步失配，支持有限步曲率；数值量化/不一致则保留未分辨，不按后验阈值选结论。
任何安全失败或caps失败候选均不下发，不授予控制资格；没有完整witness不进物理。
